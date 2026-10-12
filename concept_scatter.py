bl_info = {
    "name": "Concept Scatter",
    "author": "Claude",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Concept > スキャッター",
    "description": "UEのフォリッジのように、塗った所にアセットを散らす（地面ペイントのレイヤーを範囲に使える）",
    "category": "Object",
}

import bpy
import gpu
import math
import time
import uuid
import traceback
import mathutils
import numpy as np
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from bpy.props import (StringProperty, FloatProperty, IntProperty, BoolProperty,
                       EnumProperty, PointerProperty, CollectionProperty)
from bpy.app.handlers import persistent

MOD_NAME = "CSC_Scatter"
TREE_PREFIX = "CSC_Scatter_"
CGP_UV = "CGP_UV"
GROUP_VER = 1
INTERNAL = "csc_internal"      # custom prop on our own collections / objects
ID_ATTR = "csc_id"             # stable per-point id stored right after distributing
BRUSH_FMT = 2                  # brush image format: R = grow, G = erase (v0.2 images: R = weight, G, B = softness / noise)
MAX_ASSETS = 32
DEFAULT_NAME = "スキャッター"

_BUSY = [0]


# ====================================================================================
#  Placement core (node building).  Nothing in this section knows about the UI or the
#  PropertyGroups: build_scatter_tree(tree, spec) + push_values(tree, values) are meant
#  to be reused by other tools (e.g. the planned corner-clutter tool).
# ====================================================================================
def _sock(coll, name, typ=None):
    for s in coll:
        if s.name == name and s.enabled and (typ is None or s.type == typ):
            return s
    raise KeyError("socket not found: %s" % name)


class _GB:
    """Small helper for building geometry node trees from code."""

    def __init__(self, tree):
        self.tree = tree
        self.N = tree.nodes
        self.L = tree.links
        self.x = 0
        self.y = 0

    def col(self, x=None):
        self.x = self.x + 240 if x is None else x
        self.y = 0

    def node(self, idname, name=None, hide=True, **props):
        n = self.N.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        if name:
            n.name = n.label = name
        n.location = (self.x, self.y)
        n.hide = hide
        self.y -= 40 if hide else 160
        return n

    def link(self, v, sock):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.L.new(v, sock)
        else:
            sock.default_value = v

    def math(self, op, a, b=None, c=None, clamp=False, name=None):
        n = self.node("ShaderNodeMath", name, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.link(v, n.inputs[i])
        return n.outputs[0]

    def vmath(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self.link(a, n.inputs[0])
        if b is not None:
            self.link(b, n.inputs[1])
        if scale is not None:
            self.link(scale, _sock(n.inputs, "Scale"))
        if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE'):
            return _sock(n.outputs, "Value")
        return _sock(n.outputs, "Vector")

    def value(self, name, v=0.0):
        n = self.node("ShaderNodeValue", name)
        n.outputs[0].default_value = v
        return n.outputs[0]

    def integer(self, name, v=0):
        n = self.node("FunctionNodeInputInt", name)
        n.integer = int(v)
        return n.outputs[0]

    def rand(self, lo, hi, seed, offset, id_sock=None):
        n = self.node("FunctionNodeRandomValue", data_type='FLOAT')
        self.link(lo, _sock(n.inputs, "Min", 'VALUE'))
        self.link(hi, _sock(n.inputs, "Max", 'VALUE'))
        self.link(self.math('ADD', seed, float(offset)), _sock(n.inputs, "Seed"))
        if id_sock is not None:
            self.link(id_sock, _sock(n.inputs, "ID"))
        return _sock(n.outputs, "Value", 'VALUE')

    def xyz(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            self.link(v, n.inputs[i])
        return n.outputs[0]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.link(v, n.inputs[0])
        return n.outputs

    def smooth(self, v, a, b, lo=0.0, hi=1.0):
        n = self.node("ShaderNodeMapRange")
        n.data_type = 'FLOAT'
        n.interpolation_type = 'SMOOTHSTEP'
        n.clamp = True
        for i, x in enumerate((v, a, b, lo, hi)):
            self.link(x, n.inputs[i])
        return n.outputs[0]

    def compare(self, dtype, op, a, b):
        n = self.node("FunctionNodeCompare", data_type=dtype, operation=op)
        t = 'INT' if dtype == 'INT' else 'VALUE'
        self.link(a, _sock(n.inputs, "A", t))
        self.link(b, _sock(n.inputs, "B", t))
        return n.outputs[0]

    def boolean(self, op, a, b=None):
        n = self.node("FunctionNodeBooleanMath", operation=op)
        self.link(a, n.inputs[0])
        if b is not None:
            self.link(b, n.inputs[1])
        return n.outputs[0]

    def group(self, g, name=None):
        n = self.node("GeometryNodeGroup", name, hide=False)
        n.node_tree = g
        return n


def _new_group(name):
    g = bpy.data.node_groups.get(name)
    if g is not None and g.get("csc_ver") == GROUP_VER:
        return g, False
    if g is None:
        g = bpy.data.node_groups.new(name, "GeometryNodeTree")
    g.nodes.clear()
    g.interface.clear()
    g["csc_ver"] = GROUP_VER
    return g, True


def _group_io(g, ins, outs):
    for name, typ, dv in ins:
        s = g.interface.new_socket(name=name, in_out='INPUT', socket_type=typ)
        if dv is not None:
            s.default_value = dv
    for name, typ in outs:
        g.interface.new_socket(name=name, in_out='OUTPUT', socket_type=typ)
    b = _GB(g)
    gi = b.node("NodeGroupInput", hide=False)
    gi.location = (-500, 0)
    go = b.node("NodeGroupOutput", hide=False)
    go.location = (1400, 0)
    b.col(-250)
    return b, gi.outputs, go.inputs


def ensure_paint_mask():
    """Decodes a Ground Paint layer image exactly like the ground material does.
    R = weight, G = softness * weight, B = noise * weight (premultiplied)."""
    g, fresh = _new_group("CSC_PaintMask")
    if not fresh:
        return g
    b, I, O = _group_io(g, [("UV", 'NodeSocketVector', None), ("Image", 'NodeSocketImage', None),
                            ("Opacity", 'NodeSocketFloat', 1.0), ("Noise", 'NodeSocketFloat', 0.5)],
                        [("Mask", 'NodeSocketFloat'), ("Weight", 'NodeSocketFloat')])
    tex = b.node("GeometryNodeImageTexture", hide=False, interpolation='Linear', extension='EXTEND')
    b.link(I["Image"], _sock(tex.inputs, "Image"))
    b.link(I["UV"], _sock(tex.inputs, "Vector"))
    sp = b.node("FunctionNodeSeparateColor")
    b.link(_sock(tex.outputs, "Color"), sp.inputs[0])
    R, G, B = sp.outputs[0], sp.outputs[1], sp.outputs[2]
    b.col()
    rr = b.math('MAXIMUM', R, 1e-4)
    soft = b.math('MAXIMUM', b.math('DIVIDE', G, rr, clamp=True), 0.001)
    namt = b.math('DIVIDE', B, rr, clamp=True)
    w = b.math('MULTIPLY', R, I["Opacity"], clamp=True)
    # noise only moves the in-between weights: 0 stays 0 and 1 stays 1
    nzc = b.math('MULTIPLY', b.math('MULTIPLY', b.math('SUBTRACT', I["Noise"], 0.5), namt), 2.0)
    env = b.math('MULTIPLY', b.math('MULTIPLY', w, b.math('SUBTRACT', 1.0, w)), 4.0)
    w2 = b.math('ADD', w, b.math('MULTIPLY', nzc, env), clamp=True)
    b.col()
    half = b.math('MULTIPLY', soft, 0.5)
    m = b.smooth(w2, b.math('SUBTRACT', 0.5, half), b.math('ADD', 0.5, half))
    b.link(m, O["Mask"])
    b.link(w2, O["Weight"])
    return g


def ensure_brush_mask():
    """Decodes the foliage brush image. R = grow, G = erase (both 0..1).
    The edge noise only moves the in-between values: 0 stays 0 and 1 stays 1."""
    g, fresh = _new_group("CSC_BrushMask")
    if not fresh:
        return g
    b, I, O = _group_io(g, [("UV", 'NodeSocketVector', None), ("Image", 'NodeSocketImage', None),
                            ("Noise", 'NodeSocketFloat', 0.5), ("Amount", 'NodeSocketFloat', 0.5)],
                        [("Grow", 'NodeSocketFloat'), ("Erase", 'NodeSocketFloat')])
    tex = b.node("GeometryNodeImageTexture", hide=False, interpolation='Linear', extension='EXTEND')
    b.link(I["Image"], _sock(tex.inputs, "Image"))
    b.link(I["UV"], _sock(tex.inputs, "Vector"))
    sp = b.node("FunctionNodeSeparateColor")
    b.link(_sock(tex.outputs, "Color"), sp.inputs[0])
    b.col()
    nz = b.math('MULTIPLY', b.math('SUBTRACT', I["Noise"], 0.5), b.math('MULTIPLY', I["Amount"], 2.0))

    def edge(w):
        env = b.math('MULTIPLY', b.math('MULTIPLY', w, b.math('SUBTRACT', 1.0, w)), 4.0)
        return b.math('ADD', w, b.math('MULTIPLY', nz, env), clamp=True)

    b.link(edge(sp.outputs[0]), O["Grow"])
    b.link(edge(sp.outputs[1]), O["Erase"])
    return g


def ensure_range_mask():
    """1 inside [Min, Max], fading to 0 over Soft outside the range."""
    g, fresh = _new_group("CSC_RangeMask")
    if not fresh:
        return g
    b, I, O = _group_io(g, [("Value", 'NodeSocketFloat', 0.0), ("Min", 'NodeSocketFloat', 0.0),
                            ("Max", 'NodeSocketFloat', 1.0), ("Soft", 'NodeSocketFloat', 0.1)],
                        [("Mask", 'NodeSocketFloat')])
    s = b.math('MAXIMUM', I["Soft"], 1e-4)
    lo = b.smooth(I["Value"], b.math('SUBTRACT', I["Min"], s), I["Min"])
    hi = b.smooth(I["Value"], I["Max"], b.math('ADD', I["Max"], s), 1.0, 0.0)
    b.link(b.math('MULTIPLY', lo, hi), O["Mask"])
    return g


def ensure_camera_cull():
    """True for points inside the active camera's view (plus a margin)."""
    g, fresh = _new_group("CSC_CameraCull")
    if not fresh:
        return g
    b, I, O = _group_io(g, [("Position", 'NodeSocketVector', None), ("Tan X", 'NodeSocketFloat', 0.5),
                            ("Tan Y", 'NodeSocketFloat', 0.3), ("Margin", 'NodeSocketFloat', 0.1),
                            ("Pad", 'NodeSocketFloat', 1.0), ("Max Distance", 'NodeSocketFloat', 0.0)],
                        [("Visible", 'NodeSocketBool')])
    cam = b.node("GeometryNodeInputActiveCamera")
    oi = b.node("GeometryNodeObjectInfo", transform_space='RELATIVE')
    b.link(cam.outputs[0], _sock(oi.inputs, "Object"))
    inv = b.node("FunctionNodeInvertMatrix")
    b.link(_sock(oi.outputs, "Transform"), inv.inputs[0])
    tp = b.node("FunctionNodeTransformPoint")
    b.link(I["Position"], tp.inputs[0])
    b.link(inv.outputs[0], tp.inputs[1])
    x, y, z = b.sep(tp.outputs[0])[:3]
    b.col()
    d = b.math('MULTIPLY', z, -1.0)
    m1 = b.math('ADD', I["Margin"], 1.0)
    sx = b.math('SUBTRACT', b.math('ADD', b.math('MULTIPLY', b.math('MULTIPLY', d, I["Tan X"]), m1), I["Pad"]),
                b.math('ABSOLUTE', x))
    sy = b.math('SUBTRACT', b.math('ADD', b.math('MULTIPLY', b.math('MULTIPLY', d, I["Tan Y"]), m1), I["Pad"]),
                b.math('ABSOLUTE', y))
    sz = b.math('ADD', d, I["Pad"])
    unlimited = b.math('MULTIPLY', b.math('LESS_THAN', I["Max Distance"], 0.001), 1e9)
    sd = b.math('ADD', b.math('SUBTRACT', b.math('ADD', I["Max Distance"], I["Pad"]), d), unlimited)
    b.col()
    slack = b.math('MINIMUM', b.math('MINIMUM', sx, sy), b.math('MINIMUM', sz, sd))
    b.link(b.compare('FLOAT', 'GREATER_EQUAL', slack, 0.0), O["Visible"])
    return g


GROUP_NAMES = ("CSC_PaintMask", "CSC_BrushMask", "CSC_RangeMask", "CSC_CameraCull")


def ensure_groups():
    return ensure_paint_mask(), ensure_brush_mask(), ensure_range_mask(), ensure_camera_cull()


# names of the value nodes that push_values() writes
VALUE_NODES = ("P_density", "P_min_dist", "P_scale_min", "P_scale_max", "P_spin", "P_tilt", "P_align",
               "P_sink", "P_slope_min", "P_slope_max", "P_slope_soft", "P_h_min", "P_h_max", "P_h_soft",
               "P_noise_inv", "P_noise_t", "P_cull_margin", "P_cull_pad", "P_cull_dist", "P_cam_tx",
               "P_cam_ty", "P_vp", "P_paint_vp", "P_max", "P_cgp_nscale", "P_brush_nscale",
               "P_avoid_dist", "P_avoid_soft", "P_avoid_l_dist", "P_bbox_dist", "P_brush_noise", "P_overlay")


def build_scatter_tree(tree, spec):
    """Builds a scatter node tree.
    spec = {
      "surface": Object the points are spread on (its evaluated mesh, any transform),
      "assets":  [Object | Collection, ...]  (picked by the P_pick_<k> thresholds),
      "paint":   None | {"uv": uv map name, "target": None (= whole surface) | (Image|None, index),
                         "cover": [(Image, index), ...]  layers painted above that hide the target,
                         "invert": bool},
      "brush":   None | {"uv": uv map name, "image": Image}  foliage brush (R = grow, G = erase):
                 area = max(paint area, grow) * (1 - erase),
      "overlay": None | {"uv": uv map name, "image": Image, "material": Material}  viewport-only tint of
                 the brush image on the surface (shown while P_overlay > 0.5),
      "avoid":   None | Collection   keep away from these objects (P_avoid_dist / P_avoid_soft),
      "avoid_points": [Object, ...]  keep away from the instances of these scatter objects (P_avoid_l_dist),
      "slope", "height", "noise", "cull", "bbox": bool
    }
    All numbers live in named Value nodes (see VALUE_NODES, P_seed, P_pick_<k>, P_cgp_op_<i>)."""
    pm, bm, rm, cc = ensure_groups()
    ad = tree.animation_data
    if ad is not None:
        for fc in list(ad.drivers):
            ad.drivers.remove(fc)
    tree.nodes.clear()
    if "csc_brush_fmt" in tree:
        del tree["csc_brush_fmt"]
    tree.interface.clear()
    tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    b = _GB(tree)
    b.col(-1500)
    gi = b.node("NodeGroupInput", hide=False)
    b.col(-1250)
    P = {nm: b.value(nm) for nm in VALUE_NODES}
    seed = b.integer("P_seed", 0)

    # ---- points on the surface
    b.col(-1000)
    si = b.node("GeometryNodeObjectInfo", "surface", hide=False, transform_space='RELATIVE')
    _sock(si.inputs, "Object").default_value = spec.get("surface")
    dist = b.node("GeometryNodeDistributePointsOnFaces", "distribute", hide=False, distribute_method='POISSON')
    b.link(_sock(si.outputs, "Geometry"), _sock(dist.inputs, "Mesh"))
    b.link(P["P_min_dist"], _sock(dist.inputs, "Distance Min"))
    b.link(P["P_density"], _sock(dist.inputs, "Density Max"))
    b.link(seed, _sock(dist.inputs, "Seed"))
    st = b.node("GeometryNodeStoreNamedAttribute", data_type='INT', domain='POINT')
    b.link(_sock(dist.outputs, "Points"), _sock(st.inputs, "Geometry"))
    _sock(st.inputs, "Name").default_value = ID_ATTR
    b.link(b.node("GeometryNodeInputID").outputs[0], _sock(st.inputs, "Value", 'INT'))
    points = _sock(st.outputs, "Geometry")
    normal = _sock(dist.outputs, "Normal")
    pos = b.node("GeometryNodeInputPosition").outputs[0]
    na = b.node("GeometryNodeInputNamedAttribute", data_type='INT')
    _sock(na.inputs, "Name").default_value = ID_ATTR
    pid = _sock(na.outputs, "Attribute", 'INT')

    # ---- masks (probability that a candidate point is kept)
    b.col()
    mask = [None]

    def mul(v):
        mask[0] = v if mask[0] is None else b.math('MULTIPLY', mask[0], v)

    local_ = []

    def local_pos():
        """Point position in the surface object's own space (= the shader's Object coordinates)."""
        if not local_:
            inv = b.node("FunctionNodeInvertMatrix")
            b.link(_sock(si.outputs, "Transform"), inv.inputs[0])
            tp = b.node("FunctionNodeTransformPoint")
            b.link(pos, tp.inputs[0])
            b.link(inv.outputs[0], tp.inputs[1])
            local_.append(tp.outputs[0])
        return local_[0]

    def uv_attr(name):
        n = b.node("GeometryNodeInputNamedAttribute", data_type='FLOAT_VECTOR')
        _sock(n.inputs, "Name").default_value = name
        return _sock(n.outputs, "Attribute", 'VECTOR')

    def painted(img, uv, opacity, noise, name):
        if img is None:
            return 0.0
        gn = b.group(pm, name)
        b.link(uv, gn.inputs["UV"])
        gn.inputs["Image"].default_value = img
        b.link(opacity, gn.inputs["Opacity"])
        b.link(noise, gn.inputs["Noise"])
        return gn.outputs["Mask"]

    area = None   # where to scatter (Ground Paint layer and / or own brush)
    paint = spec.get("paint")
    if paint:
        uv = uv_attr(paint.get("uv", CGP_UV))
        nz = b.node("ShaderNodeTexNoise", noise_dimensions='3D')
        b.link(local_pos(), _sock(nz.inputs, "Vector"))
        b.link(P["P_cgp_nscale"], _sock(nz.inputs, "Scale"))
        _sock(nz.inputs, "Detail").default_value = 2.0
        nsep = b.node("FunctionNodeSeparateColor")
        b.link(_sock(nz.outputs, "Color"), nsep.inputs[0])

        def layer_mask(img, idx):
            # same edge noise channel as the ground material uses for this layer
            k = idx - 1
            ch = nsep.outputs[k % 3]
            if (k // 3) % 2:
                ch = b.math('SUBTRACT', 1.0, ch)
            return painted(img, uv, b.value("P_cgp_op_%d" % idx, 1.0), ch, "paint_%d" % idx)

        m = None
        tgt = paint.get("target")
        if tgt is not None:
            m = layer_mask(*tgt)
        for img, idx in paint.get("cover", ()):
            c = b.math('SUBTRACT', 1.0, layer_mask(img, idx))
            m = c if m is None else b.math('MULTIPLY', m, c)
        if paint.get("invert"):
            m = 0.0 if m is None else b.math('SUBTRACT', 1.0, m)
        area = m

    brush = spec.get("brush")
    if brush and brush.get("image") is not None:
        nzb = b.node("ShaderNodeTexNoise", noise_dimensions='3D')
        b.link(b.vmath('ADD', local_pos(), (17.13, 3.71, 9.27)), _sock(nzb.inputs, "Vector"))
        b.link(P["P_brush_nscale"], _sock(nzb.inputs, "Scale"))
        _sock(nzb.inputs, "Detail").default_value = 2.0
        gn = b.group(bm, "brush")
        b.link(uv_attr(brush.get("uv", CGP_UV)), gn.inputs["UV"])
        gn.inputs["Image"].default_value = brush["image"]
        b.link(_sock(nzb.outputs, "Fac"), gn.inputs["Noise"])
        b.link(P["P_brush_noise"], gn.inputs["Amount"])
        if area is not None:
            area = b.math('MAXIMUM', area, gn.outputs["Grow"])
        else:
            area = gn.outputs["Grow"]
        area = b.math('MULTIPLY', area, b.math('SUBTRACT', 1.0, gn.outputs["Erase"]))
        tree["csc_brush_fmt"] = BRUSH_FMT
    if area is not None:
        mul(area)

    if spec.get("slope"):
        z = b.sep(b.vmath('NORMALIZE', normal))[2]
        ang = b.math('ARCCOSINE', b.math('MINIMUM', b.math('MAXIMUM', z, -1.0), 1.0))
        gn = b.group(rm, "slope")
        b.link(ang, gn.inputs["Value"])
        b.link(P["P_slope_min"], gn.inputs["Min"])
        b.link(P["P_slope_max"], gn.inputs["Max"])
        b.link(P["P_slope_soft"], gn.inputs["Soft"])
        mul(gn.outputs["Mask"])

    if spec.get("height"):
        gn = b.group(rm, "height")
        b.link(b.sep(pos)[2], gn.inputs["Value"])
        b.link(P["P_h_min"], gn.inputs["Min"])
        b.link(P["P_h_max"], gn.inputs["Max"])
        b.link(P["P_h_soft"], gn.inputs["Soft"])
        mul(gn.outputs["Mask"])

    if spec.get("noise"):
        sf = b.math('MULTIPLY', seed, 7.31)
        v = b.vmath('ADD', b.vmath('SCALE', pos, scale=P["P_noise_inv"]),
                    b.xyz(sf, b.math('MULTIPLY', seed, 3.17), 0.0))
        nz2 = b.node("ShaderNodeTexNoise", noise_dimensions='3D')
        b.link(v, _sock(nz2.inputs, "Vector"))
        _sock(nz2.inputs, "Scale").default_value = 1.0
        _sock(nz2.inputs, "Detail").default_value = 3.0
        fac = _sock(nz2.outputs, "Fac")
        mul(b.smooth(fac, b.math('SUBTRACT', P["P_noise_t"], 0.05), b.math('ADD', P["P_noise_t"], 0.05)))

    def proximity(target, element):
        px = b.node("GeometryNodeProximity", target_element=element)
        b.link(target, _sock(px.inputs, "Geometry"))
        b.link(pos, _sock(px.inputs, "Sample Position"))
        # nothing to keep away from -> treat as very far
        return b.math('ADD', _sock(px.outputs, "Distance"),
                      b.math('MULTIPLY', b.math('SUBTRACT', 1.0, _sock(px.outputs, "Is Valid")), 1e9))

    if spec.get("avoid") is not None:
        b.col()
        ci = b.node("GeometryNodeCollectionInfo", "avoid", transform_space='RELATIVE')
        _sock(ci.inputs, "Collection").default_value = spec["avoid"]
        _sock(ci.inputs, "Separate Children").default_value = False
        _sock(ci.inputs, "Reset Children").default_value = False
        ri = b.node("GeometryNodeRealizeInstances")
        b.link(ci.outputs[0], ri.inputs[0])
        d = proximity(ri.outputs[0], 'FACES')
        mul(b.smooth(d, P["P_avoid_dist"], b.math('ADD', P["P_avoid_dist"], P["P_avoid_soft"])))

    if spec.get("avoid_points"):
        b.col()
        jp = b.node("GeometryNodeJoinGeometry")
        for k, o in enumerate(spec["avoid_points"]):
            oi = b.node("GeometryNodeObjectInfo", "avoid_layer_%d" % k, transform_space='RELATIVE')
            _sock(oi.inputs, "Object").default_value = o
            sc = b.node("GeometryNodeSeparateComponents")
            b.link(_sock(oi.outputs, "Geometry"), sc.inputs[0])
            ip = b.node("GeometryNodeInstancesToPoints")
            b.link(_sock(sc.outputs, "Instances"), _sock(ip.inputs, "Instances"))
            b.link(ip.outputs[0], jp.inputs[0])
        d = proximity(jp.outputs[0], 'POINTS')
        mul(b.smooth(d, b.math('MULTIPLY', P["P_avoid_l_dist"], 0.6), P["P_avoid_l_dist"]))

    # viewport-only thinning (display percentage, lighter while painting)
    b.col()
    isvp = b.node("GeometryNodeIsViewport").outputs[0]
    vpf = b.math('SUBTRACT', 1.0, b.math('MULTIPLY', isvp,
                                         b.math('SUBTRACT', 1.0, b.math('MULTIPLY', P["P_vp"], P["P_paint_vp"]))))
    mul(vpf)
    keep = b.compare('FLOAT', 'LESS_THAN', b.rand(0.0, 1.0, seed, 101, pid), mask[0])
    if spec.get("cull"):
        gn = b.group(cc, "camera_cull")
        b.link(pos, gn.inputs["Position"])
        for k, nm in (("Tan X", "P_cam_tx"), ("Tan Y", "P_cam_ty"), ("Margin", "P_cull_margin"),
                      ("Pad", "P_cull_pad"), ("Max Distance", "P_cull_dist")):
            b.link(P[nm], gn.inputs[k])
        keep = b.boolean('AND', keep, gn.outputs["Visible"])
    sg = b.node("GeometryNodeSeparateGeometry", domain='POINT')
    b.link(points, _sock(sg.inputs, "Geometry"))
    b.link(keep, _sock(sg.inputs, "Selection"))
    pts = _sock(sg.outputs, "Selection")

    # upper limit of the instance count
    ds = b.node("GeometryNodeAttributeDomainSize", component='POINTCLOUD')
    b.link(pts, ds.inputs[0])
    cnt = b.math('MAXIMUM', _sock(ds.outputs, "Point Count"), 1.0)
    ratio = b.math('ADD', b.math('DIVIDE', P["P_max"], cnt),
                   b.math('MULTIPLY', b.math('LESS_THAN', P["P_max"], 0.5), 1e9))
    keep2 = b.compare('FLOAT', 'LESS_THAN', b.rand(0.0, 1.0, seed, 102, pid), ratio)
    sg2 = b.node("GeometryNodeSeparateGeometry", domain='POINT')
    b.link(pts, _sock(sg2.inputs, "Geometry"))
    b.link(keep2, _sock(sg2.inputs, "Selection"))
    pts = _sock(sg2.outputs, "Selection")

    # ---- orientation / scale / pick
    b.col()
    spin = b.math('MULTIPLY', b.rand(0.0, math.tau, seed, 103, pid), P["P_spin"])
    base = b.node("FunctionNodeEulerToRotation")
    b.link(b.xyz(0.0, 0.0, spin), base.inputs[0])
    al = b.node("FunctionNodeAlignRotationToVector", axis='Z', pivot_axis='AUTO')
    b.link(base.outputs[0], _sock(al.inputs, "Rotation"))
    b.link(P["P_align"], _sock(al.inputs, "Factor"))
    b.link(normal, _sock(al.inputs, "Vector"))
    tx = b.math('MULTIPLY', b.rand(-1.0, 1.0, seed, 104, pid), P["P_tilt"])
    ty = b.math('MULTIPLY', b.rand(-1.0, 1.0, seed, 105, pid), P["P_tilt"])
    tilt = b.node("FunctionNodeEulerToRotation")
    b.link(b.xyz(tx, ty, 0.0), tilt.inputs[0])
    rr = b.node("FunctionNodeRotateRotation", rotation_space='LOCAL')
    b.link(al.outputs[0], _sock(rr.inputs, "Rotation"))
    b.link(tilt.outputs[0], _sock(rr.inputs, "Rotate By"))
    rot = rr.outputs[0]
    scale = b.rand(P["P_scale_min"], P["P_scale_max"], seed, 106, pid)
    # sink a little along the instance's own up axis
    rv = b.node("FunctionNodeRotateVector")
    b.link((0.0, 0.0, 1.0), _sock(rv.inputs, "Vector"))
    b.link(rot, _sock(rv.inputs, "Rotation"))
    off = b.vmath('SCALE', rv.outputs[0], scale=b.math('MULTIPLY', b.math('MULTIPLY', P["P_sink"], -1.0), scale))
    sp = b.node("GeometryNodeSetPosition")
    b.link(pts, _sock(sp.inputs, "Geometry"))
    b.link(off, _sock(sp.inputs, "Offset"))
    pts = _sock(sp.outputs, "Geometry")

    assets = [a for a in spec.get("assets", ()) if a is not None][:MAX_ASSETS]
    b.col()
    pick = b.rand(0.0, 1.0, seed, 107, pid)
    idx = None
    for k in range(len(assets) - 1):
        g = b.math('GREATER_THAN', pick, b.value("P_pick_%d" % k, (k + 1) / len(assets)))
        idx = g if idx is None else b.math('ADD', idx, g)
    far = None
    if spec.get("bbox"):
        # viewport only: far from the scene camera -> draw a wire box instead of the asset
        cam = b.node("GeometryNodeInputActiveCamera")
        coi = b.node("GeometryNodeObjectInfo", transform_space='RELATIVE')
        b.link(cam.outputs[0], _sock(coi.inputs, "Object"))
        dcam = b.vmath('DISTANCE', pos, _sock(coi.outputs, "Location"))
        far = b.boolean('AND', isvp, b.compare('FLOAT', 'GREATER_THAN', dcam, P["P_bbox_dist"]))

    def wire_box(geo):
        bb = b.node("GeometryNodeBoundBox")
        b.link(geo, _sock(bb.inputs, "Geometry"))
        dl = b.node("GeometryNodeDeleteGeometry", domain='FACE', mode='ONLY_FACE')
        b.link(_sock(bb.outputs, "Bounding Box"), _sock(dl.inputs, "Geometry"))
        return dl.outputs[0]

    def place(geo, sel):
        iop = b.node("GeometryNodeInstanceOnPoints")
        b.link(pts, _sock(iop.inputs, "Points"))
        if sel is not None:
            b.link(sel, _sock(iop.inputs, "Selection"))
        b.link(geo, _sock(iop.inputs, "Instance"))
        b.link(rot, _sock(iop.inputs, "Rotation"))
        b.link(scale, _sock(iop.inputs, "Scale"))
        b.link(iop.outputs[0], join.inputs[0])

    b.col()
    join = b.node("GeometryNodeJoinGeometry", hide=False)
    for k, a in enumerate(assets):
        box = None
        if isinstance(a, bpy.types.Collection):
            ci = b.node("GeometryNodeCollectionInfo", "asset_%d" % k, transform_space='ORIGINAL')
            _sock(ci.inputs, "Collection").default_value = a
            _sock(ci.inputs, "Separate Children").default_value = False
            _sock(ci.inputs, "Reset Children").default_value = False
            geo = ci.outputs[0]
            if far is not None:
                box = wire_box(geo)
        else:
            oi = b.node("GeometryNodeObjectInfo", "asset_%d" % k, transform_space='ORIGINAL')
            _sock(oi.inputs, "Object").default_value = a
            _sock(oi.inputs, "As Instance").default_value = True
            tr = b.node("GeometryNodeTransform")
            b.link(_sock(oi.outputs, "Geometry"), _sock(tr.inputs, "Geometry"))
            b.link(_sock(oi.outputs, "Rotation"), _sock(tr.inputs, "Rotation"))
            b.link(_sock(oi.outputs, "Scale"), _sock(tr.inputs, "Scale"))
            geo = _sock(tr.outputs, "Geometry")
            if far is not None:
                om = b.node("GeometryNodeObjectInfo", transform_space='ORIGINAL')
                _sock(om.inputs, "Object").default_value = a
                tb = b.node("GeometryNodeTransform")
                b.link(wire_box(_sock(om.outputs, "Geometry")), _sock(tb.inputs, "Geometry"))
                b.link(_sock(oi.outputs, "Rotation"), _sock(tb.inputs, "Rotation"))
                b.link(_sock(oi.outputs, "Scale"), _sock(tb.inputs, "Scale"))
                box = _sock(tb.outputs, "Geometry")
        sel = b.compare('INT', 'EQUAL', idx, k) if idx is not None else None
        if far is None:
            place(geo, sel)
        else:
            near = b.boolean('NOT', far)
            place(geo, near if sel is None else b.boolean('AND', sel, near))
            place(box, far if sel is None else b.boolean('AND', sel, far))
    ov = spec.get("overlay")
    if ov and ov.get("image") is not None and ov.get("material") is not None:
        # viewport only: a copy of the surface, nudged along its normals, tinted by the brush image
        b.col()
        show = b.boolean('AND', isvp, b.compare('FLOAT', 'GREATER_THAN', P["P_overlay"], 0.5))
        sw = b.node("GeometryNodeSwitch", "overlay_switch", input_type='GEOMETRY')
        b.link(show, _sock(sw.inputs, "Switch"))
        nrm = b.node("GeometryNodeInputNormal").outputs[0]
        osp = b.node("GeometryNodeSetPosition", "overlay_lift")
        b.link(_sock(si.outputs, "Geometry"), _sock(osp.inputs, "Geometry"))
        b.link(b.vmath('SCALE', nrm, scale=0.01), _sock(osp.inputs, "Offset"))
        sm = b.node("GeometryNodeSetMaterial", "overlay_material")
        b.link(_sock(osp.outputs, "Geometry"), _sock(sm.inputs, "Geometry"))
        _sock(sm.inputs, "Material").default_value = ov["material"]
        b.link(_sock(sm.outputs, "Geometry"), _sock(sw.inputs, "True", 'GEOMETRY'))
        b.link(_sock(sw.outputs, "Output", 'GEOMETRY'), join.inputs[0])
    b.col()
    go = b.node("NodeGroupOutput", hide=False)
    b.link(join.outputs[0], go.inputs[0])
    tree["csc_spec_paint"] = bool(paint)
    return tree


def push_values(tree, values):
    """Writes numbers into the named value nodes of a scatter tree (only when they changed)."""
    if tree is None:
        return
    N = tree.nodes
    ad = tree.animation_data
    for name, v in values.items():
        n = N.get(name)
        if n is None:
            continue
        if n.bl_idname == 'FunctionNodeInputInt':
            if n.integer != int(v):
                n.integer = int(v)
        else:
            s = n.outputs[0]
            if ad is not None and ad.drivers.find('nodes["%s"].outputs[0].default_value' % name):
                continue
            if abs(s.default_value - float(v)) > 1e-7:
                s.default_value = float(v)


def setup_camera_drivers(tree, scene):
    """Keeps P_cam_tx / P_cam_ty equal to tan(half field of view) of the scene camera."""
    for name, expr in (("P_cam_tx", "tan(ang*0.5)*min(1.0, rx/ry)"),
                       ("P_cam_ty", "tan(ang*0.5)*min(1.0, ry/rx)")):
        n = tree.nodes.get(name)
        if n is None:
            continue
        fc = n.outputs[0].driver_add("default_value")
        d = fc.driver
        d.type = 'SCRIPTED'
        for v in list(d.variables):
            d.variables.remove(v)
        for vn, path in (("ang", "camera.data.angle"), ("rx", "render.resolution_x"),
                         ("ry", "render.resolution_y")):
            var = d.variables.new()
            var.name = vn
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'SCENE'
            var.targets[0].id = scene
            var.targets[0].data_path = path
        d.expression = expr


# ====================================================================================
#  Layer data -> spec / values
# ====================================================================================
def _cgp(ground):
    return getattr(ground, "cgp", None)


def _cgp_index(ground, uid):
    cgp = _cgp(ground)
    if cgp is None or not uid:
        return -1
    for i, l in enumerate(cgp.layers):
        if l.uid == uid:
            return i
    return -1


def _layer_of(item):
    """The CSC_Layer that owns a property group item (layer itself or one of its assets)."""
    ground = item.id_data
    p = item.as_pointer()
    for l in ground.csc.layers:
        if l.as_pointer() == p:
            return ground, l
        for a in l.assets:
            if a.as_pointer() == p:
                return ground, l
    return ground, None


def _pick_thresholds(layer):
    w = [max(a.weight, 0.0) for a in layer.assets]
    tot = sum(w)
    if tot <= 0.0:
        w = [1.0] * len(w)
        tot = float(len(w)) or 1.0
    out, c = [], 0.0
    for x in w[:-1]:
        c += x
        out.append(c / tot)
    return out


def _painting(ground):
    return ground.mode == 'TEXTURE_PAINT'


def layer_spec(ground, layer):
    paint = None
    cgp = _cgp(ground)
    idx = _cgp_index(ground, layer.cgp_uid)
    if cgp is not None and idx >= 0:
        cover = []
        if layer.occlude_upper:
            for j in range(idx + 1, len(cgp.layers)):
                l = cgp.layers[j]
                if l.kind == 'MATERIAL' and l.image is not None:
                    cover.append((l.image, j))
        paint = {"uv": CGP_UV, "target": None if idx == 0 else (cgp.layers[idx].image, idx),
                 "cover": cover, "invert": layer.paint_invert}
    brush = overlay = None
    if layer.use_brush and layer.brush_image is not None and layer.brush_uv:
        brush = {"uv": layer.brush_uv, "image": layer.brush_image}
        overlay = {"uv": layer.brush_uv, "image": layer.brush_image,
                   "material": ensure_preview_material(layer, layer.brush_image, layer.brush_uv)}
    avoid = None
    if layer.use_avoid and layer.avoid_collection is not None and not _bad_collection(ground, layer.avoid_collection):
        avoid = layer.avoid_collection
    avoid_points = []
    if layer.avoid_layers:
        for other in ground.csc.layers:
            if other.as_pointer() == layer.as_pointer():
                break
            if other.obj is not None:
                avoid_points.append(other.obj)
    return {
        "surface": ground,
        "assets": [a.obj if a.obj is not None else a.coll for a in layer.assets],
        "paint": paint,
        "brush": brush,
        "overlay": overlay,
        "avoid": avoid,
        "avoid_points": avoid_points,
        "slope": layer.use_slope,
        "height": layer.use_height,
        "noise": layer.use_noise,
        "cull": layer.camera_cull,
        "bbox": layer.bbox_far,
    }


def _bad_collection(ground, coll):
    """A collection that contains the ground or our own scatter objects can't be used (self reference)."""
    if coll is None:
        return False
    for o in coll.all_objects:
        if o == ground or o.get(INTERNAL):
            return True
    return False


def _overlay_on(ground, layer):
    """Tint the active layer's brush image on the ground (always while the foliage brush is running)."""
    s = ground.csc
    if not (0 <= s.active_index < len(s.layers)) or s.layers[s.active_index].as_pointer() != layer.as_pointer():
        return False
    return s.show_overlay or _FP.get("ground") == ground.name


def layer_values(ground, layer):
    s = ground.csc
    v = {
        "P_seed": layer.seed,
        "P_density": layer.density,
        "P_min_dist": layer.min_distance,
        "P_scale_min": layer.scale_min,
        "P_scale_max": max(layer.scale_max, layer.scale_min),
        "P_spin": layer.rot_random,
        "P_tilt": layer.tilt,
        "P_align": layer.align_normal,
        "P_sink": layer.sink,
        "P_slope_min": layer.slope_min,
        "P_slope_max": max(layer.slope_max, layer.slope_min),
        "P_slope_soft": layer.slope_soft,
        "P_h_min": layer.height_min,
        "P_h_max": max(layer.height_max, layer.height_min),
        "P_h_soft": layer.height_soft,
        "P_noise_inv": 1.0 / max(layer.noise_size, 1e-3),
        "P_noise_t": 0.5 + (0.5 - layer.noise_keep) * 0.36,
        "P_cull_margin": layer.cull_margin,
        "P_cull_pad": layer.cull_pad,
        "P_cull_dist": layer.cull_distance,
        "P_vp": layer.viewport_pct,
        "P_paint_vp": s.paint_pct if (s.paint_reduce and _painting(ground)) else 1.0,
        "P_max": layer.max_count,
        "P_brush_nscale": s.brush_noise_scale,
        "P_brush_noise": s.brush_noise,
        "P_overlay": 1.0 if _overlay_on(ground, layer) else 0.0,
        "P_avoid_dist": layer.avoid_distance,
        "P_avoid_soft": layer.avoid_soft,
        "P_avoid_l_dist": layer.avoid_layers_distance,
        "P_bbox_dist": layer.bbox_distance,
    }
    for k, t in enumerate(_pick_thresholds(layer)):
        v["P_pick_%d" % k] = t
    cgp = _cgp(ground)
    if cgp is not None:
        v["P_cgp_nscale"] = cgp.edge_noise_scale
        for i, l in enumerate(cgp.layers):
            v["P_cgp_op_%d" % i] = l.opacity
    return v


# ====================================================================================
#  Scatter objects
# ====================================================================================
def _obj_name(ground, layer):
    return "CSC_%s_%s" % (ground.name, layer.name)


def ensure_scatter_collection(ground):
    s = ground.csc
    c = s.collection
    if c is None:
        c = bpy.data.collections.new("CSC_Scatter_%s" % ground.name)
        c[INTERNAL] = True
        s.collection = c
    linked = (any(c.name in p.children for p in bpy.data.collections) or
              any(c.name in sc.collection.children for sc in bpy.data.scenes))
    if not linked:
        parent = ground.users_collection[0] if ground.users_collection else bpy.context.scene.collection
        parent.children.link(c)
    return c


def _owned(idb, ground):
    """False when the data belongs to another ground (this one was duplicated with Shift+D)."""
    owner = idb.get("csc_ground")
    if owner is None:
        idb["csc_ground"] = ground
        return True
    return owner == ground


def ensure_scatter_object(ground, layer):
    obj = layer.obj
    if obj is not None and not _owned(obj, ground):
        layer.obj = None
        obj = None
    img = layer.brush_image
    if img is not None and not _owned(img, ground):
        img = img.copy()
        img.name = "CSC_%s_%s" % (ground.name, layer.uid)
        img["csc_ground"] = ground
        try:
            img.pack()
        except Exception:
            pass
        img.update_tag()
        layer.brush_image = img
    if obj is None:
        me = bpy.data.meshes.new("CSC_mesh_%s" % layer.uid)
        obj = bpy.data.objects.new(_obj_name(ground, layer), me)
        obj[INTERNAL] = True
        obj["csc_uid"] = layer.uid
        obj["csc_ground"] = ground
        obj.hide_select = True
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
        layer.obj = obj
    coll = ensure_scatter_collection(ground)
    if not obj.users_collection:
        coll.objects.link(obj)
    mod = obj.modifiers.get(MOD_NAME)
    if mod is None:
        mod = obj.modifiers.new(MOD_NAME, 'NODES')
    tree = mod.node_group
    if tree is None:
        tree = bpy.data.node_groups.new(TREE_PREFIX + layer.uid, "GeometryNodeTree")
        mod.node_group = tree
    return obj, mod, tree


def _tree(layer):
    o = layer.obj
    if o is None:
        return None
    m = o.modifiers.get(MOD_NAME)
    return m.node_group if m is not None else None


def rebuild_layer(ground, layer, scene=None):
    scene = scene or bpy.context.scene
    _BUSY[0] += 1
    try:
        obj, mod, tree = ensure_scatter_object(ground, layer)
        build_scatter_tree(tree, layer_spec(ground, layer))
        if layer.camera_cull:
            setup_camera_drivers(tree, scene)
        push_values(tree, layer_values(ground, layer))
        apply_visibility(ground)
    finally:
        _BUSY[0] -= 1
    _remember(ground)


def rebuild_all(ground):
    for l in ground.csc.layers:
        rebuild_layer(ground, l)


def push_layer(ground, layer):
    push_values(_tree(layer), layer_values(ground, layer))


def apply_visibility(ground):
    s = ground.csc
    for l in s.layers:
        vis = l.visible and (not s.solo_uid or s.solo_uid == l.uid)
        o = l.obj
        if o is None:
            continue
        md = o.modifiers.get(MOD_NAME)
        if md is not None:
            if md.show_viewport != vis:
                md.show_viewport = vis
            if md.show_render != vis:
                md.show_render = vis


def sync_assets(layer):
    """Fills the asset list from the chosen collection (keeps weights of known items)."""
    old = {}
    for a in layer.assets:
        key = a.obj.name if a.obj is not None else ("C:" + a.coll.name if a.coll is not None else "")
        old[key] = a.weight
    layer.assets.clear()
    layer.skipped = 0
    coll = layer.collection
    if coll is None:
        return
    ground = layer.id_data
    names = {o.name for o in coll.objects}
    for o in sorted(coll.objects, key=lambda o: o.name):
        if o == ground or o.get(INTERNAL) or o.get("csc_uid") is not None:
            continue
        if o.parent is not None and o.parent.name in names:
            layer.skipped += 1
            continue
        if o.type == 'EMPTY':
            if o.instance_type == 'COLLECTION' and o.instance_collection is not None:
                a = layer.assets.add()
                a.coll = o.instance_collection
                a.weight = old.get("C:" + a.coll.name, 1.0)
            continue
        if o.type not in ('MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'CURVES', 'POINTCLOUD', 'VOLUME'):
            continue
        a = layer.assets.add()
        a.obj = o
        a.weight = old.get(o.name, 1.0)
        if len(layer.assets) >= MAX_ASSETS:
            return
    for c in sorted(coll.children, key=lambda c: c.name):
        if c.get(INTERNAL) or _bad_collection(ground, c):
            continue
        a = layer.assets.add()
        a.coll = c
        a.weight = old.get("C:" + c.name, 1.0)
        if len(layer.assets) >= MAX_ASSETS:
            return


def remove_layer_data(layer):
    o = layer.obj
    tree = _tree(layer)
    if o is not None:
        me = o.data
        layer.obj = None
        bpy.data.objects.remove(o)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    if tree is not None and tree.users == 0:
        bpy.data.node_groups.remove(tree)


_AREA = {}


def ground_area(ground):
    me = ground.data
    key = (ground.name, len(me.polygons), tuple(round(x, 4) for x in ground.matrix_world.to_scale()))
    a = _AREA.get(key)
    if a is None:
        n = len(me.polygons)
        ar = np.empty(n, np.float32)
        me.polygons.foreach_get("area", ar)
        sc = ground.matrix_world.to_scale()
        a = float(ar.sum()) * abs(sc.x * sc.y * sc.z) ** (2.0 / 3.0)
        _AREA.clear()
        _AREA[key] = a
    return a


# ====================================================================================
#  Foliage brush data.  Brush image: R = grow, G = erase (B unused).  Same top-down UV as
#  Ground Paint, so the image lines up with the ground (CGP_UV when there is one).
# ====================================================================================
BRUSH_UV = "CSC_UV"


def _top_down_uv(me, uv):
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mn = co[:, :2].min(0)
    size = np.maximum(co[:, :2].max(0) - mn, 1e-6)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uv.data.foreach_set("uv", ((co[lv, :2] - mn) / size).astype(np.float32).ravel())


def ground_uv_bounds(ground):
    """(min xy, size xy) of the base mesh in the ground's local space: the top-down projection both
    CGP_UV and CSC_UV were made with."""
    me = ground.data
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mn = co[:, :2].min(0)
    return mn, np.maximum(co[:, :2].max(0) - mn, 1e-6)


def ensure_brush_uv(ground):
    """Uses Ground Paint's top-down UV when there is one, otherwise makes our own (same projection)."""
    me = ground.data
    if me.uv_layers.get(CGP_UV) is not None:
        return CGP_UV
    uv = me.uv_layers.get(BRUSH_UV)
    if uv is None:
        act = me.uv_layers.active_index
        ren = next((i for i, u in enumerate(me.uv_layers) if u.active_render), -1)
        uv = me.uv_layers.new(name=BRUSH_UV, do_init=False)
        if uv is None:
            raise RuntimeError("UVマップを追加できません（上限8個）")
        if act >= 0:
            me.uv_layers.active_index = act
        if ren >= 0:
            me.uv_layers[ren].active_render = True
    _top_down_uv(me, uv)
    return BRUSH_UV


def ensure_brush_image(ground, layer):
    """The layer's brush image (made empty when missing, converted when it is a v0.2 image)."""
    img = layer.brush_image
    if img is None:
        res = int(layer.brush_res)
        img = bpy.data.images.new("CSC_%s_%s" % (ground.name, layer.uid), res, res, alpha=False,
                                  float_buffer=False)
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
        img["csc_ground"] = ground
        img["csc_fmt"] = BRUSH_FMT
        layer.brush_image = img
        fill_image(img, 0.0)
    elif img.get("csc_fmt") != BRUSH_FMT:
        convert_v02_image(img, layer)
    return img


def read_pixels(img):
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)


def write_pixels(img, buf):
    img.pixels.foreach_set(np.ascontiguousarray(buf, dtype=np.float32).ravel())
    img.update_tag()     # pixels set from Python do not refresh geometry nodes on their own


def fill_image(img, v):
    """Everything grown (v = 1) or cleared (v = 0)."""
    w, h = img.size
    px = np.zeros((h, w, 4), dtype=np.float32)
    px[..., 0] = v
    px[..., 3] = 1.0
    write_pixels(img, px)
    img["csc_fmt"] = BRUSH_FMT
    try:
        img.pack()
    except Exception as e:
        print("CSC: pack failed", img.name, e)


def convert_v02_image(img, layer):
    """v0.2 density images (R = weight, G / B = softness / noise) -> R = grow, G = erase.
    A v0.2 'restrict' image on top of a Ground Paint range was full where it kept things, so there the
    old 'not painted' becomes 'erase'."""
    buf = read_pixels(img)
    r = buf[..., 0].copy()
    if layer.cgp_uid and layer.brush_mode == 'MUL':
        buf[..., 0] = 0.0
        buf[..., 1] = 1.0 - r
    else:
        buf[..., 1] = 0.0
    buf[..., 2] = 0.0
    buf[..., 3] = 1.0
    write_pixels(img, buf)
    img["csc_fmt"] = BRUSH_FMT
    try:
        img.pack()
    except Exception as e:
        print("CSC: pack failed", img.name, e)


def dab_box(w, h, cx, cy, rx, ry):
    """(x0, y0, x1, y1) of the pixels a dab at (cx, cy) with radii (rx, ry) can touch, or None."""
    x0 = max(int(math.floor(cx - rx)), 0)
    x1 = min(int(math.ceil(cx + rx)) + 1, w)
    y0 = max(int(math.floor(cy - ry)), 0)
    y1 = min(int(math.ceil(cy + ry)) + 1, h)
    if x0 >= x1 or y0 >= y1 or rx <= 0.0 or ry <= 0.0:
        return None
    return x0, y0, x1, y1


def paint_dab(buf, cx, cy, rx, ry, softness, strength, erase):
    """Stamps one soft round dab into buf (h, w, 4) around pixel (cx, cy); rx / ry = radii in pixels.
    Grow keeps the larger value (R) and wipes erase marks (G) where it lands; erase raises G.
    Returns (x0, y0, x1, y1) of the touched pixels, or None."""
    h, w = buf.shape[:2]
    box = dab_box(w, h, cx, cy, rx, ry)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    dx = (np.arange(x0, x1, dtype=np.float32) - cx) / rx
    dy = (np.arange(y0, y1, dtype=np.float32) - cy) / ry
    d = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
    inner = 1.0 - min(max(softness, 0.0), 1.0)
    t = np.clip((1.0 - d) / max(1.0 - inner, 1e-3), 0.0, 1.0)
    cover = t * t * (3.0 - 2.0 * t)
    part = buf[y0:y1, x0:x1]
    if erase:
        np.maximum(part[..., 1], cover * strength, out=part[..., 1])
    else:
        np.maximum(part[..., 0], cover * strength, out=part[..., 0])
        part[..., 1] *= 1.0 - cover
    return x0, y0, x1, y1


class Stroke:
    """Remembers the pixels a stroke touched (the union rectangle, as they were before) for Ctrl+Z."""

    def __init__(self, buf):
        self.buf = buf
        self.box = None
        self.patch = None

    def before(self, box):
        """Call with the rectangle the next dab will change, before changing it."""
        if self.box is None:
            self.box = box
            x0, y0, x1, y1 = box
            self.patch = self.buf[y0:y1, x0:x1].copy()
            return
        ox0, oy0, ox1, oy1 = self.box
        nx0, ny0 = min(ox0, box[0]), min(oy0, box[1])
        nx1, ny1 = max(ox1, box[2]), max(oy1, box[3])
        if (nx0, ny0, nx1, ny1) == self.box:
            return
        patch = self.buf[ny0:ny1, nx0:nx1].copy()      # outside the old box nothing was touched yet
        patch[oy0 - ny0:oy1 - ny0, ox0 - nx0:ox1 - nx0] = self.patch
        self.box, self.patch = (nx0, ny0, nx1, ny1), patch

    def result(self):
        if self.box is None:
            return None
        return self.box[0], self.box[1], self.patch


def restore_patch(buf, entry):
    x0, y0, patch = entry
    buf[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = patch


def ensure_preview_material(layer, img, uvname):
    """Translucent green (grow) / red (erase) tint of the brush image, shown on top of the ground."""
    name = "CSC_BrushPreview_%s" % layer.uid
    mat = bpy.data.materials.get(name)
    fresh = mat is None
    if fresh:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat[INTERNAL] = True
        mat.diffuse_color = (0.0, 0.0, 0.0, 0.0)      # invisible in Solid shading
        for attr, val in (("surface_render_method", 'BLENDED'), ("use_backface_culling", False)):
            try:
                setattr(mat, attr, val)
            except (AttributeError, TypeError):
                pass
    nt = mat.node_tree
    if fresh or nt.nodes.get("brush_tex") is None:
        nt.nodes.clear()
        N, L = nt.nodes, nt.links
        uvn = N.new("ShaderNodeUVMap")
        uvn.name = "brush_uv"
        uvn.location = (-900, 0)
        tex = N.new("ShaderNodeTexImage")
        tex.name = "brush_tex"
        tex.interpolation = 'Linear'
        tex.extension = 'EXTEND'
        tex.location = (-700, 0)
        L.new(uvn.outputs[0], tex.inputs["Vector"])
        sp = N.new("ShaderNodeSeparateColor")
        sp.location = (-420, 0)
        L.new(tex.outputs["Color"], sp.inputs[0])
        al = N.new("ShaderNodeMath")
        al.operation = 'MAXIMUM'
        al.location = (-220, -120)
        L.new(sp.outputs[0], al.inputs[0])
        L.new(sp.outputs[1], al.inputs[1])
        al2 = N.new("ShaderNodeMath")
        al2.operation = 'MULTIPLY'
        al2.use_clamp = True
        al2.inputs[1].default_value = 0.55
        al2.location = (-40, -120)
        L.new(al.outputs[0], al2.inputs[0])
        mx = N.new("ShaderNodeMix")
        mx.data_type = 'RGBA'
        mx.location = (-220, 120)
        _sock(mx.inputs, "A", 'RGBA').default_value = (0.15, 0.9, 0.25, 1.0)
        _sock(mx.inputs, "B", 'RGBA').default_value = (1.0, 0.2, 0.15, 1.0)
        L.new(sp.outputs[1], _sock(mx.inputs, "Factor", 'VALUE'))
        em = N.new("ShaderNodeEmission")
        em.location = (-40, 120)
        L.new(_sock(mx.outputs, "Result", 'RGBA'), em.inputs["Color"])
        tr = N.new("ShaderNodeBsdfTransparent")
        tr.location = (-40, 0)
        ms = N.new("ShaderNodeMixShader")
        ms.location = (160, 0)
        L.new(al2.outputs[0], ms.inputs[0])
        L.new(tr.outputs[0], ms.inputs[1])
        L.new(em.outputs[0], ms.inputs[2])
        out = N.new("ShaderNodeOutputMaterial")
        out.location = (360, 0)
        L.new(ms.outputs[0], out.inputs["Surface"])
    nt.nodes["brush_uv"].uv_map = uvname
    nt.nodes["brush_tex"].image = img
    return mat


# ====================================================================================
#  Foliage brush runtime state (also read by the cursor drawing and by layer_values)
# ====================================================================================
_FP = {}       # empty while the brush is not running


def foliage_running():
    return bool(_FP.get("ground"))


# ====================================================================================
#  Watching Ground Paint (layer order / opacity) and paint mode
# ====================================================================================
_GROUNDS = set()
_STATE = {}
_PENDING = set()


def _scan():
    _GROUNDS.clear()
    for o in bpy.data.objects:
        if o.type == 'MESH' and len(o.csc.layers) > 0:
            _GROUNDS.add(o.name)


def _cgp_sig(ground):
    cgp = _cgp(ground)
    if cgp is None:
        return (), ()
    struct = tuple((l.uid, l.kind, l.image.name if l.image else "") for l in cgp.layers)
    vals = (round(cgp.edge_noise_scale, 6),) + tuple(round(l.opacity, 6) for l in cgp.layers)
    return struct, vals


def _remember(ground):
    st, va = _cgp_sig(ground)
    _STATE[ground.name] = (st, va, _painting(ground))
    _GROUNDS.add(ground.name)


def _run_pending():
    names = list(_PENDING)
    _PENDING.clear()
    for n in names:
        g = bpy.data.objects.get(n)
        if g is not None and len(g.csc.layers):
            try:
                for l in g.csc.layers:
                    if l.cgp_uid:
                        rebuild_layer(g, l)
                _remember(g)
            except Exception as e:
                print("CSC: rebuild failed", n, e)
    return None


def _schedule_rebuild(name):
    _PENDING.add(name)
    if not bpy.app.timers.is_registered(_run_pending):
        bpy.app.timers.register(_run_pending, first_interval=0.05)


@persistent
def _csc_depsgraph_post(scene, depsgraph):
    if _BUSY[0] or not _GROUNDS:
        return
    for name in list(_GROUNDS):
        g = bpy.data.objects.get(name)
        if g is None:
            _scan()
            return
        if len(g.csc.layers) == 0:
            _GROUNDS.discard(name)
            continue
        st, va = _cgp_sig(g)
        painting = _painting(g)
        old = _STATE.get(name)
        if old is not None and old[0] != st:
            _schedule_rebuild(name)
        elif old is None or old[1] != va or old[2] != painting:
            _BUSY[0] += 1
            try:
                for l in g.csc.layers:
                    push_layer(g, l)
            finally:
                _BUSY[0] -= 1
        _STATE[name] = (st, va, painting)


def _fp_reset():
    """Drops the foliage brush state (the cursor drawing included)."""
    h = _FP.get("handle")
    if h is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
        except Exception:
            pass
    _FP.clear()


def _migrate_all():
    """Brings data saved by older versions up to date (v0.2 brush images / trees, left-over paint-mode state)."""
    for o in bpy.data.objects:
        if o.type != 'MESH' or len(o.csc.layers) == 0:
            continue
        prev = o.get("csc_prev_uv")
        if prev is not None:
            uv = o.data.uv_layers.get(prev)
            if uv is not None:
                o.data.uv_layers.active = uv
            del o["csc_prev_uv"]
        for l in o.csc.layers:
            try:
                img = l.brush_image
                if img is not None and img.get("csc_fmt") != BRUSH_FMT:
                    convert_v02_image(img, l)
                if l.use_brush and img is not None:
                    t = _tree(l)
                    if t is not None and t.get("csc_brush_fmt") != BRUSH_FMT:
                        rebuild_layer(o, l)
            except Exception as e:
                print("CSC: migration failed", o.name, l.name, e)


@persistent
def _csc_load_post(*args):
    _STATE.clear()
    _scan()
    _fp_reset()
    # bring shared node groups up to date in files saved by older versions
    if any(g.name in GROUP_NAMES and g.get("csc_ver") != GROUP_VER for g in bpy.data.node_groups):
        ensure_groups()
    _migrate_all()


# ====================================================================================
#  Properties
# ====================================================================================
def _upd_values(self, context):
    ground, layer = _layer_of(self)
    if layer is not None and not _BUSY[0]:
        push_layer(ground, layer)


def _upd_struct(self, context):
    ground, layer = _layer_of(self)
    if layer is not None and not _BUSY[0]:
        rebuild_layer(ground, layer, context.scene)


def _upd_collection(self, context):
    if _BUSY[0]:
        return
    ground, layer = _layer_of(self)
    if layer is None:
        return
    sync_assets(layer)
    if self.collection is not None and layer.name.startswith(DEFAULT_NAME):
        nm = self.collection.name
        for pre in ("CSC_Assets_", "Assets_", "CSC_"):
            if nm.startswith(pre) and len(nm) > len(pre):
                nm = nm[len(pre):]
        layer.name = nm
    rebuild_layer(ground, layer, context.scene)


def _upd_name(self, context):
    o = self.obj
    if o is not None:
        o.name = _obj_name(self.id_data, self)


def _upd_visible(self, context):
    apply_visibility(self.id_data)


def _upd_ground_values(self, context):
    ground = self.id_data
    for l in self.layers:
        push_layer(ground, l)


def _upd_height_toggle(self, context):
    if self.use_height and not self.height_init:
        bb = [self.id_data.matrix_world @ mathutils.Vector(c) for c in self.id_data.bound_box]
        zs = [v.z for v in bb]
        self.height_init = True
        _BUSY[0] += 1
        try:
            self.height_min = min(zs)
            self.height_max = max(zs)
        finally:
            _BUSY[0] -= 1
    _upd_struct(self, context)


def _upd_use_brush(self, context):
    if _BUSY[0]:
        return
    ground, layer = _layer_of(self)
    if layer is None:
        return
    if layer.use_brush:
        try:
            layer.brush_uv = ensure_brush_uv(ground)
        except RuntimeError as e:
            print("CSC:", e)
        ensure_brush_image(ground, layer)
    rebuild_layer(ground, layer, context.scene)


def _upd_brush_res(self, context):
    img = self.brush_image
    res = int(self.brush_res)
    if img is not None and tuple(img.size) != (res, res):
        img.scale(res, res)
        img["csc_fmt"] = BRUSH_FMT
        img.update_tag()
        try:
            img.pack()
        except Exception as e:
            print("CSC: pack failed", img.name, e)
        if foliage_running():
            _FP["reload"] = True


def _coll_poll(self, c):
    return not c.get(INTERNAL)


_ENUM_ITEMS = {}


def _cgp_items(self, context):
    items = [("NONE", "なし（地面全体）", "地面ペイントを使わず、地面全体に散らす", 'BLANK1', 0)]
    cgp = _cgp(self.id_data)
    if cgp is not None:
        for i, l in enumerate(cgp.layers):
            if i == 0:
                icon, desc = 'WORLD', "ベース：上に塗ったレイヤーがない所"
            elif l.kind == 'ADJUST':
                icon, desc = 'MODIFIER', "調整レイヤーを塗った所"
            else:
                icon, desc = 'MATERIAL', "このレイヤーを塗った所（濃いほど密に）"
            items.append((l.uid or ("L%d" % i), l.name, desc, icon, i + 1))
    _ENUM_ITEMS[self.as_pointer()] = items
    return items


def _cgp_get(self):
    i = _cgp_index(self.id_data, self.cgp_uid)
    return i + 1 if i >= 0 else 0


def _cgp_set(self, value):
    cgp = _cgp(self.id_data)
    uid = ""
    if cgp is not None and 1 <= value <= len(cgp.layers):
        uid = cgp.layers[value - 1].uid
    if uid != self.cgp_uid:
        self.cgp_uid = uid
        if not _BUSY[0]:
            rebuild_layer(self.id_data, self)


class CSC_Asset(bpy.types.PropertyGroup):
    obj: PointerProperty(type=bpy.types.Object)
    coll: PointerProperty(type=bpy.types.Collection)
    weight: FloatProperty(name="出現比率", default=1.0, min=0.0, soft_max=10.0, update=_upd_values,
                          description="このアセットが選ばれる割合（ほかのアセットとの比）。0で出なくなる")


class CSC_Layer(bpy.types.PropertyGroup):
    name: StringProperty(name="名前", default=DEFAULT_NAME, update=_upd_name)
    uid: StringProperty()
    obj: PointerProperty(type=bpy.types.Object)
    visible: BoolProperty(name="表示", default=True, update=_upd_visible,
                          description="このスキャッターレイヤーを表示する（レンダーにも反映）")
    # assets
    collection: PointerProperty(type=bpy.types.Collection, name="アセット", poll=_coll_poll,
                                update=_upd_collection,
                                description="散らす物を入れたコレクション。中のオブジェクト（と子コレクション）が1つずつアセットになる")
    assets: CollectionProperty(type=CSC_Asset)
    skipped: IntProperty(default=0)
    # placement
    density: FloatProperty(name="密度 (個/m²)", default=4.0, min=0.0, soft_max=50.0, max=2000.0,
                           update=_upd_values, description="1平方メートルあたりの最大の数（範囲の濃さで減る）")
    min_distance: FloatProperty(name="最小間隔 (m)", default=0.15, min=0.0, soft_max=5.0, update=_upd_values,
                                subtype='DISTANCE', description="アセット同士がこれ以上近づかない距離")
    seed: IntProperty(name="シード", default=0, min=0, update=_upd_values,
                      description="ばらまき方の種。変えると配置が変わる")
    scale_min: FloatProperty(name="最小", default=0.8, min=0.001, soft_max=5.0, update=_upd_values,
                             description="大きさのランダムの下限（1 = 元の大きさ）")
    scale_max: FloatProperty(name="最大", default=1.2, min=0.001, soft_max=5.0, update=_upd_values,
                             description="大きさのランダムの上限（1 = 元の大きさ）")
    rot_random: FloatProperty(name="回転のランダム", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                              update=_upd_values, description="向き（上下軸まわり）のばらつき。1でどの向きにもなる")
    tilt: FloatProperty(name="傾きのランダム", default=math.radians(5.0), min=0.0, max=math.radians(90.0),
                        subtype='ANGLE', update=_upd_values, description="ランダムに傾ける最大の角度")
    align_normal: FloatProperty(name="法線に沿わせる", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                                update=_upd_values,
                                description="0で真上に立つ、1で地面の傾きに沿う（草は少し、石は強めがおすすめ）")
    sink: FloatProperty(name="埋め込み (m)", default=0.02, soft_min=-0.5, soft_max=1.0, subtype='DISTANCE',
                        update=_upd_values, description="地面に少し沈める量（大きさに比例）。浮いて見えるときに増やす")
    # ranges
    cgp_layer: EnumProperty(name="地面ペイントのレイヤー", items=_cgp_items, get=_cgp_get, set=_cgp_set,
                            description="地面ペイントで塗ったレイヤーを、散らす範囲にする（塗りの濃さ＝密度。やわらかさ・ノイズも反映）")
    cgp_uid: StringProperty()
    occlude_upper: BoolProperty(name="上に塗ったレイヤーで消す", default=True, update=_upd_struct,
                                description="このレイヤーより上のマテリアルレイヤーで塗りつぶされた所には生やさない（見た目と一致）")
    paint_invert: BoolProperty(name="反転（塗った所を避ける）", default=False, update=_upd_struct,
                               description="範囲を反転して、塗った所以外に散らす")
    use_slope: BoolProperty(name="傾斜", default=False, update=_upd_struct,
                            description="地面の傾きで範囲をしぼる")
    slope_min: FloatProperty(name="最小", default=0.0, min=0.0, max=math.radians(180.0), subtype='ANGLE',
                             update=_upd_values, description="この角度より平らな所には生やさない（0°=水平）")
    slope_max: FloatProperty(name="最大", default=math.radians(35.0), min=0.0, max=math.radians(180.0),
                             subtype='ANGLE', update=_upd_values, description="この角度より急な所には生やさない")
    slope_soft: FloatProperty(name="ぼかし", default=math.radians(5.0), min=0.0, max=math.radians(90.0),
                              subtype='ANGLE', update=_upd_values, description="境目をなだらかにする幅")
    use_height: BoolProperty(name="高度", default=False, update=_upd_height_toggle,
                             description="高さ（ワールドのZ）で範囲をしぼる")
    height_init: BoolProperty(default=False)
    height_min: FloatProperty(name="最低", default=0.0, subtype='DISTANCE', update=_upd_values,
                              description="この高さより低い所には生やさない")
    height_max: FloatProperty(name="最高", default=10.0, subtype='DISTANCE', update=_upd_values,
                              description="この高さより高い所には生やさない")
    height_soft: FloatProperty(name="ぼかし", default=0.3, min=0.0, soft_max=10.0, subtype='DISTANCE',
                               update=_upd_values, description="境目をなだらかにする幅 (m)")
    use_noise: BoolProperty(name="ノイズ", default=False, update=_upd_struct,
                            description="ノイズでまだらにして、群生しているように見せる")
    noise_size: FloatProperty(name="大きさ (m)", default=3.0, min=0.01, soft_max=50.0, update=_upd_values,
                              description="まだら模様1つぶんのおおよその大きさ")
    noise_keep: FloatProperty(name="残す割合", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                              update=_upd_values, description="まだらのうち、生やす部分の割合")
    # foliage brush
    use_brush: BoolProperty(name="ブラシで生やした分を使う", default=False, update=_upd_use_brush,
                            description="ブラシで生やした所・消した所を、このレイヤーの範囲に反映する（ブラシを使うと自動でONになる）")
    brush_image: PointerProperty(type=bpy.types.Image)
    brush_uv: StringProperty(default="")
    brush_mode: EnumProperty(name="（v0.2の組み合わせ）", default='MUL', items=[       # only used to convert v0.2 images
        ('MUL', "しぼる", ""), ('ADD', "足す", "")])
    brush_res: EnumProperty(name="解像度", default='1024', update=_upd_brush_res, items=[
        ('512', "512", ""), ('1024', "1024", ""), ('2048', "2048", "")])
    # keep away from other things
    use_avoid: BoolProperty(name="ほかの物を避ける", default=False, update=_upd_struct,
                            description="指定したコレクションの物（建物・道・岩など）の近くには生やさない")
    avoid_collection: PointerProperty(type=bpy.types.Collection, name="避ける物", poll=_coll_poll,
                                      update=_upd_struct, description="この中の物の表面から離して配置する")
    avoid_distance: FloatProperty(name="距離 (m)", default=0.5, min=0.0, soft_max=10.0, subtype='DISTANCE',
                                  update=_upd_values, description="物の表面からこの距離までは生やさない")
    avoid_soft: FloatProperty(name="ぼかし (m)", default=0.3, min=0.0, soft_max=5.0, subtype='DISTANCE',
                              update=_upd_values, description="避ける境目をなだらかにする幅")
    avoid_layers: BoolProperty(name="上のレイヤーの物を避ける", default=False, update=_upd_struct,
                               description="一覧でこのレイヤーより上にあるスキャッターの物（岩・低木など）の近くには生やさない")
    avoid_layers_distance: FloatProperty(name="距離 (m)", default=0.3, min=0.0, soft_max=5.0, subtype='DISTANCE',
                                         update=_upd_values, description="上のレイヤーの物の中心からこの距離までは生やさない")
    # performance
    camera_cull: BoolProperty(name="カメラカリング", default=False, update=_upd_struct,
                              description="シーンのカメラに映らない所を省く（ビューポートとレンダーの両方）")
    cull_margin: FloatProperty(name="余白", default=0.1, min=0.0, soft_max=1.0, subtype='FACTOR',
                               update=_upd_values, description="画面の外側をどれだけ多めに残すか（0.1 = 10%）")
    cull_pad: FloatProperty(name="アセットの大きさ (m)", default=1.0, min=0.0, soft_max=10.0, subtype='DISTANCE',
                            update=_upd_values, description="画面の端で欠けないように、この距離ぶん外側も残す")
    cull_distance: FloatProperty(name="最大距離 (m)", default=0.0, min=0.0, soft_max=500.0, subtype='DISTANCE',
                                 update=_upd_values, description="カメラからこれより遠い所を省く（0で無制限）")
    viewport_pct: FloatProperty(name="ビューポートの表示割合", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                                update=_upd_values, description="ビューポートだけ数を減らして軽くする（レンダーは全部出る）")
    max_count: IntProperty(name="最大数", default=0, min=0, soft_max=200000, update=_upd_values,
                           description="この数を超えたら間引く（0で無制限）")
    bbox_far: BoolProperty(name="遠くは箱で表示", default=False, update=_upd_struct,
                           description="カメラから遠い物を、ビューポートだけ箱の線で表示して軽くする（レンダーは本物）")
    bbox_distance: FloatProperty(name="箱にする距離 (m)", default=40.0, min=0.0, soft_max=500.0, subtype='DISTANCE',
                                 update=_upd_values, description="シーンのカメラからこれより遠い物を箱で表示")


class CSC_Ground(bpy.types.PropertyGroup):
    layers: CollectionProperty(type=CSC_Layer)
    active_index: IntProperty(default=0, update=_upd_ground_values)
    solo_uid: StringProperty(default="")
    collection: PointerProperty(type=bpy.types.Collection)
    paint_reduce: BoolProperty(name="塗っている間は減らす", default=True, update=_upd_ground_values,
                               description="地面をテクスチャペイントしている間だけ、ビューポートの数を減らして軽くする")
    paint_pct: FloatProperty(name="塗るときの表示割合", default=0.35, min=0.0, max=1.0, subtype='FACTOR',
                             update=_upd_ground_values, description="ペイント中に表示する割合")
    # foliage brush
    brush_radius: FloatProperty(name="サイズ (m)", default=1.0, min=0.01, soft_max=30.0, max=500.0, subtype='DISTANCE',
                                description="ブラシの半径（[ ] キーでも変えられる）")
    brush_strength: FloatProperty(name="強さ", default=1.0, min=0.0, max=1.0, subtype='FACTOR',
                                  description="生やす濃さ（1で最大の密度、小さいほどまばら）。消すときは消す強さ（Shift+[ ] でも変えられる）")
    brush_softness: FloatProperty(name="やわらかさ", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                                  description="ブラシのふちのぼかし。0でくっきり、1で中心から外へなだらかに薄くなる（内側の円が目安）")
    brush_noise: FloatProperty(name="境界のノイズ", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                               update=_upd_ground_values, description="生やした所のふちをノイズで崩して自然にする強さ")
    brush_noise_scale: FloatProperty(name="ノイズの細かさ", default=1.0, min=0.01, soft_max=20.0,
                                     update=_upd_ground_values, description="1mあたりのノイズの細かさ")
    show_overlay: BoolProperty(name="塗った所を色で表示", default=False, update=_upd_ground_values,
                               description="ブラシで生やした所（緑）と消した所（赤）を、ビューポートだけ半透明で地面に重ねる（ブラシ中は自動で表示）")


# ====================================================================================
#  Operators
# ====================================================================================
def _ground(context):
    o = context.active_object
    if o is None or o.type != 'MESH' or o.get(INTERNAL):
        return None
    return o


def _active_layer(context):
    g = _ground(context)
    if g is None:
        return None, None
    s = g.csc
    if 0 <= s.active_index < len(s.layers):
        return g, s.layers[s.active_index]
    return g, None


class CSC_OT_layer_add(bpy.types.Operator):
    bl_idname = "csc.layer_add"
    bl_label = "スキャッターレイヤーを追加"
    bl_description = "選択中の地面に、新しいスキャッターレイヤーを追加する"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _ground(context) is not None

    def execute(self, context):
        g = _ground(context)
        s = g.csc
        _BUSY[0] += 1
        try:
            l = s.layers.add()
            l.uid = uuid.uuid4().hex[:8]
            used = {x.name for x in s.layers}
            k = 1
            while "%s %d" % (DEFAULT_NAME, k) in used:
                k += 1
            l.name = "%s %d" % (DEFAULT_NAME, k)
            l.seed = len(s.layers) * 7
        finally:
            _BUSY[0] -= 1
        rebuild_layer(g, l, context.scene)
        s.active_index = len(s.layers) - 1
        _scan()
        return {'FINISHED'}


class CSC_OT_layer_remove(bpy.types.Operator):
    bl_idname = "csc.layer_remove"
    bl_label = "スキャッターレイヤーを削除"
    bl_description = "選択中のスキャッターレイヤーと、その散らした物を削除する（アセット自体は消えない）"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_layer(context)[1] is not None

    def execute(self, context):
        g, l = _active_layer(context)
        s = g.csc
        i = s.active_index
        if s.solo_uid == l.uid:
            s.solo_uid = ""
        img = l.brush_image
        l.brush_image = None
        remove_layer_data(l)
        s.layers.remove(i)
        if img is not None and img.users == 0:
            bpy.data.images.remove(img)
        s.active_index = max(0, min(i, len(s.layers) - 1))
        for x in s.layers:
            if x.avoid_layers:
                rebuild_layer(g, x, context.scene)
        apply_visibility(g)
        _scan()
        return {'FINISHED'}


class CSC_OT_layer_move(bpy.types.Operator):
    bl_idname = "csc.layer_move"
    bl_label = "スキャッターレイヤーを移動"
    bl_options = {'REGISTER', 'UNDO'}
    direction: EnumProperty(items=[('UP', "上へ", ""), ('DOWN', "下へ", "")])

    @classmethod
    def poll(cls, context):
        return _active_layer(context)[1] is not None

    def execute(self, context):
        g, l = _active_layer(context)
        s = g.csc
        i = s.active_index
        j = i - 1 if self.direction == 'UP' else i + 1
        if not 0 <= j < len(s.layers):
            return {'CANCELLED'}
        s.layers.move(i, j)
        s.active_index = j
        # "avoid the layers above" depends on the order (rebuilt top to bottom, so no cycles)
        for x in s.layers:
            if x.avoid_layers:
                rebuild_layer(g, x, context.scene)
        return {'FINISHED'}


class CSC_OT_solo(bpy.types.Operator):
    bl_idname = "csc.solo"
    bl_label = "ソロ"
    bl_description = "このスキャッターレイヤーだけを表示する（もう一度押すと解除）"
    bl_options = {'REGISTER', 'UNDO'}
    uid: StringProperty()

    def execute(self, context):
        g = _ground(context)
        if g is None:
            return {'CANCELLED'}
        s = g.csc
        s.solo_uid = "" if s.solo_uid == self.uid else self.uid
        apply_visibility(g)
        return {'FINISHED'}


class CSC_OT_assets_reload(bpy.types.Operator):
    bl_idname = "csc.assets_reload"
    bl_label = "アセットを読み直す"
    bl_description = "コレクションの中身を変えたときに押すと、アセットの一覧を更新する"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _active_layer(context)[1] is not None

    def execute(self, context):
        g, l = _active_layer(context)
        sync_assets(l)
        rebuild_layer(g, l, context.scene)
        return {'FINISHED'}


class CSC_OT_rebuild(bpy.types.Operator):
    bl_idname = "csc.rebuild"
    bl_label = "作り直す"
    bl_description = "この地面のスキャッターをすべて作り直す（表示がおかしいときに）"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        g = _ground(context)
        return g is not None and len(g.csc.layers) > 0

    def execute(self, context):
        g = _ground(context)
        for l in g.csc.layers:
            sync_assets(l)
        rebuild_all(g)
        return {'FINISHED'}


def enable_brush(ground, layer, scene=None, uvname=None):
    """Makes sure the layer has a brush image and uses it (rebuilds the layer when that changes anything)."""
    uvname = uvname or ensure_brush_uv(ground)
    img = ensure_brush_image(ground, layer)
    if not layer.use_brush or layer.brush_uv != uvname:
        _BUSY[0] += 1
        try:
            layer.brush_uv = uvname
            layer.use_brush = True
        finally:
            _BUSY[0] -= 1
        rebuild_layer(ground, layer, scene)
    return img


def _layer_by_uid(ground, uid):
    for l in ground.csc.layers:
        if l.uid == uid:
            return l
    return None


def _tag_redraw_3d():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for a in win.screen.areas:
            if a.type == 'VIEW_3D':
                a.tag_redraw()


def _push_ground(ground):
    _BUSY[0] += 1
    try:
        for l in ground.csc.layers:
            push_layer(ground, l)
    finally:
        _BUSY[0] -= 1


FP_FLUSH = 0.06        # seconds between writes of the brush image while painting
FP_STATUS = ("ブラシで生やす   左ドラッグ: 生やす   Shift/Ctrl+ドラッグ: 消す   [ ]: サイズ   Shift+[ ]: 強さ   "
             "Ctrl+ホイール: サイズ   Ctrl+Z: 1回戻す   右クリック / Esc / Enter: 終了")
FP_UI_REGIONS = ('UI', 'HEADER', 'TOOL_HEADER', 'TOOLS', 'FOOTER', 'NAVIGATION_BAR', 'HUD', 'EXECUTE')


def _cursor_shader():
    """Thick lines need the polyline shader on some drivers; fall back to the plain one."""
    try:
        sh = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        sh.bind()
        sh.uniform_float("viewportSize", gpu.state.viewport_get()[2:4])
        sh.uniform_float("lineWidth", 2.5)
        return sh, True
    except Exception:
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        sh.bind()
        return sh, False


def _draw_cursor():
    """Brush circle on the ground (green = grow, red = erase; the inner circle is where it is fully strong)."""
    try:
        if not _FP.get("visible") or not _FP.get("outer"):
            return
        shader, poly = _cursor_shader()
        col = (1.0, 0.25, 0.2) if _FP.get("erase") else (0.25, 0.95, 0.3)
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        for key, a in (("outer", 0.95), ("inner", 0.55)):
            pts = _FP.get(key)
            if pts:
                shader.uniform_float("color", (col[0], col[1], col[2], a))
                batch_for_shader(shader, 'LINE_STRIP', {"pos": pts}).draw(shader)
        gpu.state.blend_set('NONE')
    except Exception as e:
        print("CSC: cursor draw failed", e)


class CSC_OT_foliage_paint(bpy.types.Operator):
    bl_idname = "csc.foliage_paint"
    bl_label = "ブラシで生やす"
    bl_description = ("オブジェクトモードのまま地面をなでて、その場に生やす。Shift / Ctrl+ドラッグで消す。"
                      "[ ] でサイズ、Shift+[ ] で強さ、Ctrl+Z で1ストローク戻す、右クリック / Esc / Enter で終了")

    @classmethod
    def poll(cls, context):
        return (_active_layer(context)[1] is not None and context.mode == 'OBJECT' and
                context.area is not None and context.area.type == 'VIEW_3D' and not foliage_running())

    # ---------------------------------------------------------------- setup / teardown
    def invoke(self, context, event):
        g, l = _active_layer(context)
        if not any(r.type == 'WINDOW' for r in context.area.regions):
            return {'CANCELLED'}
        try:
            self.uv = ensure_brush_uv(g)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.gname = g.name
        mn, size = ground_uv_bounds(g)
        self.mn, self.size = mn, size
        sc = g.matrix_world.to_scale()
        self.rscale = max((abs(sc.x) + abs(sc.y)) * 0.5, 1e-6)
        self.bufs = {}
        self.undo = []
        self.stroke = None
        self.last = None
        self.cur = None
        self.cur_uid = None
        self.last_flush = 0.0
        self.erase = False
        _fp_reset()
        _FP.update(ground=g.name, visible=False, erase=False, outer=None, inner=None, stop=False, reload=False)
        try:
            if self._prep(context, g, l) is None:
                raise RuntimeError("ブラシ用の画像を準備できません")
        except Exception as e:
            _fp_reset()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        _FP["handle"] = bpy.types.SpaceView3D.draw_handler_add(_draw_cursor, (), 'WINDOW', 'POST_VIEW')
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.02, window=context.window)
        try:
            context.workspace.status_text_set(FP_STATUS)
        except Exception:
            pass
        _push_ground(g)
        _tag_redraw_3d()
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        try:
            self._flush(force=True)
            for st in self.bufs.values():
                img = bpy.data.images.get(st["img"])
                if img is not None:
                    try:
                        img.pack()
                    except Exception as e:
                        print("CSC: pack failed", img.name, e)
        except Exception:
            traceback.print_exc()
        try:
            context.window_manager.event_timer_remove(self.timer)
        except Exception:
            pass
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass
        _fp_reset()
        g = bpy.data.objects.get(self.gname)
        if g is not None:
            _push_ground(g)
        _tag_redraw_3d()

    def cancel(self, context):
        self._finish(context)

    # ---------------------------------------------------------------- brush image buffers
    def _prep(self, context, g, l):
        """Gets the layer ready (brush image, UV, node tree) and returns its pixel buffer state."""
        img = enable_brush(g, l, context.scene, self.uv)
        st = self.bufs.get(l.uid)
        if st is None or st["img"] != img.name or st["size"] != tuple(img.size):
            st = {"img": img.name, "buf": read_pixels(img), "size": tuple(img.size), "dirty": False}
            self.bufs[l.uid] = st
        self.cur, self.cur_uid = st, l.uid
        return st

    def _flush(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_flush < FP_FLUSH:
            return
        for st in self.bufs.values():
            if st["dirty"]:
                img = bpy.data.images.get(st["img"])
                if img is not None and tuple(img.size) == st["size"]:
                    write_pixels(img, st["buf"])
                st["dirty"] = False
                self.last_flush = now

    def _reload(self):
        _FP["reload"] = False
        for uid, st in list(self.bufs.items()):
            img = bpy.data.images.get(st["img"])
            if img is None:
                del self.bufs[uid]
                continue
            st["buf"], st["size"], st["dirty"] = read_pixels(img), tuple(img.size), False
        self.undo.clear()
        self.stroke = None
        self.cur = self.bufs.get(self.cur_uid)

    # ---------------------------------------------------------------- mouse -> ground
    def _regions(self, area, mx, my):
        """(3D window region, True when the mouse is over the window region and not over a side panel,
        True when it is over the navigation gizmo corner)."""
        win = None
        over_other = False
        right, top = None, None
        for r in area.regions:
            if r.type == 'WINDOW':
                win = r if win is None else win
                continue
            if r.width <= 1 or r.height <= 1:
                continue
            if r.type in FP_UI_REGIONS and r.x <= mx < r.x + r.width and r.y <= my < r.y + r.height:
                over_other = True
        if win is None:
            return None, False, False
        right = win.x + win.width
        top = win.y + win.height
        for r in area.regions:
            if r.width <= 1 or r.height <= 1:
                continue
            if r.type == 'UI' and win.x < r.x < right:
                right = r.x
            if r.type in ('HEADER', 'TOOL_HEADER') and win.y < r.y < top:
                top = r.y
        inside = win.x <= mx < win.x + win.width and win.y <= my < win.y + win.height and not over_other
        gizmo = mx > right - 130 and my > top - 210
        return win, inside, gizmo

    def _hit(self, context, g, win, mx, my):
        """Where the mouse points on the ground, in the ground's local space: (x, y, z) or None."""
        rv3d = context.area.spaces.active.region_3d
        co = (mx - win.x, my - win.y)
        origin = view3d_utils.region_2d_to_origin_3d(win, rv3d, co)
        direction = view3d_utils.region_2d_to_vector_3d(win, rv3d, co)
        inv = g.matrix_world.inverted_safe()
        o = inv @ origin
        d = (inv.to_3x3() @ direction).normalized()
        r = g.ray_cast(o, d, depsgraph=context.evaluated_depsgraph_get())
        if not r[0]:
            return None
        return r[1].x, r[1].y, r[1].z

    def _ring(self, context, g, cx, cy, cz, r, n):
        dg = context.evaluated_depsgraph_get()
        mw = g.matrix_world
        top = cz + r * 2.0 + 1.0
        reach = r * 4.0 + 3.0
        down = mathutils.Vector((0.0, 0.0, -1.0))
        pts = []
        for k in range(n + 1):
            a = math.tau * k / n
            x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            res = g.ray_cast(mathutils.Vector((x, y, top)), down, distance=reach, depsgraph=dg)
            z = res[1].z if res[0] else cz
            pts.append(tuple(mw @ mathutils.Vector((x, y, z + 0.02))))
        return pts

    def _cursor(self, context, g, hit):
        s = g.csc
        if hit is None:
            _FP["visible"] = False
        else:
            r = s.brush_radius / self.rscale
            _FP["outer"] = self._ring(context, g, hit[0], hit[1], hit[2], r, 48)
            inner = r * (1.0 - s.brush_softness)
            _FP["inner"] = self._ring(context, g, hit[0], hit[1], hit[2], inner, 32) if s.brush_softness > 0.03 else None
            _FP["visible"] = True
        context.area.tag_redraw()

    # ---------------------------------------------------------------- painting
    def _stamp(self, g, hit):
        """Dabs at the hit, filling in the way from the previous one."""
        st = self.cur
        if st is None or self.stroke is None:
            return
        s = g.csc
        w, h = st["size"]
        r = s.brush_radius / self.rscale
        rx, ry = r / self.size[0] * w, r / self.size[1] * h
        pos = (hit[0], hit[1])
        if self.last is None:
            pts = [pos]
        else:
            dx, dy = pos[0] - self.last[0], pos[1] - self.last[1]
            dist = math.hypot(dx, dy)
            n = max(1, int(math.ceil(dist / max(r * 0.2, 1e-4))))
            pts = [(self.last[0] + dx * i / n, self.last[1] + dy * i / n) for i in range(1, n + 1)]
        for x, y in pts:
            cx = (x - self.mn[0]) / self.size[0] * w - 0.5
            cy = (y - self.mn[1]) / self.size[1] * h - 0.5
            box = dab_box(w, h, cx, cy, rx, ry)
            if box is None:
                continue
            self.stroke.before(box)
            paint_dab(st["buf"], cx, cy, rx, ry, s.brush_softness, s.brush_strength, self.erase)
            st["dirty"] = True
        self.last = pos

    def _begin(self, context, g, hit, erase):
        l = _active_layer(context)[1]
        if l is None:
            return
        if l.uid != self.cur_uid or not l.use_brush:
            if self._prep(context, g, l) is None:
                return
            _push_ground(g)
        self.erase = erase
        self.last = None
        self.stroke = Stroke(self.cur["buf"])
        self.stroke_uid = l.uid
        if hit is not None:
            self._stamp(g, hit)

    def _end(self):
        if self.stroke is None:
            return
        entry = self.stroke.result()
        if entry is not None:
            self.undo.append((self.stroke_uid, entry))
            del self.undo[:-40]
        self.stroke = None
        self.last = None
        self._flush(force=True)

    def _undo(self):
        if not self.undo:
            return
        uid, entry = self.undo.pop()
        st = self.bufs.get(uid)
        if st is not None:
            restore_patch(st["buf"], entry)
            st["dirty"] = True
            self._flush(force=True)

    # ---------------------------------------------------------------- events
    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception:
            traceback.print_exc()
            self._finish(context)
            return {'CANCELLED'}

    def _modal(self, context, event):
        g = bpy.data.objects.get(self.gname)
        if (g is None or not foliage_running() or _FP.get("stop") or context.area is None or
                context.area.type != 'VIEW_3D' or context.active_object != g or context.mode != 'OBJECT'):
            self._finish(context)
            return {'FINISHED'}
        if _FP.get("reload"):
            self._reload()
        et, val = event.type, event.value
        if et == 'TIMER':
            self._flush()
            return {'PASS_THROUGH'}
        if et == 'ESC' and val == 'PRESS':
            self._finish(context)
            return {'FINISHED'}
        mx, my = event.mouse_x, event.mouse_y
        win, inside, gizmo = self._regions(context.area, mx, my)
        _FP["erase"] = bool(event.shift or event.ctrl)
        if et == 'LEFTMOUSE' and val == 'RELEASE' and self.stroke is not None:
            self._end()
            return {'RUNNING_MODAL'}
        if win is None or not inside:
            if _FP.get("visible"):
                _FP["visible"] = False
                context.area.tag_redraw()
            return {'PASS_THROUGH'}
        if et in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'):
            hit = self._hit(context, g, win, mx, my)
            self._cursor(context, g, hit)
            if self.stroke is not None:
                if hit is not None:
                    self._stamp(g, hit)
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        if et == 'LEFTMOUSE' and not event.alt:
            if val in ('PRESS', 'DOUBLE_CLICK') and not gizmo:
                self._begin(context, g, self._hit(context, g, win, mx, my), bool(event.shift or event.ctrl))
                return {'RUNNING_MODAL'}
            if val in ('CLICK', 'CLICK_DRAG') and self.stroke is not None:
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}
        s = g.csc
        if val == 'PRESS':
            if et == 'RIGHTMOUSE' or et in ('RET', 'NUMPAD_ENTER'):
                self._finish(context)
                return {'FINISHED'}
            if et == 'Z' and event.ctrl and not event.shift and not event.alt:
                if self.stroke is None:
                    self._undo()
                return {'RUNNING_MODAL'}
            if et in ('LEFT_BRACKET', 'RIGHT_BRACKET'):
                up = et == 'RIGHT_BRACKET'
                if event.shift:
                    s.brush_strength = min(max(s.brush_strength + (0.05 if up else -0.05), 0.0), 1.0)
                else:
                    s.brush_radius = min(max(s.brush_radius * (1.12 if up else 1.0 / 1.12), 0.01), 500.0)
                    self._cursor(context, g, self._hit(context, g, win, mx, my))
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}
        if et in ('WHEELUPMOUSE', 'WHEELDOWNMOUSE') and event.ctrl and not event.alt:
            up = et == 'WHEELUPMOUSE'
            s.brush_radius = min(max(s.brush_radius * (1.12 if up else 1.0 / 1.12), 0.01), 500.0)
            self._cursor(context, g, self._hit(context, g, win, mx, my))
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


class CSC_OT_foliage_exit(bpy.types.Operator):
    bl_idname = "csc.foliage_exit"
    bl_label = "ブラシを終了"
    bl_description = "ブラシで生やすのを終わる（右クリック / Esc / Enter でも終われる）"

    @classmethod
    def poll(cls, context):
        return foliage_running()

    def execute(self, context):
        _FP["stop"] = True
        return {'FINISHED'}


class CSC_OT_brush_fill(bpy.types.Operator):
    bl_idname = "csc.brush_fill"
    bl_label = "全面に生やす / クリア"
    bl_description = "このレイヤーのブラシの内容を、地面全体に生やす / すべて消して元に戻す"
    bl_options = {'REGISTER', 'UNDO'}
    value: FloatProperty(default=1.0, min=0.0, max=1.0)

    @classmethod
    def poll(cls, context):
        return _active_layer(context)[1] is not None

    def execute(self, context):
        g, l = _active_layer(context)
        try:
            img = enable_brush(g, l, context.scene)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        fill_image(img, self.value)
        if foliage_running():
            _FP["reload"] = True
        return {'FINISHED'}


def _instances_of(depsgraph, obj):
    out = []
    for inst in depsgraph.object_instances:
        if inst.is_instance and inst.parent is not None and inst.parent.original == obj:
            out.append((inst.object.original, inst.matrix_world.copy()))
    return out


class CSC_OT_realize(bpy.types.Operator):
    bl_idname = "csc.realize"
    bl_label = "実体化"
    bl_description = ("散らした物を、1つずつ動かしたり消したりできる普通のオブジェクトにする。"
                      "新しいコレクションに入り、元のスキャッターレイヤーは非表示になる")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        g, l = _active_layer(context)
        return l is not None and l.obj is not None and context.mode == 'OBJECT'

    def _collect(self, context, g, l):
        tree = _tree(l)
        vals = layer_values(g, l)
        vals["P_vp"] = 1.0
        vals["P_paint_vp"] = 1.0
        md = l.obj.modifiers.get(MOD_NAME)
        was = md.show_viewport if md else True
        if md is not None:
            md.show_viewport = True
        push_values(tree, vals)
        dg = context.evaluated_depsgraph_get()
        dg.update()
        items = _instances_of(dg, l.obj)
        if md is not None:
            md.show_viewport = was
        push_layer(g, l)
        return items

    def invoke(self, context, event):
        g, l = _active_layer(context)
        items = self._collect(context, g, l)
        if not items:
            self.report({'WARNING'}, "実体化する物がありません")
            return {'CANCELLED'}
        return context.window_manager.invoke_confirm(
            self, event, title="実体化", confirm_text="実体化する",
            message="%d 個のオブジェクトを作ります（元のレイヤーは非表示になります）" % len(items))

    def execute(self, context):
        g, l = _active_layer(context)
        items = self._collect(context, g, l)
        if not items:
            self.report({'WARNING'}, "実体化する物がありません")
            return {'CANCELLED'}
        coll = bpy.data.collections.new("CSC_実体化_%s" % l.name)
        parent = g.users_collection[0] if g.users_collection else context.scene.collection
        parent.children.link(coll)
        for src, mw in items:
            o = src.copy()
            o.parent = None
            o.matrix_world = mw
            o.hide_viewport = False
            o.hide_render = False
            o.hide_select = False
            for k in (INTERNAL, "csc_uid"):
                if k in o:
                    del o[k]
            coll.objects.link(o)
        l.visible = False
        self.report({'INFO'}, "%d 個を実体化しました（コレクション: %s）" % (len(items), coll.name))
        return {'FINISHED'}


# ====================================================================================
#  UI
# ====================================================================================
class CSC_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        s = data
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False, icon='OUTLINER_OB_POINTCLOUD')
        sub = row.row(align=True)
        sub.alignment = 'RIGHT'
        if item.cgp_uid:
            cgp = _cgp(item.id_data)
            i = _cgp_index(item.id_data, item.cgp_uid)
            sub.label(text=cgp.layers[i].name if (cgp is not None and i >= 0) else "?", icon='BRUSH_DATA')
        sub.prop(item, "visible", text="", emboss=False, icon='HIDE_OFF' if item.visible else 'HIDE_ON')
        op = sub.operator("csc.solo", text="", emboss=False,
                          icon='SOLO_ON' if s.solo_uid == item.uid else 'SOLO_OFF')
        op.uid = item.uid


class CSC_PT_main(bpy.types.Panel):
    bl_idname = "CSC_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Concept"
    bl_label = "スキャッター"

    def draw(self, context):
        layout = self.layout
        g = _ground(context)
        if g is None:
            layout.label(text="地面のメッシュを選択してください", icon='INFO')
            return
        s = g.csc
        layout.label(text=g.name, icon='MESH_GRID')
        row = layout.row()
        row.template_list("CSC_UL_layers", "", s, "layers", s, "active_index", rows=4)
        col = row.column(align=True)
        col.operator("csc.layer_add", icon='ADD', text="")
        col.operator("csc.layer_remove", icon='REMOVE', text="")
        col.separator()
        col.operator("csc.layer_move", icon='TRIA_UP', text="").direction = 'UP'
        col.operator("csc.layer_move", icon='TRIA_DOWN', text="").direction = 'DOWN'
        if len(s.layers) == 0:
            layout.label(text="＋ でスキャッターレイヤーを追加", icon='INFO')
        elif s.solo_uid:
            layout.label(text="ソロ表示中（S のアイコンで解除）", icon='SOLO_ON')


class _CSC_Sub:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Concept"
    bl_parent_id = "CSC_PT_main"

    @classmethod
    def poll(cls, context):
        return _active_layer(context)[1] is not None


class CSC_PT_assets(_CSC_Sub, bpy.types.Panel):
    bl_idname = "CSC_PT_assets"
    bl_label = "アセット（散らす物）"

    def draw(self, context):
        layout = self.layout
        g, l = _active_layer(context)
        layout.prop(l, "collection", text="", icon='OUTLINER_COLLECTION')
        if l.collection is None:
            layout.label(text="散らす物のコレクションを選んでください", icon='INFO')
            return
        if len(l.assets) == 0:
            layout.label(text="コレクションにメッシュがありません", icon='ERROR')
        col = layout.column(align=True)
        for a in l.assets:
            r = col.row(align=True)
            if a.obj is not None:
                r.label(text=a.obj.name, icon='OUTLINER_OB_MESH')
            elif a.coll is not None:
                r.label(text=a.coll.name, icon='OUTLINER_COLLECTION')
            else:
                r.label(text="（削除済み）", icon='ERROR')
            r.prop(a, "weight", text="比率")
        if l.skipped:
            layout.label(text="子オブジェクト %d 個は除外（組み物は子コレクションに）" % l.skipped, icon='INFO')
        layout.operator("csc.assets_reload", icon='FILE_REFRESH')
        tip = layout.column(align=True)
        tip.scale_y = 0.75
        tip.label(text="元のアセットのコレクションは、非表示・除外にしても", icon='INFO')
        tip.label(text="散らした物は消えません", icon='BLANK1')


class CSC_PT_brush(_CSC_Sub, bpy.types.Panel):
    bl_idname = "CSC_PT_brush"
    bl_label = "ブラシで生やす"

    def draw(self, context):
        layout = self.layout
        g, l = _active_layer(context)
        s = g.csc
        r = layout.row()
        r.scale_y = 1.5
        if foliage_running():
            r.operator("csc.foliage_exit", text="ブラシを終了", icon='CHECKMARK')
        else:
            r.operator("csc.foliage_paint", text="ブラシで生やす", icon='BRUSH_DATA')
            if context.mode != 'OBJECT':
                layout.label(text="オブジェクトモードで使えます", icon='INFO')
        hint = layout.column(align=True)
        hint.scale_y = 0.8
        hint.label(text="左ドラッグ: 生やす / Shift・Ctrl+ドラッグ: 消す")
        hint.label(text="[ ] サイズ / Shift+[ ] 強さ / Ctrl+Z 戻す")
        hint.label(text="右クリック・Esc・Enter で終了")
        col = layout.column(align=True)
        col.prop(s, "brush_radius")
        col.prop(s, "brush_strength")
        col.prop(s, "brush_softness")
        col = layout.column(align=True)
        col.prop(s, "brush_noise")
        col.prop(s, "brush_noise_scale")
        layout.prop(s, "show_overlay", icon='RESTRICT_VIEW_OFF')
        box = layout.box()
        box.prop(l, "use_brush")
        if l.use_brush:
            box.prop(l, "brush_res")
            row = box.row(align=True)
            row.operator("csc.brush_fill", text="全面に生やす", icon='SNAP_FACE').value = 1.0
            row.operator("csc.brush_fill", text="クリア", icon='X').value = 0.0
            tip = box.column(align=True)
            tip.scale_y = 0.75
            tip.label(text="範囲 = (地面ペイント or ブラシで生やした所)")
            tip.label(text="から、ブラシで消した所を除いたもの")


class CSC_PT_place(_CSC_Sub, bpy.types.Panel):
    bl_idname = "CSC_PT_place"
    bl_label = "配置"

    def draw(self, context):
        layout = self.layout
        g, l = _active_layer(context)
        col = layout.column(align=True)
        col.prop(l, "density")
        col.prop(l, "min_distance")
        col.prop(l, "seed")
        n = ground_area(g) * l.density
        r = layout.row()
        r.scale_y = 0.8
        if n > 1_000_000:
            r.alert = True
            r.label(text="候補 約%d万個：重すぎます。密度を下げてください" % int(n / 10000), icon='ERROR')
        else:
            r.label(text="候補 約%s個（範囲で減ります）" % ("{:,}".format(int(n))), icon='INFO')
        layout.label(text="大きさ", icon='CON_SIZELIKE')
        r = layout.row(align=True)
        r.prop(l, "scale_min")
        r.prop(l, "scale_max")
        col = layout.column(align=True)
        col.prop(l, "rot_random")
        col.prop(l, "tilt")
        col.prop(l, "align_normal")
        col.prop(l, "sink")


class CSC_PT_mask(_CSC_Sub, bpy.types.Panel):
    bl_idname = "CSC_PT_mask"
    bl_label = "範囲（どこに生やすか）"

    def draw(self, context):
        layout = self.layout
        g, l = _active_layer(context)
        cgp = _cgp(g)
        box = layout.box()
        box.label(text="地面ペイントのレイヤー", icon='BRUSH_DATA')
        if cgp is not None and len(cgp.layers) > 0:
            box.prop(l, "cgp_layer", text="")
            if l.cgp_uid and _cgp_index(g, l.cgp_uid) < 0:
                box.label(text="選んでいたレイヤーが見つかりません", icon='ERROR')
            elif l.cgp_uid:
                box.prop(l, "occlude_upper")
                box.prop(l, "paint_invert")
        else:
            box.label(text="地面ペイントを使うと、塗った所に散らせます", icon='INFO')
        for flag, props in (("use_slope", ("slope_min", "slope_max", "slope_soft")),
                            ("use_height", ("height_min", "height_max", "height_soft")),
                            ("use_noise", ("noise_size", "noise_keep"))):
            box = layout.box()
            box.prop(l, flag)
            if getattr(l, flag):
                col = box.column(align=True)
                for p in props:
                    col.prop(l, p)
        # keep away from other things
        box = layout.box()
        box.prop(l, "use_avoid")
        if l.use_avoid:
            box.prop(l, "avoid_collection", text="", icon='OUTLINER_COLLECTION')
            if l.avoid_collection is None:
                box.label(text="避ける物のコレクションを選んでください", icon='INFO')
            elif _bad_collection(g, l.avoid_collection):
                box.label(text="地面やスキャッター自身を含むので使えません", icon='ERROR')
            col = box.column(align=True)
            col.prop(l, "avoid_distance")
            col.prop(l, "avoid_soft")
        box.prop(l, "avoid_layers")
        if l.avoid_layers:
            above = []
            for x in g.csc.layers:
                if x.as_pointer() == l.as_pointer():
                    break
                above.append(x.name)
            box.prop(l, "avoid_layers_distance")
            if above:
                box.label(text="避ける: " + "、".join(above), icon='INFO')
            else:
                box.label(text="一覧でこのレイヤーより上にレイヤーがありません", icon='INFO')


class CSC_PT_perf(_CSC_Sub, bpy.types.Panel):
    bl_idname = "CSC_PT_perf"
    bl_label = "軽量化"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        g, l = _active_layer(context)
        box = layout.box()
        box.prop(l, "camera_cull", icon='CAMERA_DATA')
        if l.camera_cull:
            col = box.column(align=True)
            col.prop(l, "cull_margin")
            col.prop(l, "cull_pad")
            col.prop(l, "cull_distance")
            if context.scene.camera is None:
                box.label(text="シーンにカメラがありません", icon='ERROR')
        col = layout.column(align=True)
        col.prop(l, "viewport_pct")
        col.prop(l, "max_count")
        box = layout.box()
        box.prop(l, "bbox_far", icon='SHADING_BBOX')
        if l.bbox_far:
            box.prop(l, "bbox_distance")
            if context.scene.camera is None:
                box.label(text="シーンにカメラがありません", icon='ERROR')
        box = layout.box()
        box.label(text="この地面のすべてのレイヤー", icon='MESH_GRID')
        box.prop(g.csc, "paint_reduce")
        r = box.row()
        r.enabled = g.csc.paint_reduce
        r.prop(g.csc, "paint_pct")
        layout.operator("csc.rebuild", icon='FILE_REFRESH')


class CSC_PT_finish(_CSC_Sub, bpy.types.Panel):
    bl_idname = "CSC_PT_finish"
    bl_label = "実体化"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        g, l = _active_layer(context)
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="散らした物を普通のオブジェクトにします。")
        col.label(text="1つずつ動かしたり消したりできます。")
        r = layout.row()
        r.scale_y = 1.3
        r.operator("csc.realize", icon='OUTLINER_OB_GROUP_INSTANCE')


# ====================================================================================
#  Registration
# ====================================================================================
@persistent
def _csc_save_pre(*args):
    for img in bpy.data.images:
        if img.name.startswith("CSC_") and img.is_dirty:
            try:
                img.pack()
            except Exception as e:
                print("CSC: pack failed", img.name, e)


_HANDLERS = ((bpy.app.handlers.depsgraph_update_post, _csc_depsgraph_post),
             (bpy.app.handlers.load_post, _csc_load_post),
             (bpy.app.handlers.save_pre, _csc_save_pre))

classes = (CSC_Asset, CSC_Layer, CSC_Ground, CSC_OT_layer_add, CSC_OT_layer_remove, CSC_OT_layer_move,
           CSC_OT_solo, CSC_OT_assets_reload, CSC_OT_rebuild, CSC_OT_foliage_paint, CSC_OT_foliage_exit,
           CSC_OT_brush_fill, CSC_OT_realize, CSC_UL_layers, CSC_PT_main, CSC_PT_assets, CSC_PT_brush,
           CSC_PT_place, CSC_PT_mask, CSC_PT_perf, CSC_PT_finish)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Object.csc = PointerProperty(type=CSC_Ground)
    for lst, fn in _HANDLERS:
        for h in list(lst):
            if getattr(h, "__name__", "") == fn.__name__:
                lst.remove(h)
        lst.append(fn)
    # bpy.data is not available while add-ons register: initialise a moment later
    if not bpy.app.timers.is_registered(_deferred_init):
        bpy.app.timers.register(_deferred_init, first_interval=0.1)


def _deferred_init():
    try:
        _csc_load_post()
    except Exception as e:
        print("CSC: init failed", e)
    return None


def unregister():
    _fp_reset()
    for fn in (_run_pending, _deferred_init):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)
    for lst, fn in _HANDLERS:
        for h in list(lst):
            if getattr(h, "__name__", "") == fn.__name__:
                lst.remove(h)
    del bpy.types.Object.csc
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
