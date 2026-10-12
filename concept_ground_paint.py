bl_info = {
    "name": "Concept Ground Paint",
    "author": "Claude",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Concept",
    "description": "UE5のランドスケープペイントのように、地面をマテリアルレイヤーや調整レイヤーで塗り分ける",
    "category": "Paint",
}

import bpy
import os
import re
import uuid
import numpy as np
from bpy.props import (StringProperty, FloatProperty, IntProperty, BoolProperty,
                       EnumProperty, FloatVectorProperty, PointerProperty, CollectionProperty)
from bpy.app.handlers import persistent

UV_NAME = "CGP_UV"
HL_NAME = "CGP_HeightLerp"
HL_VER = 4
IMG_FMT = 1  # paint image channels: R = weight, G = softness * weight, B = noise * weight
KEYS = ("Color", "Roughness", "Metallic", "Normal", "Height")
DEBUG_COLORS = [(0.3, 0.3, 0.3), (0.9, 0.2, 0.15), (0.2, 0.75, 0.2), (0.2, 0.4, 0.95),
                (0.95, 0.8, 0.1), (0.8, 0.3, 0.9), (0.1, 0.85, 0.85), (1.0, 0.5, 0.1)]

ADJ_TYPES = [
    ('HSV', "明るさ・彩度・色相", "塗った所の明るさ・彩度・色相を変える"),
    ('ROUGH', "ラフネス", "塗った所のラフネス（ツヤ）を指定の値にする"),
    ('METAL', "メタル", "塗った所のメタル（金属っぽさ）を指定の値にする"),
    ('WET', "濡れ", "塗った所を濡らす（暗く・ツヤあり・水たまりで平ら）"),
    ('TINT', "色をのせる", "塗った所に色を重ねる"),
]
ADJ_LABELS = {a[0]: a[1] for a in ADJ_TYPES}
ADJ_ICONS = {'HSV': 'MOD_HUE_SATURATION', 'ROUGH': 'MATSHADERBALL', 'METAL': 'SHADING_RENDERED',
             'WET': 'MOD_OCEAN', 'TINT': 'MOD_TINT'}
ADJ_PARAMS = ("adj_hue", "adj_sat", "adj_val", "adj_rough", "adj_metal",
              "wet_dark", "wet_rough", "wet_flat", "tint_color")

_SUSPEND = [0]

# ------------------------------------------------------------------ node copy helpers
_SKIP = {"rna_type", "name", "type", "inputs", "outputs", "internal_links", "parent", "select",
         "dimensions", "bl_idname", "bl_label", "bl_description", "bl_icon", "bl_static_type",
         "bl_width_default", "bl_width_min", "bl_width_max", "bl_height_default", "bl_height_min",
         "bl_height_max", "location_absolute", "is_active_output", "warning_propagation"}


def _sock_by_id(coll, ident):
    for s in coll:
        if s.identifier == ident:
            return s
    return None


def _src(sock):
    if sock is not None and sock.is_linked:
        lk = sock.links
        if lk:
            return lk[0].from_socket
    return None


def _copy_ramp(src, dst):
    for a in ("color_mode", "hue_interpolation", "interpolation"):
        try:
            setattr(dst, a, getattr(src, a))
        except Exception:
            pass
    els = dst.elements
    while len(els) > 1:
        els.remove(els[-1])
    for i, e in enumerate(src.elements):
        el = els[0] if i == 0 else els.new(e.position)
        el.position = e.position
        el.color = tuple(e.color)


def _copy_curves(src, dst):
    for a in ("black_level", "white_level", "clip_min_x", "clip_min_y", "clip_max_x",
              "clip_max_y", "use_clip", "extend", "tone"):
        try:
            setattr(dst, a, getattr(src, a))
        except Exception:
            pass
    for cs, cd in zip(src.curves, dst.curves):
        pts = cd.points
        while len(pts) > 2:
            pts.remove(pts[-1])
        for i, p in enumerate(cs.points):
            if i < 2:
                pts[i].location = tuple(p.location)
            else:
                pts.new(p.location[0], p.location[1])
        for p, q in zip(cs.points, pts):
            try:
                q.handle_type = p.handle_type
            except Exception:
                pass
    try:
        dst.update()
    except Exception:
        pass


def _copy_node(src, tree):
    dst = tree.nodes.new(src.bl_idname)
    props = [p for p in src.bl_rna.properties if p.identifier not in _SKIP and not p.is_readonly]
    props.sort(key=lambda p: 0 if p.type == 'ENUM' else 1)
    for p in props:
        try:
            setattr(dst, p.identifier, getattr(src, p.identifier))
        except Exception:
            pass
    if getattr(src, "color_ramp", None) is not None:
        _copy_ramp(src.color_ramp, dst.color_ramp)
    m = getattr(src, "mapping", None)
    if m is not None and hasattr(m, "curves"):
        _copy_curves(m, dst.mapping)
    for cs, cd in ((src.inputs, dst.inputs), (src.outputs, dst.outputs)):
        for s in cs:
            d = _sock_by_id(cd, s.identifier)
            if d is None:
                continue
            d.hide = s.hide
            if hasattr(s, "default_value") and hasattr(d, "default_value"):
                v = s.default_value
                try:
                    d.default_value = tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
                except Exception:
                    try:
                        d.default_value = v
                    except Exception:
                        pass
    return dst


def _copy_nodes(src_tree, dst_tree):
    m = {}
    for n in src_tree.nodes:
        if n.type in ('OUTPUT_MATERIAL', 'OUTPUT_AOV', 'OUTPUT_LIGHT', 'OUTPUT_WORLD'):
            continue
        m[n.name] = _copy_node(n, dst_tree)
    for n in src_tree.nodes:
        if n.name in m and n.parent is not None and n.parent.name in m:
            m[n.name].parent = m[n.parent.name]
    for l in src_tree.links:
        a = m.get(l.from_node.name)
        b = m.get(l.to_node.name)
        if a is None or b is None:
            continue
        fs = _sock_by_id(a.outputs, l.from_socket.identifier)
        ts = _sock_by_id(b.inputs, l.to_socket.identifier)
        if fs is not None and ts is not None:
            nl = dst_tree.links.new(fs, ts)
            nl.is_muted = l.is_muted
    return m


def _find_output(tree):
    outs = [n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL']
    for n in outs:
        if n.is_active_output:
            return n
    return outs[0] if outs else None


class _NB:
    """Small helper for building node trees from code."""

    def __init__(self, tree):
        self.N = tree.nodes
        self.L = tree.links
        self.x = 0
        self.y = 0

    def _place(self, n):
        n.location = (self.x, self.y)
        self.y -= 45

    def link(self, v, sock):
        if isinstance(v, bpy.types.NodeSocket):
            self.L.new(v, sock)
        elif v is not None:
            sock.default_value = v

    def math(self, op, a, b=None, clamp=False, name=None):
        n = self.N.new("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        n.hide = True
        self._place(n)
        if name:
            n.name = n.label = name
        self.link(a, n.inputs[0])
        if b is not None:
            self.link(b, n.inputs[1])
        return n.outputs[0]

    def mix(self, dt, fac, a, b, blend='MIX'):
        n = self.N.new("ShaderNodeMix")
        n.data_type = dt
        if dt == 'RGBA':
            n.blend_type = blend
        n.hide = True
        self._place(n)
        ia, ib, o = {'RGBA': (6, 7, 2), 'FLOAT': (2, 3, 0), 'VECTOR': (4, 5, 1)}[dt]
        self.link(fac, n.inputs[0])
        self.link(a, n.inputs[ia])
        self.link(b, n.inputs[ib])
        return n.outputs[o]

    def value(self, name, v):
        n = self.N.new("ShaderNodeValue")
        n.name = n.label = name
        n.outputs[0].default_value = v
        self._place(n)
        return n.outputs[0]

    def node(self, idname, name=None):
        n = self.N.new(idname)
        if name:
            n.name = n.label = name
        self._place(n)
        return n


# ------------------------------------------------------------------ node groups
def ensure_heightlerp():
    g = bpy.data.node_groups.get(HL_NAME)
    if g is not None and g.get("cgp_ver") == HL_VER:
        return g
    if g is None:
        g = bpy.data.node_groups.new(HL_NAME, "ShaderNodeTree")
    g.nodes.clear()
    g.interface.clear()
    g["cgp_ver"] = HL_VER
    types_ = {"Color": "NodeSocketColor", "Roughness": "NodeSocketFloat", "Metallic": "NodeSocketFloat",
              "Normal": "NodeSocketVector", "Height": "NodeSocketFloat"}
    for p in ("A ", "B "):
        for k in KEYS:
            g.interface.new_socket(name=p + k, in_out='INPUT', socket_type=types_[k])
    for nm in ("Weight", "Softness", "Height Strength"):
        g.interface.new_socket(name=nm, in_out='INPUT', socket_type="NodeSocketFloat")
    for k in KEYS:
        g.interface.new_socket(name=k, in_out='OUTPUT', socket_type=types_[k])
    g.interface.new_socket(name="Factor", in_out='OUTPUT', socket_type="NodeSocketFloat")
    N, L = g.nodes, g.links
    gi = N.new("NodeGroupInput"); gi.location = (-1400, 0)
    go = N.new("NodeGroupOutput"); go.location = (600, 0)
    I, O = gi.outputs, go.inputs

    def math(op, a, b=None, c=None, clamp=False):
        n = N.new("ShaderNodeMath"); n.operation = op; n.use_clamp = clamp
        for idx, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, bpy.types.NodeSocket):
                L.new(v, n.inputs[idx])
            else:
                n.inputs[idx].default_value = v
        return n.outputs[0]

    w = I["Weight"]
    soft = math('MAXIMUM', math('MINIMUM', I["Softness"], 1.0), 0.001)
    # the softer the stroke, the less the material heights cut into the edge, so a soft brush always
    # blurs. hs * (1 - soft) also keeps weight 0 / 1 at exactly 0 / 1 for any softness.
    hs = math('MULTIPLY', math('MINIMUM', I["Height Strength"], 0.999), math('SUBTRACT', 1.0, soft))
    a = math('ADD', math('MULTIPLY', I["A Height"], hs), math('SUBTRACT', 1.0, w))
    b = math('ADD', math('MULTIPLY', I["B Height"], hs), w)
    mx = math('SUBTRACT', math('MAXIMUM', a, b), soft)
    wa = math('MAXIMUM', math('SUBTRACT', a, mx), 0.0)
    wb = math('MAXIMUM', math('SUBTRACT', b, mx), 0.0)
    f = math('DIVIDE', wb, math('ADD', wa, wb), clamp=True)

    def mix(dt, A, B):
        n = N.new("ShaderNodeMix"); n.data_type = dt
        ia, ib, o = {'RGBA': (6, 7, 2), 'FLOAT': (2, 3, 0), 'VECTOR': (4, 5, 1)}[dt]
        L.new(f, n.inputs[0]); L.new(A, n.inputs[ia]); L.new(B, n.inputs[ib])
        return n.outputs[o]

    L.new(mix('RGBA', I["A Color"], I["B Color"]), O["Color"])
    L.new(mix('FLOAT', I["A Roughness"], I["B Roughness"]), O["Roughness"])
    L.new(mix('FLOAT', I["A Metallic"], I["B Metallic"]), O["Metallic"])
    nrm = N.new("ShaderNodeVectorMath"); nrm.operation = 'NORMALIZE'
    L.new(mix('VECTOR', I["A Normal"], I["B Normal"]), nrm.inputs[0])
    L.new(nrm.outputs[0], O["Normal"])
    L.new(mix('FLOAT', I["A Height"], I["B Height"]), O["Height"])
    L.new(f, O["Factor"])
    return g


def build_layer_group(mat, remap):
    gname = "CGP_Layer::%s::%s" % (mat.name, "W" if remap else "O")
    g = bpy.data.node_groups.get(gname)
    if g is None:
        g = bpy.data.node_groups.new(gname, "ShaderNodeTree")
    g.nodes.clear()
    g.interface.clear()
    g.interface.new_socket(name="Vector", in_out='INPUT', socket_type='NodeSocketVector')
    types_ = {"Color": "NodeSocketColor", "Roughness": "NodeSocketFloat", "Metallic": "NodeSocketFloat",
              "Normal": "NodeSocketVector", "Height": "NodeSocketFloat"}
    for k in KEYS:
        g.interface.new_socket(name=k, in_out='OUTPUT', socket_type=types_[k])
    N, L = g.nodes, g.links
    gi = N.new("NodeGroupInput"); gi.location = (-2200, 0)
    go = N.new("NodeGroupOutput"); go.location = (800, 0)
    vals = {"Color": tuple(mat.diffuse_color), "Roughness": mat.roughness, "Metallic": mat.metallic,
            "Normal": None, "Height": None}
    src_tree = mat.node_tree if (mat.use_nodes and mat.node_tree) else None
    if src_tree is not None:
        m = _copy_nodes(src_tree, g)
        out = _find_output(src_tree)
        bsdf = None
        if out is not None:
            s = _src(out.inputs[0])
            if s is not None and s.node.type == 'BSDF_PRINCIPLED':
                bsdf = s.node
        if bsdf is None:
            bsdf = next((n for n in src_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if bsdf is not None:
            for key, sn in (("Color", "Base Color"), ("Roughness", "Roughness"),
                            ("Metallic", "Metallic"), ("Normal", "Normal")):
                si = bsdf.inputs.get(sn)
                if si is None:
                    continue
                fs = _src(si)
                if fs is not None and fs.node.name in m:
                    vals[key] = _sock_by_id(m[fs.node.name].outputs, fs.identifier)
                elif key == "Color":
                    vals[key] = tuple(si.default_value)
                elif key != "Normal":
                    vals[key] = si.default_value
            N.remove(m[bsdf.name])
        if out is not None:
            ds = _src(_sock_by_id(out.inputs, "Displacement"))
            if ds is not None and ds.node.type == 'DISPLACEMENT':
                hsrc = _src(ds.node.inputs.get("Height"))
                if hsrc is not None and hsrc.node.name in m:
                    vals["Height"] = _sock_by_id(m[hsrc.node.name].outputs, hsrc.identifier)
        if remap:
            vin = gi.outputs["Vector"]
            for n in list(N):
                if n.type == 'TEX_COORD':
                    for o in n.outputs:
                        if o.name in ("UV", "Generated", "Object"):
                            for lk in list(o.links):
                                L.new(vin, lk.to_socket)
                elif n.type == 'UVMAP':
                    for o in n.outputs:
                        for lk in list(o.links):
                            L.new(vin, lk.to_socket)
                elif n.type == 'NORMAL_MAP':
                    n.uv_map = UV_NAME
                elif n.type.startswith('TEX_') and n.type not in ('TEX_ENVIRONMENT', 'TEX_SKY', 'TEX_IES', 'TEX_POINTDENSITY'):
                    vi = n.inputs.get("Vector")
                    if vi is not None and not vi.is_linked:
                        L.new(vin, vi)
    # outputs
    v = vals["Color"]
    if not isinstance(v, bpy.types.NodeSocket):
        rgb = N.new("ShaderNodeRGB"); rgb.location = (400, 300)
        rgb.outputs[0].default_value = v
        v = rgb.outputs[0]
    color_sock = v
    L.new(v, go.inputs["Color"])
    for k in ("Roughness", "Metallic"):
        v = vals[k]
        if not isinstance(v, bpy.types.NodeSocket):
            vn = N.new("ShaderNodeValue"); vn.outputs[0].default_value = v
            v = vn.outputs[0]
        L.new(v, go.inputs[k])
    v = vals["Normal"]
    if v is None:
        v = N.new("ShaderNodeNewGeometry").outputs["Normal"]
    L.new(v, go.inputs["Normal"])
    v = vals["Height"]
    if v is None:
        bw = N.new("ShaderNodeRGBToBW"); L.new(color_sock, bw.inputs[0])
        pw = N.new("ShaderNodeMath"); pw.operation = 'POWER'; pw.inputs[1].default_value = 0.4545
        L.new(bw.outputs[0], pw.inputs[0])
        v = pw.outputs[0]
    L.new(v, go.inputs["Height"])
    return g


# ------------------------------------------------------------------ textures -> material
TEX_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".tga", ".bmp", ".webp", ".hdr")
_CH_TOKENS = {
    'ARM': {"arm", "orm"},
    'NORMAL': {"normal", "normals", "nor", "nrm", "norm", "nrml", "normalgl", "normaldx"},
    'HEIGHT': {"height", "displacement", "disp", "displace", "bump", "hgt"},
    'ROUGH': {"roughness", "rough", "rgh"},
    'GLOSS': {"gloss", "glossiness"},
    'METAL': {"metallic", "metalness", "metal", "mtl"},
    'AO': {"ao", "ambientocclusion", "ambient", "occlusion", "occ"},
    'SKIP': {"opacity", "alpha", "mask", "spec", "specular", "emission", "emissive", "translucency",
             "sss", "cavity", "preview", "thumb", "thumbnail", "id", "fuzz"},
    'BASE': {"basecolor", "albedo", "diffuse", "diff", "color", "colour", "col", "basecol", "base"},
}
_CH_ORDER = ('ARM', 'NORMAL', 'HEIGHT', 'ROUGH', 'GLOSS', 'METAL', 'AO', 'SKIP', 'BASE')
_ALL_CH = set().union(*_CH_TOKENS.values())
_JUNK = {"gl", "dx", "directx", "opengl", "jpg", "jpeg", "png", "exr", "tif", "tiff", "tga", "map", "tex", "texture"}


def _classify_texture(path):
    """(channel, is_directx_normal, set_name). The channel word is the last one in the file name,
    e.g. rock_wall_nor_gl_2k -> NORMAL / rock_wall, Metal_Plate_Color -> BASE / Metal_Plate."""
    stem = os.path.splitext(os.path.basename(path))[0]
    toks = [t for t in re.split(r"[\s_\-\.]+", stem) if t]
    low = [t.lower() for t in toks]
    k = None
    for j in range(len(low) - 1, -1, -1):
        if low[j] in _ALL_CH:
            k = j
            break
    ch = None
    drop = set()
    if k is not None:
        ch = next(c for c in _CH_ORDER if low[k] in _CH_TOKENS[c])
        drop.add(k)
        j = k - 1
        while j >= 0 and low[j] in _CH_TOKENS[ch]:
            drop.add(j)
            j -= 1
    dx = ch == 'NORMAL' and any(t in ("dx", "directx", "normaldx") for t in low)
    keep = [t for j, (t, tl) in enumerate(zip(toks, low))
            if j not in drop and tl not in _JUNK and not re.fullmatch(r"\d+k", tl)]
    return ch, dx, ("_".join(keep) or "Texture")


def group_texture_files(paths):
    groups = {}
    for p in paths:
        if os.path.splitext(p)[1].lower() not in TEX_EXTS or not os.path.isfile(p):
            continue
        ch, dx, name = _classify_texture(p)
        if ch == 'SKIP':
            continue
        if ch is None:
            ch = 'BASE'
        g = groups.setdefault(name, {})
        if ch == 'NORMAL':
            if 'NORMAL' in g and not g.get('NORMAL_DX') and dx:
                continue
            g['NORMAL'] = p
            g['NORMAL_DX'] = dx
        elif ch not in g:
            g[ch] = p
    return groups


def build_material_from_textures(name, g):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    N, L = nt.nodes, nt.links
    bsdf = next(n for n in N if n.type == 'BSDF_PRINCIPLED')
    out = next(n for n in N if n.type == 'OUTPUT_MATERIAL')
    bsdf.location = (300, 0)
    out.location = (650, 0)
    tc = N.new("ShaderNodeTexCoord"); tc.location = (-1100, 0)
    y = [400]

    def tex(path, data):
        img = bpy.data.images.load(path, check_existing=True)
        if data:
            try:
                img.colorspace_settings.name = "Non-Color"
            except TypeError:
                pass
        n = N.new("ShaderNodeTexImage")
        n.image = img
        n.location = (-800, y[0])
        y[0] -= 300
        L.new(tc.outputs["UV"], n.inputs["Vector"])
        return n

    ao = None
    if 'ARM' in g:
        t = tex(g['ARM'], True)
        sep = N.new("ShaderNodeSeparateColor"); sep.location = (-500, y[0] + 300)
        L.new(t.outputs["Color"], sep.inputs[0])
        ao = sep.outputs[0]
        if 'ROUGH' not in g and 'GLOSS' not in g:
            L.new(sep.outputs[1], bsdf.inputs["Roughness"])
        if 'METAL' not in g:
            L.new(sep.outputs[2], bsdf.inputs["Metallic"])
    if 'AO' in g:
        ao = tex(g['AO'], True).outputs["Color"]
    if 'BASE' in g:
        col = tex(g['BASE'], False).outputs["Color"]
        if ao is not None:
            mx = N.new("ShaderNodeMix"); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'
            mx.location = (-200, 300)
            mx.inputs[0].default_value = 0.5
            L.new(col, mx.inputs[6]); L.new(ao, mx.inputs[7])
            col = mx.outputs[2]
        L.new(col, bsdf.inputs["Base Color"])
    if 'ROUGH' in g:
        L.new(tex(g['ROUGH'], True).outputs["Color"], bsdf.inputs["Roughness"])
    elif 'GLOSS' in g:
        inv = N.new("ShaderNodeInvert"); inv.location = (-300, y[0] + 300)
        L.new(tex(g['GLOSS'], True).outputs["Color"], inv.inputs["Color"])
        L.new(inv.outputs["Color"], bsdf.inputs["Roughness"])
    if 'METAL' in g:
        L.new(tex(g['METAL'], True).outputs["Color"], bsdf.inputs["Metallic"])
    if 'NORMAL' in g:
        t = tex(g['NORMAL'], True)
        src = t.outputs["Color"]
        if g.get('NORMAL_DX'):
            sep = N.new("ShaderNodeSeparateColor")
            cmb = N.new("ShaderNodeCombineColor")
            inv = N.new("ShaderNodeMath"); inv.operation = 'SUBTRACT'; inv.inputs[0].default_value = 1.0
            L.new(src, sep.inputs[0])
            L.new(sep.outputs[0], cmb.inputs[0])
            L.new(sep.outputs[1], inv.inputs[1])
            L.new(inv.outputs[0], cmb.inputs[1])
            L.new(sep.outputs[2], cmb.inputs[2])
            src = cmb.outputs[0]
        nm = N.new("ShaderNodeNormalMap"); nm.location = (0, -400)
        L.new(src, nm.inputs["Color"])
        L.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if 'HEIGHT' in g:
        t = tex(g['HEIGHT'], True)
        dp = N.new("ShaderNodeDisplacement"); dp.location = (300, -500)
        dp.inputs["Scale"].default_value = 0.05
        L.new(t.outputs["Color"], dp.inputs["Height"])
        L.new(dp.outputs[0], out.inputs["Displacement"])
    return mat


# ------------------------------------------------------------------ data helpers
def ensure_uv(obj):
    me = obj.data
    uv = me.uv_layers.get(UV_NAME)
    if uv is None:
        uv = me.uv_layers.new(name=UV_NAME, do_init=False)
        if uv is None:
            raise RuntimeError("UVマップを追加できません（上限8個）")
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mn = co[:, :2].min(0)
    size = np.maximum(co[:, :2].max(0) - mn, 1e-6)
    lv = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", lv)
    uvs = ((co[lv, :2] - mn) / size).astype(np.float32)
    uv.data.foreach_set("uv", uvs.ravel())
    return uv


def ensure_layer_image(obj, layer):
    img = layer.image
    if img is None:
        res = int(obj.cgp.resolution)
        img = bpy.data.images.new("CGP_%s_%s" % (obj.name, layer.uid), res, res, alpha=False, float_buffer=False)
        img.generated_color = (0, 0, 0, 1)
        try:
            img.colorspace_settings.name = "Non-Color"
        except TypeError:
            pass
        layer.image = img
        layer.fmt = IMG_FMT
    return img


def migrate_layer_image(s, layer):
    """v0.1 images were plain grey weights; v0.2 also stores softness / noise per pixel."""
    img = layer.image
    if img is None or layer.fmt >= IMG_FMT:
        layer.fmt = IMG_FMT
        return
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    a = a.reshape(-1, 4)
    r = a[:, 0].copy()
    a[:, 1] = r * s.softness
    a[:, 2] = r * s.edge_noise
    a[:, 3] = 1.0
    img.pixels.foreach_set(a.ravel())
    img.update()
    img.pack()
    layer.fmt = IMG_FMT


def _layer_sockets(nb, i, layer, tc, cache):
    if layer.material is not None:
        remap = layer.coord_mode == 'WORLD'
        key = (layer.material.name, remap)
        g = cache.get(key)
        if g is None:
            g = build_layer_group(layer.material, remap)
            cache[key] = g
        gn = nb.node("ShaderNodeGroup")
        gn.node_tree = g
        gn.name = gn.label = "L%d_group" % i
        if remap:
            sc = nb.node("ShaderNodeVectorMath", "L%d_tile" % i)
            sc.operation = 'SCALE'
            nb.L.new(tc.outputs["Object"], sc.inputs[0])
            sc.inputs["Scale"].default_value = 1.0 / max(layer.tile_size, 1e-4)
            nb.L.new(sc.outputs[0], gn.inputs["Vector"])
        return {k: gn.outputs[k] for k in KEYS}
    rgb = nb.node("ShaderNodeRGB", "L%d_rgb" % i)
    rgb.outputs[0].default_value = (*layer.color, 1.0)
    r = nb.value("L%d_rough" % i, layer.roughness)
    geo = nb.node("ShaderNodeNewGeometry")
    nz = nb.node("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 2.0
    nb.L.new(tc.outputs["Object"], nz.inputs["Vector"])
    return {"Color": rgb.outputs[0], "Roughness": r, "Metallic": 0.0,
            "Normal": geo.outputs["Normal"], "Height": nz.outputs["Fac"]}


def _avg_image(img):
    """Average colour (linear RGB) of an image, read from a tiny scaled copy."""
    tmp = None
    try:
        if img is None or img.size[0] == 0:
            return None
        tmp = img.copy()
        tmp.scale(16, 16)
        a = np.empty(16 * 16 * 4, np.float32)
        tmp.pixels.foreach_get(a)
        c = a.reshape(-1, 4)[:, :3].mean(0)
        if not img.is_float and img.colorspace_settings.name.lower().startswith("srgb"):
            c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return c
    except Exception:
        return None
    finally:
        if tmp is not None:
            bpy.data.images.remove(tmp)


def _guess_color(sock, depth=0):
    """Rough average colour of whatever feeds a colour socket."""
    if sock is None or depth > 8:
        return None
    src = _src(sock)
    if src is None:
        v = getattr(sock, "default_value", None)
        if v is None:
            return None
        try:
            return np.array(tuple(v)[:3], dtype=np.float32) if len(v) >= 3 else None
        except TypeError:
            return np.array([v, v, v], dtype=np.float32)
    n = src.node
    t = n.type
    if t == 'TEX_IMAGE':
        return _avg_image(n.image)
    if t == 'VALTORGB':
        r = n.color_ramp
        return np.mean([tuple(r.evaluate(k / 15.0))[:3] for k in range(16)], 0)
    if t == 'RGB':
        return np.array(tuple(n.outputs[0].default_value)[:3])
    if t in ('MIX', 'MIX_RGB'):
        if t == 'MIX':
            if n.data_type != 'RGBA':
                return None
            fa, a, b = n.inputs[0], n.inputs[6], n.inputs[7]
        else:
            fa, a, b = n.inputs[0], n.inputs[1], n.inputs[2]
        ga = _guess_color(a, depth + 1)
        gb = _guess_color(b, depth + 1)
        if ga is None:
            return gb
        if gb is None:
            return ga
        f = 0.5 if fa.is_linked else float(fa.default_value)
        if n.blend_type == 'MULTIPLY':
            return ga * (1 - f + f * gb)
        return ga * (1 - f) + gb * f
    if t in ('HUE_SAT', 'BRIGHTCONTRAST', 'GAMMA', 'INVERT', 'CURVE_RGB', 'REROUTE'):
        inp = n.inputs.get("Color")
        if inp is None and len(n.inputs):
            inp = n.inputs[0]
        g = _guess_color(inp, depth + 1)
        if g is not None and t == 'INVERT':
            g = 1.0 - g
        return g
    return None


def guess_material_color(mat):
    try:
        if mat.use_nodes and mat.node_tree:
            out = _find_output(mat.node_tree)
            b = None
            if out is not None:
                s = _src(out.inputs[0])
                if s is not None and s.node.type == 'BSDF_PRINCIPLED':
                    b = s.node
            if b is None:
                b = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
            if b is not None:
                c = _guess_color(b.inputs.get("Base Color"))
                if c is not None:
                    return tuple(float(x) for x in np.clip(c, 0.0, 1.0))
    except Exception as e:
        print("CGP: colour guess failed", mat.name, e)
    return tuple(mat.diffuse_color[:3])


def _light_sockets(nb, i, layer, geo_normal):
    """Cheap stand-in for a layer while painting: one flat colour, no textures."""
    if layer.material is None:
        rgb = nb.node("ShaderNodeRGB", "L%d_rgb" % i)
        rgb.outputs[0].default_value = (*layer.color, 1.0)
        rough = nb.value("L%d_rough" % i, layer.roughness)
    else:
        if not layer.preview_set:
            layer.preview_color = guess_material_color(layer.material)
            layer.preview_set = True
        rgb = nb.node("ShaderNodeRGB", "L%d_prev" % i)
        rgb.outputs[0].default_value = (*layer.preview_color, 1.0)
        rough = 0.7
    return {"Color": rgb.outputs[0], "Roughness": rough, "Metallic": 0.0, "Normal": geo_normal, "Height": 0.5}


def _painted_weight(nb, i, layer, img, uvn, noise_ch):
    """Returns (weight, softness) sockets decoded from the paint image."""
    tex = nb.node("ShaderNodeTexImage", "L%d_splat" % i)
    tex.image = img
    tex.interpolation = 'Linear'
    tex.extension = 'EXTEND'
    tex.hide = True
    nb.L.new(uvn.outputs["UV"], tex.inputs["Vector"])
    sep = nb.node("ShaderNodeSeparateColor")
    sep.hide = True
    nb.L.new(tex.outputs["Color"], sep.inputs[0])
    R, G, B = sep.outputs[0], sep.outputs[1], sep.outputs[2]
    rr = nb.math('MAXIMUM', R, 1e-4)
    soft = nb.math('MAXIMUM', nb.math('DIVIDE', G, rr, clamp=True), 0.001)
    noise_amt = nb.math('DIVIDE', B, rr, clamp=True)
    w = nb.math('MULTIPLY', R, layer.opacity, clamp=True, name="L%d_opacity" % i)
    # noise only moves the in-between weights: 0 stays 0 and 1 stays 1
    nzc = nb.math('MULTIPLY', nb.math('MULTIPLY', nb.math('SUBTRACT', noise_ch, 0.5), noise_amt), 2.0)
    env = nb.math('MULTIPLY', nb.math('MULTIPLY', w, nb.math('SUBTRACT', 1.0, w)), 4.0)
    w2 = nb.math('ADD', w, nb.math('MULTIPLY', nzc, env), clamp=True)
    return w2, soft


def _adjust_mask(nb, w, soft):
    mr = nb.node("ShaderNodeMapRange")
    mr.interpolation_type = 'SMOOTHSTEP'
    mr.clamp = True
    mr.hide = True
    half = nb.math('MULTIPLY', soft, 0.5)
    nb.L.new(w, mr.inputs[0])
    nb.L.new(nb.math('SUBTRACT', 0.5, half), mr.inputs[1])
    nb.L.new(nb.math('ADD', 0.5, half), mr.inputs[2])
    mr.inputs[3].default_value = 0.0
    mr.inputs[4].default_value = 1.0
    return mr.outputs[0]


def _apply_adjust(nb, i, layer, prev, mask):
    out = dict(prev)
    t = layer.adj_type
    if t == 'HSV':
        hs = nb.node("ShaderNodeHueSaturation")
        nb.L.new(nb.math('ADD', nb.value("L%d_adj_hue" % i, layer.adj_hue), 0.5), hs.inputs["Hue"])
        nb.L.new(nb.value("L%d_adj_sat" % i, layer.adj_sat), hs.inputs["Saturation"])
        nb.L.new(nb.value("L%d_adj_val" % i, layer.adj_val), hs.inputs["Value"])
        nb.L.new(mask, hs.inputs["Fac"])
        nb.L.new(prev["Color"], hs.inputs["Color"])
        out["Color"] = hs.outputs["Color"]
    elif t == 'ROUGH':
        out["Roughness"] = nb.mix('FLOAT', mask, prev["Roughness"], nb.value("L%d_adj_rough" % i, layer.adj_rough))
    elif t == 'METAL':
        out["Metallic"] = nb.mix('FLOAT', mask, prev["Metallic"], nb.value("L%d_adj_metal" % i, layer.adj_metal))
    elif t == 'WET':
        keep = nb.math('SUBTRACT', 1.0, nb.value("L%d_wet_dark" % i, layer.wet_dark))
        out["Color"] = nb.mix('RGBA', mask, prev["Color"], keep, blend='MULTIPLY')
        out["Roughness"] = nb.mix('FLOAT', mask, prev["Roughness"], nb.value("L%d_wet_rough" % i, layer.wet_rough))
        geo = nb.node("ShaderNodeNewGeometry")
        flat = nb.math('MULTIPLY', mask, nb.value("L%d_wet_flat" % i, layer.wet_flat))
        nrm = nb.node("ShaderNodeVectorMath")
        nrm.operation = 'NORMALIZE'
        nb.L.new(nb.mix('VECTOR', flat, prev["Normal"], geo.outputs["Normal"]), nrm.inputs[0])
        out["Normal"] = nrm.outputs[0]
    elif t == 'TINT':
        rgb = nb.node("ShaderNodeRGB", "L%d_tint_color" % i)
        rgb.outputs[0].default_value = (*layer.tint_color, 1.0)
        out["Color"] = nb.mix('RGBA', mask, prev["Color"], rgb.outputs[0], blend=layer.tint_blend)
    return out


def _want_light(obj):
    return obj.cgp.light_paint and obj.mode == 'TEXTURE_PAINT'


def _is_light(obj):
    m = obj.cgp.material
    return m is not None and bool(m.get("cgp_light", False))


def rebuild_material(obj, light=None):
    s = obj.cgp
    if light is None:
        light = _want_light(obj)
    ensure_uv(obj)
    mat = s.material
    if mat is None:
        mat = bpy.data.materials.new("CGP_%s" % obj.name)
        mat.use_nodes = True
        s.material = mat
    nt = mat.node_tree
    nt.nodes.clear()
    nb = _NB(nt)
    hl = ensure_heightlerp()
    n_layers = len(s.layers)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (n_layers * 600 + 500, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (n_layers * 600 + 150, 0)
    nb.x, nb.y = -900, 600
    g_hs = nb.value("G_hs", s.height_strength)
    g_dbg = nb.value("G_debug", 1.0 if s.show_debug else 0.0)
    g_nzs = nb.value("G_noise_scale", s.edge_noise_scale)
    tc = nb.node("ShaderNodeTexCoord")
    uvn = nb.node("ShaderNodeUVMap")
    uvn.uv_map = UV_NAME
    geo_normal = nb.node("ShaderNodeNewGeometry").outputs["Normal"]
    # one shared edge-noise for every layer (its 3 colour channels, flipped every 3 layers)
    noise_chs = None
    if n_layers > 1:
        nz = nb.node("ShaderNodeTexNoise")
        nz.hide = True
        nb.L.new(tc.outputs["Object"], nz.inputs["Vector"])
        nb.L.new(g_nzs, nz.inputs["Scale"])
        nz.inputs["Detail"].default_value = 2.0
        nsep = nb.node("ShaderNodeSeparateColor")
        nsep.hide = True
        nb.L.new(nz.outputs["Color"], nsep.inputs[0])
        noise_chs = nsep.outputs
    cache = {}
    prev = None
    for i, layer in enumerate(s.layers):
        nb.x, nb.y = i * 600, 600
        dbg_col = (*DEBUG_COLORS[i % len(DEBUG_COLORS)], 1.0)
        is_mat = i == 0 or layer.kind == 'MATERIAL'
        if is_mat:
            if light:
                outs = _light_sockets(nb, i, layer, geo_normal)
            else:
                outs = _layer_sockets(nb, i, layer, tc, cache)
            outs["Height"] = nb.math('MULTIPLY_ADD', outs["Height"], layer.height_contrast, clamp=True, name="L%d_h" % i)
            nt.nodes["L%d_h" % i].inputs[2].default_value = layer.height_offset
            outs["Color"] = nb.mix('RGBA', g_dbg, outs["Color"], dbg_col)
        if i == 0:
            prev = outs
            continue
        img = ensure_layer_image(obj, layer)
        migrate_layer_image(s, layer)
        k = i - 1
        noise_ch = noise_chs[k % 3]
        if (k // 3) % 2:
            noise_ch = nb.math('SUBTRACT', 1.0, noise_ch)
        w, soft = _painted_weight(nb, i, layer, img, uvn, noise_ch)
        if is_mat:
            hn = nb.node("ShaderNodeGroup", "L%d_blend" % i)
            hn.node_tree = hl
            for k in KEYS:
                nb.link(prev[k], hn.inputs["A " + k])
                nb.link(outs[k], hn.inputs["B " + k])
            nb.L.new(w, hn.inputs["Weight"])
            nb.L.new(soft, hn.inputs["Softness"])
            nb.L.new(g_hs, hn.inputs["Height Strength"])
            prev = {k: hn.outputs[k] for k in KEYS}
        else:
            mask = _adjust_mask(nb, w, soft)
            prev = _apply_adjust(nb, i, layer, prev, mask)
            prev["Color"] = nb.mix('RGBA', nb.math('MULTIPLY', mask, g_dbg), prev["Color"], dbg_col)
    if prev is not None:
        nb.link(prev["Color"], bsdf.inputs["Base Color"])
        nb.link(prev["Roughness"], bsdf.inputs["Roughness"])
        nb.link(prev["Metallic"], bsdf.inputs["Metallic"])
        nb.link(prev["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    if len(obj.material_slots) == 0:
        obj.data.materials.append(mat)
    elif obj.material_slots[0].material != mat:
        obj.material_slots[0].material = mat
    mat["cgp_light"] = bool(light)
    if light:
        _watch_obj(obj)
    return mat


# ------------------------------------------------------------------ lighter display while painting
_watch = set()


def _lower_subdiv(obj):
    s = obj.cgp
    if not s.paint_lower_subdiv or obj.get("cgp_subdiv_backup") is not None:
        return
    data = {}
    for md in obj.modifiers:
        if md.type in ('SUBSURF', 'MULTIRES') and md.levels > s.paint_subdiv_level:
            data[md.name] = md.levels
            md.levels = s.paint_subdiv_level
    if data:
        obj["cgp_subdiv_backup"] = data
        _watch_obj(obj)


def _restore_subdiv(obj):
    data = obj.get("cgp_subdiv_backup")
    if data is None:
        return
    for name in list(data.keys()):
        md = obj.modifiers.get(name)
        if md is not None:
            md.levels = int(data[name])
    del obj["cgp_subdiv_backup"]


def _watch_obj(obj):
    """Watch an object until it leaves Texture Paint, then undo the painting-only changes."""
    _watch.add(obj.name)
    if not bpy.app.timers.is_registered(_poll_restore):
        bpy.app.timers.register(_poll_restore, first_interval=0.5)


def _end_paint(obj):
    _restore_subdiv(obj)
    if obj.cgp.enabled and _is_light(obj):
        rebuild_material(obj, light=False)
    _watch.discard(obj.name)


def _poll_restore():
    for name in list(_watch):
        obj = bpy.data.objects.get(name)
        if obj is None:
            _watch.discard(name)
            continue
        if obj.mode != 'TEXTURE_PAINT':
            _end_paint(obj)
    return 0.5 if _watch else None


# ------------------------------------------------------------------ property callbacks
def _layer_index(layer):
    obj = layer.id_data
    for i, l in enumerate(obj.cgp.layers):
        if l.as_pointer() == layer.as_pointer():
            return obj, i
    return obj, -1


def _node(obj, name):
    mat = obj.cgp.material
    if mat is not None and mat.node_tree is not None:
        return mat.node_tree.nodes.get(name)
    return None


def _upd_rebuild(self, context):
    obj = self.id_data
    if obj.cgp.enabled and not _SUSPEND[0]:
        rebuild_material(obj)


def _upd_material(self, context):
    self.preview_set = False
    _upd_rebuild(self, context)


def _upd_preview(self, context):
    obj, i = _layer_index(self)
    n = _node(obj, "L%d_prev" % i)
    if n is not None:
        n.outputs[0].default_value = (*self.preview_color, 1.0)


def _upd_light(self, context):
    obj = self.id_data
    if self.enabled and obj.mode == 'TEXTURE_PAINT':
        rebuild_material(obj)


def _upd_tile(self, context):
    obj, i = _layer_index(self)
    n = _node(obj, "L%d_tile" % i)
    if n is not None:
        n.inputs["Scale"].default_value = 1.0 / max(self.tile_size, 1e-4)


def _upd_height(self, context):
    obj, i = _layer_index(self)
    n = _node(obj, "L%d_h" % i)
    if n is not None:
        n.inputs[1].default_value = self.height_contrast
        n.inputs[2].default_value = self.height_offset


def _upd_opacity(self, context):
    obj, i = _layer_index(self)
    n = _node(obj, "L%d_opacity" % i)
    if n is not None:
        n.inputs[1].default_value = self.opacity


def _upd_color(self, context):
    obj, i = _layer_index(self)
    n = _node(obj, "L%d_rgb" % i)
    if n is not None:
        n.outputs[0].default_value = (*self.color, 1.0)
    n = _node(obj, "L%d_rough" % i)
    if n is not None:
        n.outputs[0].default_value = self.roughness


def _upd_param(self, context):
    obj, i = _layer_index(self)
    for p in ADJ_PARAMS:
        n = _node(obj, "L%d_%s" % (i, p))
        if n is None:
            continue
        v = getattr(self, p)
        n.outputs[0].default_value = (*v, 1.0) if p == "tint_color" else v


def _upd_global(self, context):
    obj = self.id_data
    for name, v in (("G_hs", self.height_strength), ("G_debug", 1.0 if self.show_debug else 0.0),
                    ("G_noise_scale", self.edge_noise_scale)):
        n = _node(obj, name)
        if n is not None:
            n.outputs[0].default_value = v


def _apply_brush_colors(ts, s):
    ups = ts.unified_paint_settings
    ups.use_unified_color = True
    fg = (1.0, s.brush_softness, s.brush_noise)
    bg = (0.0, 0.0, 0.0)
    flipped = ups.color[0] < 0.5 and ups.secondary_color[0] > 0.5
    if flipped:
        ups.color, ups.secondary_color = bg, fg
    else:
        ups.color, ups.secondary_color = fg, bg


def _set_canvas(context, img, s):
    ts = context.scene.tool_settings
    ip = ts.image_paint
    ip.mode = 'IMAGE'
    ip.canvas = img
    _apply_brush_colors(ts, s)


def _upd_brush(self, context):
    obj = self.id_data
    if obj.mode == 'TEXTURE_PAINT':
        _apply_brush_colors(context.scene.tool_settings, self)


def _upd_active(self, context):
    obj = self.id_data
    i = self.active_index
    if obj.mode == 'TEXTURE_PAINT' and 0 < i < len(self.layers):
        img = self.layers[i].image
        if img is not None:
            _set_canvas(context, img, self)


def _upd_res(self, context):
    res = int(self.resolution)
    for l in self.layers:
        if l.image is not None and tuple(l.image.size) != (res, res):
            l.image.scale(res, res)


def _upd_subdiv_level(self, context):
    obj = self.id_data
    data = obj.get("cgp_subdiv_backup")
    if data is None:
        return
    for name in list(data.keys()):
        md = obj.modifiers.get(name)
        if md is not None:
            md.levels = min(int(data[name]), self.paint_subdiv_level)


def _mat_poll(self, m):
    return not m.name.startswith("CGP_")


class CGP_Layer(bpy.types.PropertyGroup):
    name: StringProperty(name="名前", default="レイヤー")
    uid: StringProperty()
    fmt: IntProperty(default=0)
    kind: EnumProperty(name="種類", default='MATERIAL', update=_upd_rebuild, items=[
        ('MATERIAL', "マテリアル", "マテリアル（または単色）を塗るレイヤー"),
        ('ADJUST', "調整", "塗った所の明るさやラフネスなどを変えるレイヤー")])
    material: PointerProperty(type=bpy.types.Material, name="マテリアル", poll=_mat_poll, update=_upd_material,
                              description="このレイヤーに使うマテリアル。空なら単色レイヤー")
    preview_color: FloatVectorProperty(name="簡易表示の色", subtype='COLOR', size=3, min=0.0, max=1.0,
                                       default=(0.5, 0.5, 0.5), update=_upd_preview,
                                       description="塗っている間の簡易表示で使う色（マテリアルから自動で推定。変更もできる）")
    preview_set: BoolProperty(default=False)
    coord_mode: EnumProperty(name="座標", update=_upd_rebuild, default='WORLD', items=[
        ('WORLD', "ワールド", "タイルサイズ(m)で地面全体に敷き詰める"),
        ('ORIGINAL', "元のまま", "マテリアル本来の座標（UVなど）をそのまま使う")])
    tile_size: FloatProperty(name="タイルサイズ (m)", default=2.0, min=0.01, soft_max=50.0, update=_upd_tile)
    height_contrast: FloatProperty(name="高さコントラスト", default=1.0, min=0.0, soft_max=4.0, update=_upd_height,
                                   description="レイヤー境界での凹凸の効き方")
    height_offset: FloatProperty(name="高さオフセット", default=0.0, soft_min=-1.0, soft_max=1.0, update=_upd_height,
                                 description="プラスにすると他のレイヤーの上に出やすくなる")
    opacity: FloatProperty(name="不透明度", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_upd_opacity)
    color: FloatVectorProperty(name="色", subtype='COLOR', size=3, min=0.0, max=1.0,
                               default=(0.5, 0.5, 0.5), update=_upd_color)
    roughness: FloatProperty(name="粗さ", default=0.8, min=0.0, max=1.0, subtype='FACTOR', update=_upd_color)
    image: PointerProperty(type=bpy.types.Image)
    # adjustment layers
    adj_type: EnumProperty(name="調整の種類", items=ADJ_TYPES, default='HSV', update=_upd_rebuild)
    adj_hue: FloatProperty(name="色相", default=0.0, min=-0.5, max=0.5, update=_upd_param)
    adj_sat: FloatProperty(name="彩度", default=1.0, min=0.0, soft_max=2.0, update=_upd_param)
    adj_val: FloatProperty(name="明るさ", default=1.0, min=0.0, soft_max=2.0, update=_upd_param)
    adj_rough: FloatProperty(name="ラフネス", default=0.2, min=0.0, max=1.0, subtype='FACTOR', update=_upd_param,
                             description="塗った所のラフネス（0でツルツル、1でザラザラ）")
    adj_metal: FloatProperty(name="メタル", default=1.0, min=0.0, max=1.0, subtype='FACTOR', update=_upd_param)
    wet_dark: FloatProperty(name="暗さ", default=0.6, min=0.0, max=1.0, subtype='FACTOR', update=_upd_param,
                            description="濡れて色が暗くなる量")
    wet_rough: FloatProperty(name="濡れのラフネス", default=0.05, min=0.0, max=1.0, subtype='FACTOR',
                             update=_upd_param, description="小さいほどツヤツヤ")
    wet_flat: FloatProperty(name="水たまり", default=0.6, min=0.0, max=1.0, subtype='FACTOR', update=_upd_param,
                            description="濃く塗った所の凹凸を平らにして水たまりっぽくする")
    tint_color: FloatVectorProperty(name="色", subtype='COLOR', size=3, min=0.0, max=1.0,
                                    default=(0.35, 0.18, 0.08), update=_upd_param)
    tint_blend: EnumProperty(name="重ね方", default='MIX', update=_upd_rebuild, items=[
        ('MIX', "通常", ""), ('MULTIPLY', "乗算", ""), ('OVERLAY', "オーバーレイ", ""),
        ('SCREEN', "スクリーン", ""), ('SOFT_LIGHT', "ソフトライト", "")])


class CGP_Settings(bpy.types.PropertyGroup):
    enabled: BoolProperty(default=False)
    layers: CollectionProperty(type=CGP_Layer)
    active_index: IntProperty(default=0, update=_upd_active)
    material: PointerProperty(type=bpy.types.Material)
    original_material: PointerProperty(type=bpy.types.Material)
    height_strength: FloatProperty(name="高さの影響", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                                   update=_upd_global,
                                   description="くっきりしたブラシで塗った境界を、各レイヤーの凹凸で崩す強さ（やわらかいブラシほど自動で弱まる）")
    edge_noise_scale: FloatProperty(name="ノイズの細かさ", default=1.0, min=0.01, soft_max=20.0,
                                    update=_upd_global, description="1mあたりのノイズの細かさ")
    show_debug: BoolProperty(name="レイヤーを色分け表示", default=False, update=_upd_global)
    resolution: EnumProperty(name="塗りの解像度", default='1024', update=_upd_res, items=[
        ('512', "512", ""), ('1024', "1024", ""), ('2048', "2048", ""), ('4096', "4096", "")])
    # brush (stored into each stroke)
    brush_softness: FloatProperty(name="境界のやわらかさ", default=0.15, min=0.0, max=1.0, subtype='FACTOR',
                                  update=_upd_brush,
                                  description="これから塗るストロークの境界。小さいほどくっきり（凹凸に沿って切り替わる）、1に近いほどぼやっとぼける")
    brush_noise: FloatProperty(name="境界のノイズ", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                               update=_upd_brush, description="これから塗るストロークの境界をノイズで崩す強さ")
    # lighter display while painting
    paint_lower_subdiv: BoolProperty(name="塗るときサブディビジョンを下げる", default=True,
                                     description="ペイント中だけサブディビジョン/マルチレゾの表示レベルを下げて軽くする")
    paint_subdiv_level: IntProperty(name="塗るときのレベル", default=0, min=0, max=6, update=_upd_subdiv_level,
                                    description="0が一番軽い。数字を上げるほど細かく重くなる")
    light_paint: BoolProperty(name="塗っている間は簡易表示", default=True, update=_upd_light,
                              description="ペイント中だけ各マテリアルを単色で表示して軽くする。塗り終了で元の見た目に戻る")
    # v0.1 settings (kept to convert old paint data)
    softness: FloatProperty(default=0.15)
    edge_noise: FloatProperty(default=0.5)


# ------------------------------------------------------------------ operators
def _terrain(context):
    obj = context.active_object
    if obj is not None and obj.type == 'MESH' and obj.cgp.enabled:
        return obj
    return None


class CGP_OT_enable(bpy.types.Operator):
    bl_idname = "cgp.enable"
    bl_label = "地面ペイントを有効化"
    bl_description = "選択中のメッシュを地面ペイント用にセットアップ（今のマテリアルがベースレイヤーになります）"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        o = context.active_object
        return o is not None and o.type == 'MESH'

    def execute(self, context):
        obj = context.active_object
        s = obj.cgp
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        orig = obj.active_material
        if orig is not None and orig.name.startswith("CGP_"):
            orig = s.original_material
        s.original_material = orig
        if len(s.layers) == 0:
            _SUSPEND[0] += 1
            try:
                base = s.layers.add()
                base.uid = uuid.uuid4().hex[:8]
                base.fmt = IMG_FMT
                base.name = "ベース"
                base.color = DEBUG_COLORS[0]
                base.material = orig
                base.coord_mode = 'ORIGINAL' if orig is not None else 'WORLD'
            finally:
                _SUSPEND[0] -= 1
        s.enabled = True
        rebuild_material(obj)
        return {'FINISHED'}


class CGP_OT_disable(bpy.types.Operator):
    bl_idname = "cgp.disable"
    bl_label = "地面ペイントを解除"
    bl_description = "元のマテリアルに戻す（塗ったデータは残るので、再度有効化すれば戻せます）"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        _restore_subdiv(obj)
        obj.cgp.enabled = False
        if len(obj.material_slots) > 0:
            obj.material_slots[0].material = obj.cgp.original_material
        return {'FINISHED'}


def _new_layer(s, kind, adj_type='HSV'):
    l = s.layers.add()
    l.uid = uuid.uuid4().hex[:8]
    l.fmt = IMG_FMT
    l.kind = kind
    if kind == 'ADJUST':
        l.adj_type = adj_type
        l.name = ADJ_LABELS[adj_type]
    else:
        l.name = "レイヤー %d" % (len(s.layers) - 1)
        l.color = DEBUG_COLORS[(len(s.layers) - 1) % len(DEBUG_COLORS)]
    return l


class CGP_OT_layer_add(bpy.types.Operator):
    bl_idname = "cgp.layer_add"
    bl_label = "レイヤーを追加"
    bl_options = {'REGISTER', 'UNDO'}
    kind: EnumProperty(items=[('MATERIAL', "マテリアル", ""), ('ADJUST', "調整", "")], default='MATERIAL')
    adj_type: EnumProperty(items=ADJ_TYPES, default='HSV')

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        s = obj.cgp
        _SUSPEND[0] += 1
        try:
            _new_layer(s, self.kind, self.adj_type)
        finally:
            _SUSPEND[0] -= 1
        rebuild_material(obj)
        s.active_index = len(s.layers) - 1
        return {'FINISHED'}


class CGP_OT_layer_remove(bpy.types.Operator):
    bl_idname = "cgp.layer_remove"
    bl_label = "レイヤーを削除"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        s = obj.cgp
        i = s.active_index
        if i <= 0 or i >= len(s.layers):
            self.report({'WARNING'}, "ベースレイヤーは削除できません")
            return {'CANCELLED'}
        img = s.layers[i].image
        s.layers.remove(i)
        rebuild_material(obj)
        if img is not None and img.users == 0:
            bpy.data.images.remove(img)
        s.active_index = min(i, len(s.layers) - 1)
        return {'FINISHED'}


class CGP_OT_layer_move(bpy.types.Operator):
    bl_idname = "cgp.layer_move"
    bl_label = "レイヤーを移動"
    bl_options = {'REGISTER', 'UNDO'}
    direction: EnumProperty(items=[('UP', "上へ", ""), ('DOWN', "下へ", "")])

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        s = obj.cgp
        i = s.active_index
        j = i + 1 if self.direction == 'UP' else i - 1
        if i <= 0 or j <= 0 or j >= len(s.layers):
            return {'CANCELLED'}
        s.layers.move(i, j)
        s.active_index = j
        rebuild_material(obj)
        return {'FINISHED'}


class CGP_OT_paint_layer(bpy.types.Operator):
    bl_idname = "cgp.paint_layer"
    bl_label = "このレイヤーを塗る"
    bl_description = "テクスチャペイントモードに入り、選択レイヤーを塗れる状態にする"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        s = obj.cgp
        i = s.active_index
        if i <= 0:
            self.report({'WARNING'}, "ベースは塗れません。上のレイヤーを選んでください")
            return {'CANCELLED'}
        img = ensure_layer_image(obj, s.layers[i])
        uv = obj.data.uv_layers.get(UV_NAME)
        if uv is not None:
            obj.data.uv_layers.active = uv
        if obj.mode != 'TEXTURE_PAINT':
            bpy.ops.object.mode_set(mode='TEXTURE_PAINT')
        _lower_subdiv(obj)
        if s.light_paint and not _is_light(obj):
            rebuild_material(obj, light=True)
        _watch_obj(obj)
        _set_canvas(context, img, s)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for sp in area.spaces:
                    if sp.type == 'VIEW_3D' and sp.shading.type in ('SOLID', 'WIREFRAME'):
                        sp.shading.type = 'MATERIAL'
        return {'FINISHED'}


class CGP_OT_paint_exit(bpy.types.Operator):
    bl_idname = "cgp.paint_exit"
    bl_label = "塗り終了"
    bl_description = "オブジェクトモードに戻り、サブディビジョンの表示レベルを元に戻す"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        _end_paint(obj)
        return {'FINISHED'}


class CGP_OT_fill(bpy.types.Operator):
    bl_idname = "cgp.fill"
    bl_label = "塗りつぶし"
    bl_description = "選択レイヤーを全面塗りつぶし / クリア"
    bl_options = {'REGISTER', 'UNDO'}
    value: FloatProperty(default=1.0, min=0.0, max=1.0)

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        s = obj.cgp
        i = s.active_index
        if i <= 0:
            return {'CANCELLED'}
        layer = s.layers[i]
        img = ensure_layer_image(obj, layer)
        w, h = img.size
        v = self.value
        px = np.empty((w * h, 4), dtype=np.float32)
        px[:, 0] = v
        px[:, 1] = v * s.brush_softness
        px[:, 2] = v * s.brush_noise
        px[:, 3] = 1.0
        img.pixels.foreach_set(px.ravel())
        img.update()
        img.pack()
        layer.fmt = IMG_FMT
        return {'FINISHED'}


class CGP_OT_rebuild(bpy.types.Operator):
    bl_idname = "cgp.rebuild"
    bl_label = "マテリアルを再構築"
    bl_description = "レイヤーに使っているマテリアルを編集したときに押すと反映されます"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        rebuild_material(obj)
        return {'FINISHED'}


class CGP_OT_import_textures(bpy.types.Operator):
    bl_idname = "cgp.import_textures"
    bl_label = "テクスチャからレイヤー作成"
    bl_description = ("テクスチャ画像（カラー・ノーマル・ラフネス・ハイトなど）を選ぶと、ファイル名から種類を判別して"
                      "マテリアルを作り、新しいレイヤーとして追加します。ビューポートへのドラッグ＆ドロップでも使えます")
    bl_options = {'REGISTER', 'UNDO'}
    directory: StringProperty(subtype='DIR_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={'SKIP_SAVE', 'HIDDEN'})
    filter_image: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})
    filter_folder: BoolProperty(default=True, options={'HIDDEN', 'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return _terrain(context) is not None

    def invoke(self, context, event):
        if self.directory and len(self.files) > 0:
            return self.execute(context)
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        obj = _terrain(context)
        if obj is None:
            return {'CANCELLED'}
        paths = [os.path.join(self.directory, f.name) for f in self.files if f.name]
        groups = group_texture_files(paths)
        if not groups:
            self.report({'WARNING'}, "使える画像が見つかりませんでした")
            return {'CANCELLED'}
        s = obj.cgp
        names = []
        _SUSPEND[0] += 1
        try:
            for name, g in groups.items():
                mat = build_material_from_textures(name, g)
                l = _new_layer(s, 'MATERIAL')
                l.name = mat.name
                l.material = mat
                names.append(mat.name)
        finally:
            _SUSPEND[0] -= 1
        rebuild_material(obj)
        s.active_index = len(s.layers) - 1
        self.report({'INFO'}, "レイヤーを作成: " + ", ".join(names))
        return {'FINISHED'}


class CGP_FH_textures(bpy.types.FileHandler):
    bl_idname = "CGP_FH_textures"
    bl_label = "地面ペイントのレイヤーにする"
    bl_import_operator = "cgp.import_textures"
    bl_file_extensions = ";".join(TEX_EXTS)

    @classmethod
    def poll_drop(cls, context):
        area = getattr(context, "area", None)
        return area is not None and area.type == 'VIEW_3D' and _terrain(context) is not None


# ------------------------------------------------------------------ UI
class CGP_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        if index == 0:
            ic = 'WORLD'
        elif item.kind == 'ADJUST':
            ic = ADJ_ICONS.get(item.adj_type, 'MODIFIER')
        else:
            ic = 'MATERIAL' if item.material else 'COLOR'
        row.prop(item, "name", text="", emboss=False, icon=ic)
        sub = row.row()
        sub.alignment = 'RIGHT'
        if index > 0 and item.kind == 'ADJUST':
            sub.label(text="調整")
        else:
            sub.label(text=item.material.name if item.material else "単色")

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        n = len(items)
        return [self.bitflag_filter_item] * n, [n - 1 - i for i in range(n)]


class CGP_MT_add_layer(bpy.types.Menu):
    bl_idname = "CGP_MT_add_layer"
    bl_label = "レイヤーを追加"

    def draw(self, context):
        lay = self.layout
        lay.operator_context = 'INVOKE_DEFAULT'
        op = lay.operator("cgp.layer_add", text="マテリアル / 単色レイヤー", icon='MATERIAL')
        op.kind = 'MATERIAL'
        lay.operator("cgp.import_textures", text="テクスチャ画像から作成…", icon='IMAGE_DATA')
        lay.separator()
        for ident, label, _d in ADJ_TYPES:
            op = lay.operator("cgp.layer_add", text="調整: " + label, icon=ADJ_ICONS[ident])
            op.kind = 'ADJUST'
            op.adj_type = ident


class CGP_PT_main(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Concept"
    bl_label = "地面ペイント"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            layout.label(text="地面のメッシュを選択してください", icon='INFO')
            return
        s = obj.cgp
        if not s.enabled:
            layout.label(text=obj.name, icon='MESH_GRID')
            layout.operator("cgp.enable", text="この地面でペイントを始める", icon='BRUSH_DATA')
            return
        row = layout.row()
        row.template_list("CGP_UL_layers", "", s, "layers", s, "active_index", rows=6)
        col = row.column(align=True)
        col.menu("CGP_MT_add_layer", icon='ADD', text="")
        col.operator("cgp.layer_remove", icon='REMOVE', text="")
        col.separator()
        col.operator("cgp.layer_move", icon='TRIA_UP', text="").direction = 'UP'
        col.operator("cgp.layer_move", icon='TRIA_DOWN', text="").direction = 'DOWN'
        tip = layout.row()
        tip.scale_y = 0.7
        tip.label(text="テクスチャ画像はビューポートにドロップでもレイヤーにできます", icon='IMAGE_DATA')
        if 0 <= s.active_index < len(s.layers):
            i = s.active_index
            layer = s.layers[i]
            box = layout.box()
            if i > 0 and layer.kind == 'ADJUST':
                box.prop(layer, "adj_type", text="")
                col = box.column(align=True)
                t = layer.adj_type
                if t == 'HSV':
                    col.prop(layer, "adj_val")
                    col.prop(layer, "adj_sat")
                    col.prop(layer, "adj_hue")
                elif t == 'ROUGH':
                    col.prop(layer, "adj_rough")
                elif t == 'METAL':
                    col.prop(layer, "adj_metal")
                elif t == 'WET':
                    col.prop(layer, "wet_dark")
                    col.prop(layer, "wet_rough")
                    col.prop(layer, "wet_flat")
                elif t == 'TINT':
                    col.prop(layer, "tint_color")
                    col.prop(layer, "tint_blend")
            else:
                box.prop(layer, "material")
                if layer.material is None:
                    r = box.row(align=True)
                    r.prop(layer, "color", text="")
                    r.prop(layer, "roughness")
                else:
                    box.row().prop(layer, "coord_mode", expand=True)
                    if layer.coord_mode == 'WORLD':
                        box.prop(layer, "tile_size")
                    if s.light_paint:
                        box.prop(layer, "preview_color")
                col = box.column(align=True)
                col.prop(layer, "height_contrast")
                col.prop(layer, "height_offset")
            if i > 0:
                box.prop(layer, "opacity")
                painting = (obj.mode == 'TEXTURE_PAINT' and layer.image is not None and
                            context.scene.tool_settings.image_paint.canvas == layer.image)
                r = box.row()
                r.scale_y = 1.4
                if painting:
                    r.operator("cgp.paint_exit", text="塗り終了", icon='CHECKMARK')
                else:
                    r.operator("cgp.paint_layer", text="このレイヤーを塗る", icon='BRUSH_DATA')
                bb = box.box()
                bb.label(text="ブラシ（これから塗るストロークに反映）", icon='BRUSH_DATA')
                c = bb.column(align=True)
                c.prop(s, "brush_softness")
                c.prop(s, "brush_noise")
                r = box.row(align=True)
                r.operator("cgp.fill", text="全面塗る", icon='SNAP_FACE').value = 1.0
                r.operator("cgp.fill", text="クリア", icon='X').value = 0.0
                if painting:
                    hint = box.column(align=True)
                    hint.scale_y = 0.8
                    hint.label(text="F: ブラシサイズ / Shift+F: 強さ")
                    hint.label(text="X: 塗る ⇔ 消す を切り替え")
                    hint.label(text="ブラシの色はこのパネルが自動で設定します")
            else:
                box.label(text="ベース：一番下の地面（塗る必要なし）", icon='INFO')
        box = layout.box()
        box.label(text="ブレンド", icon='MOD_SMOOTH')
        box.prop(s, "height_strength")
        box.prop(s, "edge_noise_scale")
        box.prop(s, "show_debug")
        box.prop(s, "resolution")
        box = layout.box()
        box.label(text="軽くする", icon='MOD_SUBSURF')
        box.prop(s, "light_paint")
        box.prop(s, "paint_lower_subdiv")
        r = box.row()
        r.enabled = s.paint_lower_subdiv
        r.prop(s, "paint_subdiv_level")
        layout.operator("cgp.rebuild", icon='FILE_REFRESH')
        layout.operator("cgp.disable", text="解除（元のマテリアルに戻す）", icon='CANCEL')


# ------------------------------------------------------------------ handlers
@persistent
def _cgp_save_pre(*args):
    for img in bpy.data.images:
        if img.name.startswith("CGP_") and img.is_dirty:
            try:
                img.pack()
            except Exception as e:
                print("CGP: pack failed", img.name, e)


@persistent
def _cgp_load_post(*args):
    _watch.clear()
    for obj in bpy.data.objects:
        if obj.get("cgp_subdiv_backup") is not None or (obj.cgp.enabled and _is_light(obj)):
            _watch.add(obj.name)
    if _watch and not bpy.app.timers.is_registered(_poll_restore):
        bpy.app.timers.register(_poll_restore, first_interval=1.0)


_HANDLERS = ((bpy.app.handlers.save_pre, _cgp_save_pre), (bpy.app.handlers.load_post, _cgp_load_post))

classes = (CGP_Layer, CGP_Settings, CGP_UL_layers, CGP_MT_add_layer, CGP_OT_enable, CGP_OT_disable,
           CGP_OT_layer_add, CGP_OT_layer_remove, CGP_OT_layer_move, CGP_OT_paint_layer, CGP_OT_paint_exit,
           CGP_OT_fill, CGP_OT_rebuild, CGP_OT_import_textures, CGP_FH_textures, CGP_PT_main)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Object.cgp = PointerProperty(type=CGP_Settings)
    for lst, fn in _HANDLERS:
        for h in list(lst):
            if getattr(h, "__name__", "") == fn.__name__:
                lst.remove(h)
        lst.append(fn)
    _cgp_load_post()


def unregister():
    if bpy.app.timers.is_registered(_poll_restore):
        bpy.app.timers.unregister(_poll_restore)
    for lst, fn in _HANDLERS:
        for h in list(lst):
            if getattr(h, "__name__", "") == fn.__name__:
                lst.remove(h)
    del bpy.types.Object.cgp
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
