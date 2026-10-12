"""Checks that can run without Blender (stub `bpy`).  Run:  python3 tests/mock_check.py

 1. every panel's draw() runs; every property / operator / icon / operator argument it names exists
 2. paint_dab / Stroke / restore_patch (the brush maths and Ctrl+Z)
 3. v0.2 -> v0.3 brush image conversion
 4. the foliage brush modal operator driven by fake events on a flat ground
It does NOT replace testing in Blender (node trees, GPU drawing, real events are not exercised).
Needs: numpy, mathutils (pip install numpy mathutils)."""
import importlib.util
import math
import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# icons the add-on is allowed to use (names that exist in Blender 4.x)
KNOWN_ICONS = {
    'NONE', 'ADD', 'BLANK1', 'BRUSH_DATA', 'CAMERA_DATA', 'CHECKMARK', 'CON_SIZELIKE', 'ERROR', 'FILE_REFRESH',
    'HIDE_OFF', 'HIDE_ON', 'INFO', 'MESH_GRID', 'OUTLINER_COLLECTION', 'OUTLINER_OB_GROUP_INSTANCE',
    'OUTLINER_OB_MESH', 'OUTLINER_OB_POINTCLOUD', 'REMOVE', 'RESTRICT_VIEW_OFF', 'SHADING_BBOX', 'SNAP_FACE',
    'SOLO_ON', 'SOLO_OFF', 'TRIA_DOWN', 'TRIA_UP', 'X', 'WORLD', 'MODIFIER', 'MATERIAL',
}


# ------------------------------------------------------------------ stub bpy
class Prop:
    def __init__(self, kind, **kw):
        self.kind, self.kw = kind, kw


def _mk(kind):
    return lambda **kw: Prop(kind, **kw)


class _Base:
    pass


class _Handlers:
    depsgraph_update_post, load_post, save_pre = [], [], []


class _Timers:
    _reg = set()

    @classmethod
    def register(cls, fn, first_interval=0.0, persistent=False):
        cls._reg.add(fn)

    @classmethod
    def is_registered(cls, fn):
        return fn in cls._reg

    @classmethod
    def unregister(cls, fn):
        cls._reg.discard(fn)


def install_stubs():
    bpy = types.ModuleType("bpy")
    t = types.ModuleType("bpy.types")
    for n in ("Operator", "Panel", "PropertyGroup", "UIList", "Object", "Collection", "Image", "Material",
              "NodeSocket", "UILayout", "Scene"):
        setattr(t, n, type(n, (_Base,), {}))
    t.SpaceView3D = types.SimpleNamespace(
        draw_handler_add=lambda *a, **k: "HANDLE", draw_handler_remove=lambda *a, **k: None)
    bpy.types = t
    props = types.ModuleType("bpy.props")
    for n in ("StringProperty", "FloatProperty", "IntProperty", "BoolProperty", "EnumProperty",
              "PointerProperty", "CollectionProperty"):
        setattr(props, n, _mk(n))
    bpy.props = props
    handlers = types.ModuleType("bpy.app.handlers")
    handlers.depsgraph_update_post, handlers.load_post, handlers.save_pre = [], [], []
    handlers.persistent = lambda f: f
    bpy.app = types.SimpleNamespace(handlers=handlers, timers=_Timers)
    bpy.utils = types.SimpleNamespace(register_class=lambda c: None, unregister_class=lambda c: None)
    bpy.data = types.SimpleNamespace(objects=_Registry(), images=_Registry())
    bpy.context = None
    gpu = types.ModuleType("gpu")
    gx = types.ModuleType("gpu_extras")
    gxb = types.ModuleType("gpu_extras.batch")
    gxb.batch_for_shader = lambda *a, **k: None
    be = types.ModuleType("bpy_extras")
    v3 = types.ModuleType("bpy_extras.view3d_utils")
    be.view3d_utils = v3
    mods = {"bpy": bpy, "bpy.types": t, "bpy.props": props, "bpy.app": bpy.app, "bpy.app.handlers": handlers,
            "gpu": gpu, "gpu_extras": gx, "gpu_extras.batch": gxb, "bpy_extras": be,
            "bpy_extras.view3d_utils": v3}
    sys.modules.update(mods)
    return bpy, v3


class _Registry(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def load_addon():
    bpy, v3 = install_stubs()
    spec = importlib.util.spec_from_file_location("concept_scatter", os.path.join(ROOT, "concept_scatter.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["concept_scatter"] = m
    spec.loader.exec_module(m)
    return m, bpy, v3


# ------------------------------------------------------------------ fake data
class PG:
    """Property group instance built from the class annotations."""

    def __init__(self, cls, owner=None):
        object.__setattr__(self, "_cls", cls)
        for c in reversed(cls.__mro__):
            for name, p in getattr(c, "__annotations__", {}).items():
                if not isinstance(p, Prop):
                    continue
                if p.kind == "CollectionProperty":
                    v = Coll(p.kw["type"])
                elif p.kind == "PointerProperty":
                    v = None
                elif p.kind == "EnumProperty":
                    v = p.kw.get("default", "NONE")
                else:
                    v = p.kw.get("default", {"StringProperty": "", "BoolProperty": False}.get(p.kind, 0))
                object.__setattr__(self, name, v)
        object.__setattr__(self, "id_data", owner)

    def as_pointer(self):
        return id(self)


class Coll(list):
    def __init__(self, typ):
        super().__init__()
        self.typ = typ

    def add(self):
        it = PG(self.typ, getattr(self, "owner", None))
        self.append(it)
        return it

    def clear(self):
        del self[:]


class Obj:
    type = 'MESH'
    mode = 'OBJECT'

    def __init__(self, name, csc_cls):
        self.name = name
        self.csc = PG(csc_cls, self)
        self.csc.layers.owner = self
        self._props = {}

    def get(self, k, d=None):
        return self._props.get(k, d)

    def as_pointer(self):
        return id(self)


class OpArgs:
    def __init__(self, cls):
        object.__setattr__(self, "cls", cls)

    def __setattr__(self, k, v):
        ann = {}
        for c in reversed(self.cls.__mro__):
            ann.update(getattr(c, "__annotations__", {}))
        assert k in ann, "operator %s has no argument %s" % (self.cls.bl_idname, k)


class Layout:
    def __init__(self, env):
        object.__setattr__(self, "env", env)

    def __setattr__(self, k, v):
        pass

    def _child(self, *a, **k):
        return Layout(self.env)

    def __getattr__(self, name):
        if name in ("row", "column", "box", "split", "column_flow", "grid_flow", "menu_pie"):
            return self._child
        raise AttributeError(name)

    def prop(self, data, name, text=None, icon='NONE', **kw):
        env = self.env
        if isinstance(data, PG):
            names = set()
            for c in data._cls.__mro__:
                names |= {k for k, v in getattr(c, "__annotations__", {}).items() if isinstance(v, Prop)}
            assert name in names, "%s has no property %s" % (data._cls.__name__, name)
        else:
            assert hasattr(data, name), "%r has no property %s" % (data, name)
        assert icon in KNOWN_ICONS, "unknown icon " + icon
        env["props"].add(name)

    def operator(self, idname, text="", icon='NONE', **kw):
        cls = self.env["ops"].get(idname)
        assert cls is not None, "unknown operator " + idname
        assert icon in KNOWN_ICONS, "unknown icon " + icon
        self.env["opsused"].add(idname)
        return OpArgs(cls)

    def label(self, text="", icon='NONE', **kw):
        assert icon in KNOWN_ICONS, "unknown icon " + icon

    def separator(self, **kw):
        pass

    def template_list(self, listtype, listid, data, propname, active_data, active_propname, rows=5, **kw):
        assert listtype in self.env["uilists"], "unknown UIList " + listtype


class Ctx:
    def __init__(self, obj, mode='OBJECT'):
        self.active_object = obj
        self.mode = mode
        self.scene = types.SimpleNamespace(camera=None)
        self.area = None


# ------------------------------------------------------------------ 1. panels
def check_panels(m):
    classes = {c.__name__: c for c in m.classes}
    ops = {c.bl_idname: c for c in m.classes if hasattr(c, "bl_idname") and "_OT_" in c.__name__}
    uilists = {c.__name__ for c in m.classes if "_UL_" in c.__name__}
    panels = [c for c in m.classes if "_PT_" in c.__name__]
    assert any(c.__name__ == "CSC_PT_brush" for c in panels), "brush panel is not registered"
    env = {"props": set(), "ops": ops, "opsused": set(), "uilists": uilists}
    m.ground_area = lambda g: 100.0
    g = Obj("G", m.CSC_Ground)
    ctx = Ctx(g)
    m.bpy.context = types.SimpleNamespace(scene=ctx.scene)
    n = 0
    # no layers
    for st in range(6):
        g.csc.layers.clear()
        if st >= 1:
            l = g.csc.layers.add()
            l.uid, l.name = "u1", "Layer"
            g.csc.active_index = 0
        if st >= 2:
            l.collection = types.SimpleNamespace(name="Assets")
            a = l.assets.add()
            a.obj = types.SimpleNamespace(name="Rock")
            a2 = l.assets.add()
            a2.coll = types.SimpleNamespace(name="Group")
        if st >= 3:
            for flag in ("use_brush", "use_slope", "use_height", "use_noise", "use_avoid", "avoid_layers",
                         "camera_cull", "bbox_far"):
                setattr(l, flag, True)
        if st == 4:
            m._FP.update(ground="G")
        if st == 5:
            m._FP.clear()
            ctx.mode = 'EDIT'
        for P in [c for c in panels]:
            lay = Layout(env)
            inst = P.__new__(P)
            inst.layout = lay
            if getattr(P, "poll", None) and not P.poll(ctx) and P.__name__ != "CSC_PT_main":
                continue
            P.draw(inst, ctx)
            n += 1
    m._FP.clear()
    print("panels: %d draws OK; %d operators used by the UI" % (n, len(env["opsused"])))
    for want in ("csc.foliage_paint", "csc.foliage_exit", "csc.brush_fill"):
        assert want in env["opsused"], want + " is not reachable from the UI"
    for gone in ("csc.brush_paint", "csc.brush_exit"):
        assert gone not in ops, gone + " should have been removed"
    for pr in ("brush_radius", "brush_strength", "brush_softness", "brush_noise", "brush_noise_scale",
               "show_overlay", "use_brush", "brush_res"):
        assert pr in env["props"], pr + " is not shown in any panel"


# ------------------------------------------------------------------ 2. brush maths
def check_dab(m):
    buf = np.zeros((64, 64, 4), np.float32)
    box = m.dab_box(64, 64, 10.0, 10.0, 5.0, 5.0)
    assert box == (5, 5, 16, 16), box
    assert m.dab_box(64, 64, -50.0, 10.0, 5.0, 5.0) is None
    assert m.dab_box(64, 64, 10.0, 10.0, 0.0, 5.0) is None
    m.paint_dab(buf, 32.0, 32.0, 10.0, 6.0, 0.5, 1.0, False)
    assert buf[32, 40, 0] < 1.0
    assert buf[32, 32, 0] == 1.0, "centre should be fully grown"
    assert buf[32, 44, 0] == 0.0 and buf[40, 32, 0] == 0.0, "outside the radius must stay empty"
    assert 0.0 < buf[32, 38, 0] < 1.0 or buf[32, 38, 0] == 1.0
    assert 0.0 < buf[36, 32, 0] < 1.0, "ellipse: y radius is 6"
    assert (buf[..., 1] == 0).all() and (buf[..., 2] == 0).all()
    # strength caps the value, max() keeps the stronger one
    b2 = np.zeros((16, 16, 4), np.float32)
    m.paint_dab(b2, 8.0, 8.0, 4.0, 4.0, 0.0, 0.4, False)
    assert abs(b2[8, 8, 0] - 0.4) < 1e-6
    m.paint_dab(b2, 8.0, 8.0, 4.0, 4.0, 0.0, 0.2, False)
    assert abs(b2[8, 8, 0] - 0.4) < 1e-6, "a weaker dab must not lower the value"
    # erase raises G, grow wipes it again
    m.paint_dab(b2, 8.0, 8.0, 4.0, 4.0, 0.0, 1.0, True)
    assert b2[8, 8, 1] == 1.0
    m.paint_dab(b2, 8.0, 8.0, 4.0, 4.0, 0.0, 1.0, False)
    assert b2[8, 8, 1] == 0.0 and b2[8, 8, 0] == 1.0
    # a dab partly outside the image must not raise
    m.paint_dab(b2, -1.0, 15.0, 6.0, 6.0, 0.5, 1.0, False)
    m.paint_dab(b2, 20.0, 15.0, 6.0, 6.0, 0.5, 1.0, False)

    # Stroke: several dabs growing the touched box, then restore -> exactly the old image
    rng = np.random.default_rng(1)
    base = rng.random((48, 48, 4)).astype(np.float32)
    buf = base.copy()
    st = m.Stroke(buf)
    for cx, cy, erase in ((20, 20, False), (30, 22, False), (34, 36, True), (8, 40, False), (24, 6, True)):
        bx = m.dab_box(48, 48, cx, cy, 7.0, 5.0)
        st.before(bx)
        m.paint_dab(buf, cx, cy, 7.0, 5.0, 0.5, 0.8, erase)
    assert not np.array_equal(buf, base)
    entry = st.result()
    m.restore_patch(buf, entry)
    assert np.array_equal(buf, base), "undo did not restore the image"
    assert m.Stroke(buf).result() is None
    print("dab / stroke / undo: OK")


# ------------------------------------------------------------------ 3. image conversion
class Pixels:
    def __init__(self, img):
        self.img = img

    def foreach_get(self, out):
        out[:] = self.img.data.ravel()

    def foreach_set(self, arr):
        self.img.data = np.asarray(arr, np.float32).reshape(self.img.data.shape).copy()


class Img(dict):
    def __init__(self, name, w, h):
        super().__init__()
        self.name, self.size = name, (w, h)
        self.data = np.zeros((h, w, 4), np.float32)
        self.pixels = Pixels(self)
        self.tags = 0
        self.packed = 0

    def update_tag(self):
        self.tags += 1

    def pack(self):
        self.packed += 1


def check_convert(m):
    for cgp_uid, mode, want_r, want_g in (("L", 'MUL', 0.0, 0.75), ("L", 'ADD', 0.25, 0.0), ("", 'MUL', 0.25, 0.0)):
        img = Img("CSC_x", 4, 4)
        img.data[..., 0] = 0.25
        img.data[..., 1] = 0.9
        img.data[..., 2] = 0.9
        img.data[..., 3] = 1.0
        layer = types.SimpleNamespace(cgp_uid=cgp_uid, brush_mode=mode)
        m.convert_v02_image(img, layer)
        assert np.allclose(img.data[..., 0], want_r) and np.allclose(img.data[..., 1], want_g), (cgp_uid, mode)
        assert (img.data[..., 2] == 0).all() and img["csc_fmt"] == m.BRUSH_FMT and img.tags == 1 and img.packed == 1
    img = Img("CSC_y", 4, 4)
    m.fill_image(img, 1.0)
    assert (img.data[..., 0] == 1).all() and (img.data[..., 1] == 0).all() and img.data[0, 0, 3] == 1
    print("v0.2 image conversion / fill: OK")


# ------------------------------------------------------------------ 4. modal operator
class Region:
    def __init__(self, typ, x, y, w, h):
        self.type, self.x, self.y, self.width, self.height = typ, x, y, w, h


class Event:
    def __init__(self, type, value='NOTHING', x=400, y=300, shift=False, ctrl=False, alt=False):
        self.type, self.value, self.mouse_x, self.mouse_y = type, value, x, y
        self.shift, self.ctrl, self.alt = shift, ctrl, alt


def check_modal(m, bpy, v3):
    import mathutils
    gobj = Obj("G", m.CSC_Ground)
    gobj.type = 'MESH'
    gobj.matrix_world = mathutils.Matrix.Diagonal((2.0, 2.0, 2.0, 1.0))      # 2x scale, 20 m wide plane
    gobj.data = types.SimpleNamespace()

    # flat ground at z = 0, local size 10 x 10, centred on the origin
    def ray_cast(origin, direction, distance=1e30, depsgraph=None):
        if abs(direction.z) < 1e-9:
            return (False, None, None, -1)
        t = -origin.z / direction.z
        if t < 0 or t > distance:
            return (False, None, None, -1)
        p = origin + direction * t
        if abs(p.x) > 5 or abs(p.y) > 5:
            return (False, None, None, -1)
        return (True, p, mathutils.Vector((0, 0, 1)), 0)

    gobj.ray_cast = ray_cast
    gobj.mode = 'OBJECT'
    layer = gobj.csc.layers.add()
    layer.uid, layer.name = "u1", "Grass"
    gobj.csc.active_index = 0
    img = Img("CSC_G_u1", 128, 128)
    img.data[..., 3] = 1.0
    img["csc_fmt"] = m.BRUSH_FMT
    images = {"CSC_G_u1": img}
    bpy.data.objects["G"] = gobj
    bpy.data.images = types.SimpleNamespace(get=lambda n, d=None: images.get(n, d))
    bpy.data.objects = types.SimpleNamespace(get=lambda n, d=None: {"G": gobj}.get(n, d))

    # the add-on side that needs real Blender data is replaced
    pushed = []
    m.ensure_brush_uv = lambda g: "CSC_UV"
    m.ground_uv_bounds = lambda g: (np.array([-5.0, -5.0], np.float32), np.array([10.0, 10.0], np.float32))
    m.enable_brush = lambda g, l, scene=None, uvname=None: img
    m._push_ground = lambda g: pushed.append(1)
    m._tag_redraw_3d = lambda: None
    m._fp_reset_orig = m._fp_reset

    win = Region('WINDOW', 0, 0, 800, 600)
    ui = Region('UI', 800, 0, 200, 600)
    hdr = Region('HEADER', 0, 600, 1000, 30)
    area = types.SimpleNamespace(type='VIEW_3D', regions=[hdr, ui, win], tagged=0,
                                 spaces=types.SimpleNamespace(active=types.SimpleNamespace(region_3d=object())),
                                 tag_redraw=lambda: None)
    status = []
    wm = types.SimpleNamespace(event_timer_add=lambda *a, **k: "T", event_timer_remove=lambda t: None,
                               modal_handler_add=lambda op: None)
    ctx = types.SimpleNamespace(
        active_object=gobj, mode='OBJECT', area=area, scene=types.SimpleNamespace(camera=None), window=object(),
        window_manager=wm, evaluated_depsgraph_get=lambda: None,
        workspace=types.SimpleNamespace(status_text_set=lambda t: status.append(t)))

    # camera looks straight down from z = 40 on a 800x600 window: 1 px = 0.05 world units
    def to_origin(region, rv3d, co):
        return mathutils.Vector(((co[0] - 400) * 0.05, (co[1] - 300) * 0.05, 40.0))

    v3.region_2d_to_origin_3d = to_origin
    v3.region_2d_to_vector_3d = lambda region, rv3d, co: mathutils.Vector((0, 0, -1))

    # world (x, y) -> mouse px; ground is scaled 2x so local = world / 2
    s = gobj.csc
    s.brush_radius, s.brush_strength, s.brush_softness = 2.0, 1.0, 0.5
    op = m.CSC_OT_foliage_paint()
    reported = []
    op.report = lambda lvl, msg: reported.append(msg)
    assert op.invoke(ctx, Event('NOTHING')) == {'RUNNING_MODAL'}, reported
    assert m.foliage_running() and status and status[0].startswith("ブラシで生やす")

    def send(ev):
        return op.modal(ctx, ev)

    def px(wx, wy):                      # world -> pixel in the window
        return int(wx / 0.05 + 400), int(wy / 0.05 + 300)

    def pix(lx, ly):                     # ground-local (x, y) -> image pixel index
        return int((ly + 5) / 10 * 128), int((lx + 5) / 10 * 128)

    # move: cursor appears, event is passed on
    x, y = px(0, 0)
    assert send(Event('MOUSEMOVE', x=x, y=y)) == {'PASS_THROUGH'}
    assert _FP_visible(m) and len(m._FP["outer"]) == 49 and len(m._FP["inner"]) == 33
    # outer ring radius on the ground = 2 m world
    r = max(math.hypot(px_[0], px_[1]) for px_ in m._FP["outer"])
    assert abs(r - 2.0) < 0.05, r

    # stroke: press, drag to the right, release
    assert send(Event('LEFTMOUSE', 'PRESS', x=x, y=y)) == {'RUNNING_MODAL'}
    x2, _ = px(4, 0)
    assert send(Event('MOUSEMOVE', x=x2, y=y)) == {'RUNNING_MODAL'}
    assert send(Event('LEFTMOUSE', 'RELEASE', x=x2, y=y)) == {'RUNNING_MODAL'}
    assert send(Event('TIMER')) == {'PASS_THROUGH'}
    row, col = pix(0.0, 0.0)
    assert img.data[row, col, 0] == 1.0, "start of the stroke should be grown"
    row2, col2 = pix(2.0, 0.0)                       # 4 m world = 2 m local
    assert img.data[row2, col2, 0] == 1.0, "end of the stroke should be grown"
    rm, cm = pix(1.0, 0.0)
    assert img.data[rm, cm, 0] == 1.0, "dabs must be filled in between"
    assert img.data[pix(1.0, 3.0)][0] == 0.0, "far from the stroke stays empty"
    grown = img.data[..., 0].copy()
    assert img.tags > 0

    # erase with shift
    xe, ye = px(2, 0)
    send(Event('LEFTMOUSE', 'PRESS', x=xe, y=ye, shift=True))
    send(Event('LEFTMOUSE', 'RELEASE', x=xe, y=ye, shift=True))
    assert img.data[pix(1.0, 0.0)][1] == 1.0, "erase should raise G"
    # undo the erase, then the grow
    assert send(Event('Z', 'PRESS', ctrl=True, x=x, y=y)) == {'RUNNING_MODAL'}
    assert img.data[..., 1].max() == 0.0 and np.array_equal(img.data[..., 0], grown), "undo 1"
    assert send(Event('Z', 'PRESS', ctrl=True, x=x, y=y)) == {'RUNNING_MODAL'}
    assert img.data[..., 0].max() == 0.0, "undo 2 should empty the image"

    # keys
    r0 = s.brush_radius
    assert send(Event('RIGHT_BRACKET', 'PRESS', x=x, y=y)) == {'RUNNING_MODAL'} and s.brush_radius > r0
    assert send(Event('LEFT_BRACKET', 'PRESS', x=x, y=y)) == {'RUNNING_MODAL'} and abs(s.brush_radius - r0) < 1e-6
    st0 = s.brush_strength
    send(Event('LEFT_BRACKET', 'PRESS', x=x, y=y, shift=True))
    assert s.brush_strength < st0
    r0 = s.brush_radius
    send(Event('WHEELUPMOUSE', 'PRESS', x=x, y=y, ctrl=True))
    assert s.brush_radius > r0

    # pass-through: navigation, sidebar clicks, alt+click, gizmo corner
    assert send(Event('MIDDLEMOUSE', 'PRESS', x=x, y=y)) == {'PASS_THROUGH'}
    assert send(Event('NUMPAD_5', 'PRESS', x=x, y=y)) == {'PASS_THROUGH'}
    assert send(Event('WHEELUPMOUSE', 'PRESS', x=x, y=y)) == {'PASS_THROUGH'}
    assert send(Event('LEFTMOUSE', 'PRESS', x=900, y=300)) == {'PASS_THROUGH'}, "sidebar click"
    assert send(Event('LEFTMOUSE', 'PRESS', x=x, y=y, alt=True)) == {'PASS_THROUGH'}
    assert send(Event('LEFTMOUSE', 'PRESS', x=780, y=590)) == {'PASS_THROUGH'}, "gizmo corner"
    assert send(Event('RIGHT_BRACKET', 'PRESS', x=900, y=300)) == {'PASS_THROUGH'}, "keys over the sidebar"
    before = img.data.copy()
    assert send(Event('LEFTMOUSE', 'PRESS', x=900, y=300)) == {'PASS_THROUGH'}
    assert np.array_equal(before, img.data)

    # miss: pointing outside the ground paints nothing and does not crash
    xo, yo = px(30, 0)
    send(Event('MOUSEMOVE', x=xo, y=yo))
    assert not _FP_visible(m)
    send(Event('LEFTMOUSE', 'PRESS', x=xo, y=yo))
    send(Event('MOUSEMOVE', x=xo + 3, y=yo))
    send(Event('LEFTMOUSE', 'RELEASE', x=xo + 3, y=yo))
    assert img.data[..., 0].max() == 0.0

    # fill from the panel while running -> buffers are re-read
    s.brush_strength = 1.0
    img.data[..., 0] = 0.5
    m._FP["reload"] = True
    send(Event('TIMER'))
    send(Event('LEFTMOUSE', 'PRESS', x=x, y=y))
    send(Event('LEFTMOUSE', 'RELEASE', x=x, y=y))
    assert img.data[pix(0, 0)][0] == 1.0 and img.data[pix(4.0, 4.0)][0] == 0.5

    # finish with Enter, and by the exit operator
    assert send(Event('RET', 'PRESS', x=x, y=y)) == {'FINISHED'}
    assert not m.foliage_running() and status[-1] is None and img.packed >= 1
    assert op.invoke(ctx, Event('NOTHING')) == {'RUNNING_MODAL'}
    m.CSC_OT_foliage_exit().execute(ctx)
    assert send(Event('TIMER')) == {'FINISHED'} and not m.foliage_running()
    assert op.invoke(ctx, Event('NOTHING')) == {'RUNNING_MODAL'}
    assert send(Event('ESC', 'PRESS', x=900, y=300)) == {'FINISHED'}, "Esc exits even over the sidebar"
    print("foliage brush modal flow: OK")


def _FP_visible(m):
    return bool(m._FP.get("visible"))


def main():
    m, bpy, v3 = load_addon()
    check_panels(m)
    check_dab(m)
    check_convert(m)
    check_modal(m, bpy, v3)
    print("ALL OK")


if __name__ == "__main__":
    main()
