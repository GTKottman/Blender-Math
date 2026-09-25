"""Blender Math Bridge - the Blender side of the Blender Math MCP server.

Runs a small JSON-over-TCP server inside Blender (localhost only).  The MCP
server does all the math (typesetting, sampling, verification) and sends
ready-made geometry here; this add-on only builds objects, materials, cameras
and renders.  It has no dependencies beyond Blender itself.

Protocol: one JSON object per line.
    request : {"type": "<command>", "params": {...}}
    response: {"status": "ok", "result": ...} | {"status": "error", "message": "..."}
"""

bl_info = {
    "name": "Blender Math Bridge (MCP)",
    "author": "Blender-Math contributors",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Math MCP",
    "description": "Lets an AI (via the blender-math MCP server) typeset LaTeX and draw exact math in Blender",
    "category": "Interface",
}

import base64
import contextlib
import io
import json
import math
import os
import queue
import socket
import tempfile
import threading
import traceback

import bpy
import mathutils

ADDON_VERSION = ".".join(map(str, bl_info["version"]))
DEFAULT_PORT = 9877
COLLECTION_NAME = "Math"
META_KEY = "math_meta"


# =============================================================================== helpers


def _math_collection(name=COLLECTION_NAME):
    scene = bpy.context.scene
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        with contextlib.suppress(RuntimeError):
            scene.collection.children.link(coll)
    return coll


def _link(obj, params):
    coll = _math_collection(params.get("collection") or COLLECTION_NAME)
    coll.objects.link(obj)
    loc = params.get("location")
    if loc is not None:
        obj.location = loc
    rot = params.get("rotation")  # radians, XYZ euler
    if rot is not None:
        obj.rotation_euler = rot
    sc = params.get("scale")
    if sc is not None:
        obj.scale = sc if isinstance(sc, (list, tuple)) else (sc, sc, sc)
    parent = params.get("parent")
    if parent:
        p = _get_obj(parent)
        obj.parent = p
        obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    meta = params.get("metadata")
    if meta:
        obj[META_KEY] = json.dumps(meta)
    if params.get("face_camera"):
        _face_camera(obj)
    return obj


def _get_obj(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise ValueError(f"No object named {name!r}")
    return obj


def _face_camera(obj):
    cam = bpy.context.scene.camera
    if cam is None:
        raise ValueError("face_camera needs a scene camera; call setup_scene first")
    con = obj.constraints.new("COPY_ROTATION")
    con.name = "Face camera"
    con.target = cam


def _set_blend(mat, alpha):
    if alpha >= 1.0:
        return
    if hasattr(mat, "surface_render_method"):  # EEVEE Next (4.2+)
        mat.surface_render_method = "BLENDED"
    if hasattr(mat, "blend_method"):
        with contextlib.suppress(TypeError, AttributeError):
            mat.blend_method = "BLEND"


def _material(name, spec):
    """spec: {color: linear rgba, style: flat|shaded|glossy|metal, vertex_colors: bool,
    emission_strength, roughness}"""
    spec = spec or {}
    color = list(spec.get("color", [1, 1, 1, 1]))
    if len(color) == 3:
        color.append(1.0)
    style = spec.get("style", "flat")
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color  # Solid-view color
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    color_socket_src = None
    if spec.get("vertex_colors"):
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = "Col"
        attr.location = (-400, 0)
        color_socket_src = attr.outputs["Color"]

    if style == "flat":
        shader = nt.nodes.new("ShaderNodeEmission")
        shader.inputs["Strength"].default_value = float(spec.get("emission_strength", 1.0))
        cin = shader.inputs["Color"]
    else:
        shader = nt.nodes.new("ShaderNodeBsdfPrincipled")
        cin = shader.inputs["Base Color"]
        shader.inputs["Roughness"].default_value = float(
            spec.get("roughness", {"glossy": 0.2, "metal": 0.25}.get(style, 0.5)))
        if style == "metal":
            shader.inputs["Metallic"].default_value = 1.0
    cin.default_value = color
    if color_socket_src is not None:
        nt.links.new(color_socket_src, cin)
    last = shader.outputs[0]
    alpha = float(color[3])
    if alpha < 1.0:
        transp = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        mix.inputs["Fac"].default_value = alpha
        nt.links.new(transp.outputs[0], mix.inputs[1])
        nt.links.new(last, mix.inputs[2])
        last = mix.outputs[0]
        _set_blend(mat, alpha)
    nt.links.new(last, out.inputs["Surface"])
    return mat


def _assign_material(obj, spec):
    mat = _material(f"Math_{obj.name}", spec)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def _obj_summary(obj):
    info = {
        "name": obj.name,
        "type": obj.type,
        "location": [round(v, 6) for v in obj.matrix_world.translation],
        "dimensions": [round(v, 6) for v in obj.dimensions],
        "parent": obj.parent.name if obj.parent else None,
    }
    if META_KEY in obj:
        with contextlib.suppress(Exception):
            info["math"] = json.loads(obj[META_KEY])
    return info


# =============================================================================== commands


def cmd_ping(params):
    return {"blender_version": bpy.app.version_string, "addon_version": ADDON_VERSION}


def cmd_create_curve(params):
    """Bezier/poly splines -> curve object.

    splines: [{"co": [...], "hl": [...], "hr": [...], "cyclic": bool}  (bezier)
              | {"poly": [[x,y,z],...], "cyclic": bool}]
    fill: filled 2D shape (glyphs, disks).  bevel_depth: tube radius for lines.
    """
    name = params.get("name", "MathCurve")
    splines = params.get("splines") or []
    if not splines:
        raise ValueError("create_curve needs at least one spline")
    fill = bool(params.get("fill", False))
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "2D" if fill else "3D"
    cu.resolution_u = int(params.get("resolution", 12))
    for sp in splines:
        if "co" in sp:
            n = len(sp["co"])
            s = cu.splines.new("BEZIER")
            s.bezier_points.add(n - 1)
            for i, bp in enumerate(s.bezier_points):
                bp.handle_left_type = "FREE"
                bp.handle_right_type = "FREE"
                bp.co = sp["co"][i]
                bp.handle_left = sp["hl"][i]
                bp.handle_right = sp["hr"][i]
        else:
            pts = sp["poly"]
            s = cu.splines.new("POLY")
            s.points.add(len(pts) - 1)
            flat = []
            for p in pts:
                flat.extend((p[0], p[1], p[2] if len(p) > 2 else 0.0, 1.0))
            s.points.foreach_set("co", flat)
        s.use_cyclic_u = bool(sp.get("cyclic", False))
        s.resolution_u = cu.resolution_u
    if fill:
        cu.fill_mode = "BOTH"
        cu.extrude = float(params.get("extrude", 0.0))
    else:
        cu.fill_mode = "FULL"
        cu.bevel_depth = float(params.get("bevel_depth", 0.01))
        cu.bevel_resolution = int(params.get("bevel_resolution", 4))
        if hasattr(cu, "use_fill_caps"):
            cu.use_fill_caps = True
    obj = bpy.data.objects.new(name, cu)
    _link(obj, params)
    _assign_material(obj, params.get("material"))
    return _obj_summary(obj)


def cmd_create_mesh(params):
    import bmesh

    name = params.get("name", "MathMesh")
    verts = params.get("vertices") or []
    faces = params.get("faces") or []
    if not verts:
        raise ValueError("create_mesh needs vertices")
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    colors = params.get("colors")
    if colors:
        attr = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        flat = [c for rgba in colors for c in (list(rgba) + [1.0])[:4]]
        attr.data.foreach_set("color", flat)
    me.validate()
    md = params.get("merge_distance")
    if md:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(md))
        bm.to_mesh(me)
        bm.free()
    if params.get("smooth"):
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    obj = bpy.data.objects.new(name, me)
    _link(obj, params)
    _assign_material(obj, params.get("material"))
    return _obj_summary(obj)


def cmd_create_empty(params):
    obj = bpy.data.objects.new(params.get("name", "MathEmpty"), None)
    obj.empty_display_type = params.get("display", "PLAIN_AXES")
    obj.empty_display_size = float(params.get("size", 0.25))
    _link(obj, params)
    return _obj_summary(obj)


def _math_objects():
    coll = bpy.data.collections.get(COLLECTION_NAME)
    objs = set(coll.all_objects) if coll else set()
    objs |= {o for o in bpy.context.scene.objects if META_KEY in o}
    return sorted(objs, key=lambda o: o.name)


def cmd_get_scene_info(params):
    scene = bpy.context.scene
    cam = scene.camera
    return {
        "scene": scene.name,
        "blender_version": bpy.app.version_string,
        "render_engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "camera": None if cam is None else {
            "name": cam.name,
            "type": cam.data.type,
            "location": list(cam.location),
            "ortho_scale": cam.data.ortho_scale,
        },
        "math_objects": [_obj_summary(o) for o in _math_objects()],
        "other_objects": [o.name for o in scene.objects if o not in set(_math_objects())][:200],
    }


def cmd_get_object(params):
    obj = _get_obj(params["name"])
    info = _obj_summary(obj)
    info["matrix_world"] = [list(r) for r in obj.matrix_world]
    info["children"] = [c.name for c in obj.children]
    return info


def _delete(obj, recursive=True):
    if recursive:
        for c in list(obj.children):
            _delete(c, True)
    data = obj.data
    mats = list(getattr(data, "materials", []) or [])
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        if isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)
        elif isinstance(data, bpy.types.Mesh):
            bpy.data.meshes.remove(data)
    for m in mats:
        if m is not None and m.users == 0:
            bpy.data.materials.remove(m)


def cmd_delete(params):
    removed = []
    if params.get("all_math"):
        targets = _math_objects()
    else:
        targets = [_get_obj(n) for n in params.get("names", [])]
    names = {o.name for o in targets}
    for name in sorted(names):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            removed.append(name)
            _delete(obj, recursive=params.get("recursive", True))
    return {"removed": removed}


def cmd_transform(params):
    obj = _get_obj(params["name"])
    if params.get("location") is not None:
        obj.location = params["location"]
    if params.get("rotation") is not None:
        obj.rotation_euler = params["rotation"]
    if params.get("scale") is not None:
        s = params["scale"]
        obj.scale = s if isinstance(s, (list, tuple)) else (s, s, s)
    return _obj_summary(obj)


def cmd_set_material(params):
    obj = _get_obj(params["name"])
    targets = [obj] + (list(obj.children_recursive) if params.get("include_children") else [])
    for o in targets:
        if o.type in ("MESH", "CURVE", "FONT", "SURFACE"):
            _assign_material(o, params.get("material"))
    return {"updated": [o.name for o in targets]}


def _look_at(obj, target):
    direction = mathutils.Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _set_engine(scene, engine):
    names = {
        "eevee": ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"],
        "cycles": ["CYCLES"],
        "workbench": ["BLENDER_WORKBENCH"],
    }.get((engine or "eevee").lower(), [engine])
    for n in names:
        try:
            scene.render.engine = n
            return n
        except TypeError:
            continue
    raise ValueError(f"Render engine {engine!r} not available")


def cmd_setup_scene(params):
    """Camera, background and color management for crisp, color-exact math."""
    scene = bpy.context.scene
    mode = params.get("mode", "2d")
    # Colors: 'Standard' view transform shows material colors exactly as specified.
    vs = scene.view_settings
    with contextlib.suppress(TypeError):
        vs.view_transform = "Standard"
    with contextlib.suppress(TypeError):
        vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.render.resolution_x = int(params.get("resolution", [1920, 1080])[0])
    scene.render.resolution_y = int(params.get("resolution", [1920, 1080])[1])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(params.get("transparent", False))
    _set_engine(scene, params.get("engine", "eevee"))

    # World background
    world = scene.world or bpy.data.worlds.new("MathWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = params.get("background", [0.0, 0.0, 0.0, 1.0])
    bg.inputs["Strength"].default_value = 1.0
    wout = next((n for n in world.node_tree.nodes if n.type == "OUTPUT_WORLD"), None)
    if wout is None:
        wout = world.node_tree.nodes.new("ShaderNodeOutputWorld")
    world.node_tree.links.new(bg.outputs[0], wout.inputs["Surface"])

    # Camera
    cam = scene.camera
    if cam is None or cam.type != "CAMERA":
        cam = bpy.data.objects.new("MathCamera", bpy.data.cameras.new("MathCamera"))
        scene.collection.objects.link(cam)
        scene.camera = cam
    center = params.get("center", [0, 0, 0])
    if mode == "2d":
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = float(params.get("view_width", 16.0))
        cam.location = (center[0], center[1], center[2] + 50.0)
        cam.rotation_euler = (0, 0, 0)
        cam.data.clip_end = 1000.0
    else:
        cam.data.type = "PERSP"
        cam.data.lens = float(params.get("lens", 50.0))
        cam.location = params.get("camera_location", [center[0] + 12, center[1] - 12, center[2] + 9])
        _look_at(cam, center)
        cam.data.clip_end = 1000.0
        if params.get("add_light", True) and bpy.data.objects.get("MathSun") is None:
            sun = bpy.data.objects.new("MathSun", bpy.data.lights.new("MathSun", "SUN"))
            sun.data.energy = 3.0
            sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(30))
            scene.collection.objects.link(sun)
        bg.inputs["Strength"].default_value = float(params.get("ambient", 1.0))

    if params.get("viewport", True):
        _viewport_to_camera()
    return {"camera": cam.name, "mode": mode, "engine": scene.render.engine}


def _viewport_to_camera():
    """Make every open 3D viewport look through the camera with material preview."""
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.region_3d.view_perspective = "CAMERA"
                    space.shading.type = "MATERIAL"
                    with contextlib.suppress(AttributeError):
                        space.shading.use_scene_world = True
                        space.shading.use_scene_lights = True


def _bbox(objs):
    """World-space bounds of the evaluated geometry.

    ``Object.bound_box`` is not reliable for curve objects (it can report a
    placeholder unit cube), so the evaluated mesh vertices are measured instead.
    """
    import numpy as np

    deps = bpy.context.evaluated_depsgraph_get()
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for o in objs:
        if o.type not in ("MESH", "CURVE", "FONT", "SURFACE"):
            continue
        ev = o.evaluated_get(deps)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            continue
        try:
            n = len(me.vertices)
            if n == 0:
                continue
            co = np.empty(n * 3)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            m = np.array(ev.matrix_world)
            w = co @ m[:3, :3].T + m[:3, 3]
            lo = np.minimum(lo, w.min(axis=0))
            hi = np.maximum(hi, w.max(axis=0))
        finally:
            ev.to_mesh_clear()
    if not np.all(np.isfinite(lo)):
        raise ValueError("Nothing to frame")
    return mathutils.Vector(lo.tolist()), mathutils.Vector(hi.tolist())


def cmd_frame(params):
    """Fit the camera to the given objects (or all math objects) with a margin."""
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise ValueError("No camera; call setup_scene first")
    names = params.get("names")
    objs = [_get_obj(n) for n in names] if names else _math_objects()
    lo, hi = _bbox(objs)
    margin = float(params.get("margin", 0.08))
    center = (lo + hi) / 2
    size = hi - lo
    aspect = scene.render.resolution_x / scene.render.resolution_y
    if cam.data.type == "ORTHO":
        # Ortho camera looks down -Z in its own frame; assume a top view for 2D scenes.
        w, h = size.x, size.y
        scale = max(w, h * aspect) * (1 + 2 * margin)
        cam.data.ortho_scale = max(scale, 1e-3)
        cam.location = (center.x, center.y, hi.z + 50.0)
        cam.rotation_euler = (0, 0, 0)
    else:
        # Tight fit: move the camera along its viewing direction until every
        # bounding-box corner is inside the frustum (with margin).
        rot = cam.matrix_world.to_quaternion()
        inv = rot.inverted()
        forward = rot @ mathutils.Vector((0, 0, -1))
        cam.data.sensor_fit = "AUTO"
        ax, ay = cam.data.angle_x, cam.data.angle_y
        if aspect < 1:  # portrait: AUTO fit applies the angle to the height
            ax, ay = ay, ax
        tx, ty = math.tan(ax / 2) / (1 + 2 * margin), math.tan(ay / 2) / (1 + 2 * margin)
        dist = 0.0
        for cx in (lo.x, hi.x):
            for cy in (lo.y, hi.y):
                for cz in (lo.z, hi.z):
                    c = inv @ (mathutils.Vector((cx, cy, cz)) - center)  # camera-space offset
                    dist = max(dist, abs(c.x) / tx + c.z, abs(c.y) / ty + c.z)
        cam.location = center - forward * max(dist, cam.data.clip_start * 2)
    return {"center": list(center), "size": list(size), "camera": cam.name}


def cmd_render(params):
    """Render the camera view and return a base64 PNG."""
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError("No camera; call setup_scene first")
    r = scene.render
    saved = (r.resolution_x, r.resolution_y, r.resolution_percentage, r.filepath, r.engine,
             r.image_settings.file_format)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="math_render_")
    os.close(fd)
    try:
        res = params.get("resolution")
        if res:
            r.resolution_x, r.resolution_y = int(res[0]), int(res[1])
            r.resolution_percentage = 100
        if params.get("engine"):
            _set_engine(scene, params["engine"])
        if r.engine == "CYCLES" and params.get("samples"):
            scene.cycles.samples = int(params["samples"])
        elif params.get("samples") and hasattr(scene, "eevee"):
            with contextlib.suppress(AttributeError):
                scene.eevee.taa_render_samples = int(params["samples"])
        r.image_settings.file_format = "PNG"
        r.filepath = path
        bpy.ops.render.render(write_still=True)
        with open(path, "rb") as fh:
            data = fh.read()
        return {"png_base64": base64.b64encode(data).decode("ascii"),
                "width": r.resolution_x, "height": r.resolution_y, "engine": r.engine}
    finally:
        (r.resolution_x, r.resolution_y, r.resolution_percentage, r.filepath, r.engine,
         r.image_settings.file_format) = saved
        with contextlib.suppress(OSError):
            os.remove(path)


def cmd_execute_code(params):
    prefs = _prefs()
    if prefs is not None and not prefs.allow_code_execution:
        raise PermissionError(
            "Arbitrary code execution is disabled. Enable it in the Math MCP panel / add-on preferences.")
    buf = io.StringIO()
    ns = {"bpy": bpy, "mathutils": mathutils, "math": math}
    with contextlib.redirect_stdout(buf):
        exec(params["code"], ns)  # noqa: S102 - explicit, opt-in feature
    return {"stdout": buf.getvalue()}


COMMANDS = {
    "ping": cmd_ping,
    "create_curve": cmd_create_curve,
    "create_mesh": cmd_create_mesh,
    "create_empty": cmd_create_empty,
    "get_scene_info": cmd_get_scene_info,
    "get_object": cmd_get_object,
    "delete": cmd_delete,
    "transform": cmd_transform,
    "set_material": cmd_set_material,
    "setup_scene": cmd_setup_scene,
    "frame": cmd_frame,
    "render": cmd_render,
    "execute_code": cmd_execute_code,
}


def dispatch(request):
    """Run one request dict on the current (main) thread and return a response dict."""
    try:
        kind = request.get("type")
        fn = COMMANDS.get(kind)
        if fn is None:
            raise ValueError(f"Unknown command {kind!r}")
        result = fn(request.get("params") or {})
        return {"status": "ok", "result": result}
    except Exception as exc:  # report every failure to the caller
        return {"status": "error", "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=4)}


# =============================================================================== server


class BridgeServer:
    """Accepts connections on a thread; executes commands on Blender's main thread."""

    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT):
        self.host, self.port = host, port
        self.jobs = queue.Queue()
        self._sock = None
        self._thread = None
        self._running = False

    @property
    def running(self):
        return self._running

    def start(self, use_timer=True):
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="MathBridgeAccept")
        self._thread.start()
        if use_timer and not bpy.app.timers.is_registered(self._pump):
            bpy.app.timers.register(self._pump, persistent=True)
        print(f"[Math Bridge] listening on {self.host}:{self.port}")

    def stop(self):
        self._running = False
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        self._sock = None
        if bpy.app.timers.is_registered(self._pump):
            bpy.app.timers.unregister(self._pump)
        print("[Math Bridge] stopped")

    # -- worker threads -------------------------------------------------------
    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn):
        conn.settimeout(None)
        f = conn.makefile("rwb")
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    response = {"status": "error", "message": f"Bad JSON: {exc}"}
                else:
                    done = threading.Event()
                    box = {}
                    self.jobs.put((request, box, done))
                    done.wait()
                    response = box["response"]
                f.write((json.dumps(response) + "\n").encode("utf-8"))
                f.flush()
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError):
                f.close()
                conn.close()

    # -- main thread ----------------------------------------------------------
    def _pump(self):
        self.pump()
        return 0.05 if self._running else None

    def pump(self):
        """Execute all queued jobs (must run on Blender's main thread)."""
        while True:
            try:
                request, box, done = self.jobs.get_nowait()
            except queue.Empty:
                return
            box["response"] = dispatch(request)
            if box["response"]["status"] == "ok" and request.get("type") not in ("ping", "get_scene_info",
                                                                                  "get_object", "render"):
                with contextlib.suppress(Exception):
                    bpy.ops.ed.undo_push(message=f"Math: {request.get('type')}")
            done.set()


_server = None


def get_server():
    return _server


# =============================================================================== UI


def _prefs():
    addon = bpy.context.preferences.addons.get(__name__)
    return addon.preferences if addon else None


class MATHBRIDGE_Preferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    port: bpy.props.IntProperty(name="Port", default=DEFAULT_PORT, min=1024, max=65535)
    auto_start: bpy.props.BoolProperty(name="Start server when Blender starts", default=False)
    allow_code_execution: bpy.props.BoolProperty(
        name="Allow arbitrary Python execution",
        description="Let the AI run any Python code in Blender (execute_blender_code tool). Off = safer",
        default=False,
    )

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "port")
        col.prop(self, "auto_start")
        col.prop(self, "allow_code_execution")


class MATHBRIDGE_OT_start(bpy.types.Operator):
    bl_idname = "mathbridge.start"
    bl_label = "Start Math MCP server"

    def execute(self, context):
        global _server
        prefs = _prefs()
        port = prefs.port if prefs else DEFAULT_PORT
        if _server and _server.running:
            return {"FINISHED"}
        try:
            _server = BridgeServer(port=port)
            _server.start()
        except OSError as exc:
            self.report({"ERROR"}, f"Could not listen on port {port}: {exc}")
            return {"CANCELLED"}
        return {"FINISHED"}


class MATHBRIDGE_OT_stop(bpy.types.Operator):
    bl_idname = "mathbridge.stop"
    bl_label = "Stop Math MCP server"

    def execute(self, context):
        if _server:
            _server.stop()
        return {"FINISHED"}


class MATHBRIDGE_PT_panel(bpy.types.Panel):
    bl_label = "Math MCP"
    bl_idname = "MATHBRIDGE_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Math MCP"

    def draw(self, context):
        layout = self.layout
        prefs = _prefs()
        running = bool(_server and _server.running)
        if prefs:
            layout.prop(prefs, "port")
            layout.prop(prefs, "allow_code_execution")
        if running:
            layout.label(text=f"Listening on 127.0.0.1:{_server.port}", icon="CHECKMARK")
            layout.operator("mathbridge.stop", icon="PAUSE")
        else:
            layout.label(text="Server stopped", icon="X")
            layout.operator("mathbridge.start", icon="PLAY")


CLASSES = (MATHBRIDGE_Preferences, MATHBRIDGE_OT_start, MATHBRIDGE_OT_stop, MATHBRIDGE_PT_panel)


def _auto_start():
    prefs = _prefs()
    if prefs and prefs.auto_start:
        bpy.ops.mathbridge.start()
    return None


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.app.timers.register(_auto_start, first_interval=1.0)


def unregister():
    if _server:
        _server.stop()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
