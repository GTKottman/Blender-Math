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
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
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
    # Every material ends in a mix with transparency (node "MathAlpha", Fac = opacity)
    # so any object can be faded in/out by keyframing one value.
    alpha = float(color[3])
    transp = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.name = mix.label = "MathAlpha"
    mix.inputs["Fac"].default_value = alpha
    nt.links.new(transp.outputs[0], mix.inputs[1])
    nt.links.new(last, mix.inputs[2])
    _set_blend(mat, alpha)
    final = mix.outputs[0]
    box = spec.get("clip_box")  # [xmin, xmax, ymin, ymax] in object space: hide outside
    if box:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(tc.outputs["Object"], sep.inputs[0])
        mask = None
        for axis, lo, hi in (("X", box[0], box[1]), ("Y", box[2], box[3])):
            gt = nt.nodes.new("ShaderNodeMath")
            gt.operation = "GREATER_THAN"
            nt.links.new(sep.outputs[axis], gt.inputs[0])
            gt.inputs[1].default_value = float(lo)
            lt = nt.nodes.new("ShaderNodeMath")
            lt.operation = "LESS_THAN"
            nt.links.new(sep.outputs[axis], lt.inputs[0])
            lt.inputs[1].default_value = float(hi)
            both = nt.nodes.new("ShaderNodeMath")
            both.operation = "MULTIPLY"
            nt.links.new(gt.outputs[0], both.inputs[0])
            nt.links.new(lt.outputs[0], both.inputs[1])
            if mask is None:
                mask = both
            else:
                m2 = nt.nodes.new("ShaderNodeMath")
                m2.operation = "MULTIPLY"
                nt.links.new(mask.outputs[0], m2.inputs[0])
                nt.links.new(both.outputs[0], m2.inputs[1])
                mask = m2
        transp2 = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix2 = nt.nodes.new("ShaderNodeMixShader")
        mix2.name = "MathMask"
        nt.links.new(mask.outputs[0], mix2.inputs["Fac"])
        nt.links.new(transp2.outputs[0], mix2.inputs[1])
        nt.links.new(final, mix2.inputs[2])
        final = mix2.outputs[0]
    nt.links.new(final, out.inputs["Surface"])
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
    """Delete by names (missing names are ignored), by node (metadata), or all math objects."""
    removed = []
    if params.get("all_math"):
        targets = _math_objects()
    else:
        targets = [bpy.data.objects[n] for n in params.get("names", []) if n in bpy.data.objects]
        node = params.get("node")
        if node:
            for o in bpy.data.objects:
                if META_KEY in o:
                    with contextlib.suppress(Exception):
                        if json.loads(o[META_KEY]).get("node") == node:
                            targets.append(o)
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
    # A fresh setup owns the camera: drop orbit rigs and camera animation from earlier scenes.
    cam.animation_data_clear()
    cam.data.animation_data_clear()
    if cam.parent is not None:
        cam.parent = None
        cam.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    piv = bpy.data.objects.get("MathCameraPivot")
    if piv is not None:
        bpy.data.objects.remove(piv, do_unlink=True)
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


def _world_points(objs, max_points=20000):
    """Evaluated vertex positions (world space) of the objects, subsampled to max_points."""
    import numpy as np

    deps = bpy.context.evaluated_depsgraph_get()
    chunks = []
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
            if n:
                co = np.empty(n * 3)
                me.vertices.foreach_get("co", co)
                m = np.array(ev.matrix_world)
                chunks.append(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
        finally:
            ev.to_mesh_clear()
    if not chunks:
        return None
    pts = np.concatenate(chunks)
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    return pts


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
    if params.get("center") is not None:
        # keep a fixed point (e.g. the math origin) at the center of the view
        c = mathutils.Vector(params["center"])
        half = mathutils.Vector([max(abs(lo[i] - c[i]), abs(hi[i] - c[i])) for i in range(3)])
        lo, hi, center = c - half, c + half, c
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
        # Tight fit on the actual geometry: each vertex must be inside the frustum.
        import numpy as np

        pts = _world_points(objs)
        R = np.array(inv.to_matrix())
        cam_pts = (pts - np.array(center)) @ R.T  # camera-space offsets from the target
        # re-center the view on the projected extent, then find the distance
        dist = float(np.max(np.maximum(np.abs(cam_pts[:, 0]) / tx, np.abs(cam_pts[:, 1]) / ty) + cam_pts[:, 2]))
        cam.location = center - forward * max(dist, cam.data.clip_start * 2)
    out = {"center": list(center), "size": list(size), "camera": cam.name, "aspect": aspect}
    if cam.data.type == "ORTHO":
        w = cam.data.ortho_scale
        out["view_box"] = [cam.location.x - w / 2, cam.location.y - w / aspect / 2,
                           cam.location.x + w / 2, cam.location.y + w / aspect / 2]
    return out


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
        if params.get("frame") is not None:
            scene.frame_set(int(params["frame"]))
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


# =============================================================================== storage / bounds


STORE_KEY = "math_construction"


def cmd_store(params):
    """Persist (or read back) the MCP server's construction graph inside the .blend file."""
    scene = bpy.context.scene
    if "data" in params:
        scene[STORE_KEY] = json.dumps(params["data"])
        return {"stored": True}
    raw = scene.get(STORE_KEY)
    return {"data": json.loads(raw) if raw else None}


def cmd_bounds(params):
    """World-space bounding boxes and centers of objects (children included)."""
    out = {}
    for n in params.get("names", []):
        obj = _get_obj(n)
        objs = [obj] + list(obj.children_recursive)
        try:
            lo, hi = _bbox(objs)
        except ValueError:
            loc = obj.matrix_world.translation
            lo = hi = loc
        out[n] = {"min": list(lo), "max": list(hi), "center": list((lo + hi) / 2), "size": list(hi - lo)}
    return out


# =============================================================================== animation


def _alpha_sockets(obj, include_children=True):
    objs = [obj] + (list(obj.children_recursive) if include_children else [])
    for o in objs:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m and m.use_nodes and "MathAlpha" in m.node_tree.nodes:
                yield m, m.node_tree.nodes["MathAlpha"].inputs["Fac"]


@contextlib.contextmanager
def _interpolation(kind):
    edit = bpy.context.preferences.edit
    old = edit.keyframe_new_interpolation_type
    edit.keyframe_new_interpolation_type = kind
    try:
        yield
    finally:
        edit.keyframe_new_interpolation_type = old


def _resolve(obj, path):
    owner = obj
    parts = path.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    return owner, parts[-1]


def _set_value(owner, prop, index, value):
    if index is None or index < 0:
        if isinstance(value, (list, tuple)):
            setattr(owner, prop, value)
        else:
            setattr(owner, prop, value)
    else:
        getattr(owner, prop)[index] = value


def cmd_keyframes(params):
    """Insert keyframes.  items: [{object, path, keys: [[frame, value], ...], index, interpolation,
    include_children}].  Special paths: 'alpha' (material opacity), 'visible' (hide keys),
    'shape:<name>' (shape key value), 'shape_eval_time'."""
    count = 0
    for it in params.get("items", []):
        obj = _get_obj(it["object"])
        path = it["path"]
        index = it.get("index")
        interp = it.get("interpolation", "BEZIER")
        with _interpolation("CONSTANT" if path == "visible" else interp):
            for frame, value in it["keys"]:
                frame = float(frame)
                if path == "alpha":
                    for _m, sock in _alpha_sockets(obj, it.get("include_children", True)):
                        sock.default_value = float(value)
                        sock.keyframe_insert("default_value", frame=frame)
                elif path == "visible":
                    targets = [obj] + (list(obj.children_recursive) if it.get("include_children", True) else [])
                    for o in targets:
                        o.hide_render = not bool(value)
                        o.hide_viewport = not bool(value)
                        o.keyframe_insert("hide_render", frame=frame)
                        o.keyframe_insert("hide_viewport", frame=frame)
                elif path.startswith("shape:"):
                    kb = obj.data.shape_keys.key_blocks[path[6:]]
                    kb.value = float(value)
                    kb.keyframe_insert("value", frame=frame)
                elif path == "shape_eval_time":
                    key = obj.data.shape_keys
                    key.eval_time = float(value)
                    key.keyframe_insert("eval_time", frame=frame)
                else:
                    owner, prop = _resolve(obj, path)
                    _set_value(owner, prop, index, value)
                    if index is None or index < 0:
                        owner.keyframe_insert(prop, frame=frame)
                    else:
                        owner.keyframe_insert(prop, index=index, frame=frame)
                count += 1
    return {"keyframes": count}


def cmd_shape_keys(params):
    """Add shape keys to a curve or mesh.  keys: [{name, co, hl?, hr?}] with one entry per point
    (curves: bezier points / poly points in spline order; meshes: vertices).
    relative=False makes absolute keys driven by eval_time (key i at eval_time 10*i)."""
    obj = _get_obj(params["object"])
    data = obj.data
    if data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)
    data.shape_keys.use_relative = bool(params.get("relative", True))
    for k in params["keys"]:
        kb = obj.shape_key_add(name=k["name"], from_mix=False)
        co = k["co"]
        if isinstance(data, bpy.types.Curve):
            for i, pt in enumerate(kb.data):
                pt.co = co[i]
                if hasattr(pt, "handle_left") and "hl" in k:
                    pt.handle_left = k["hl"][i]
                    pt.handle_right = k["hr"][i]
        else:
            flat = [c for v in co for c in v]
            kb.data.foreach_set("co", flat)
    return {"object": obj.name, "keys": [kb.name for kb in data.shape_keys.key_blocks]}


def cmd_timeline(params):
    scene = bpy.context.scene
    if "fps" in params:
        scene.render.fps = int(params["fps"])
        scene.render.fps_base = 1.0
    if "frame_start" in params:
        scene.frame_start = int(params["frame_start"])
    if "frame_end" in params:
        scene.frame_end = int(params["frame_end"])
    if "frame" in params:
        scene.frame_set(int(params["frame"]))
    return {"fps": scene.render.fps, "frame_start": scene.frame_start, "frame_end": scene.frame_end,
            "frame": scene.frame_current}


def cmd_camera_pivot(params):
    """Parent the camera to an empty at ``center`` (for orbits); returns the pivot name."""
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        raise ValueError("No camera; call setup_scene first")
    piv = bpy.data.objects.get("MathCameraPivot")
    if piv is None:
        piv = bpy.data.objects.new("MathCameraPivot", None)
        scene.collection.objects.link(piv)
    world = cam.matrix_world.copy()
    piv.location = params.get("center", [0, 0, 0])
    piv.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()
    cam.parent = piv
    cam.matrix_parent_inverse = piv.matrix_world.inverted()
    cam.matrix_world = world
    return {"pivot": piv.name, "camera": cam.name}


def cmd_render_animation(params):
    """Render the frame range to a video (MP4/H.264) or a PNG sequence."""
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError("No camera; call setup_scene first")
    r = scene.render
    path = params["filepath"]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    res = params.get("resolution")
    if res:
        r.resolution_x, r.resolution_y = int(res[0]), int(res[1])
        r.resolution_percentage = 100
    if params.get("engine"):
        _set_engine(scene, params["engine"])
    if params.get("samples"):
        if r.engine == "CYCLES":
            scene.cycles.samples = int(params["samples"])
        else:
            with contextlib.suppress(AttributeError):
                scene.eevee.taa_render_samples = int(params["samples"])
    for k in ("frame_start", "frame_end"):
        if params.get(k) is not None:
            setattr(scene, k, int(params[k]))
    fmt = params.get("format", "mp4")
    ims = r.image_settings
    if fmt == "mp4":
        if hasattr(ims, "media_type"):  # Blender 5.x
            ims.media_type = "VIDEO"
        ims.file_format = "FFMPEG"
        r.ffmpeg.format = "MPEG4"
        r.ffmpeg.codec = "H264"
        with contextlib.suppress(TypeError, AttributeError):
            r.ffmpeg.constant_rate_factor = "HIGH"
        r.filepath = path
    else:
        if hasattr(ims, "media_type"):
            ims.media_type = "IMAGE"
        ims.file_format = "PNG"
        r.filepath = os.path.join(path, "frame_")
    bpy.ops.render.render(animation=True)
    return {"filepath": path, "frames": [scene.frame_start, scene.frame_end], "fps": r.fps,
            "format": fmt}


def cmd_describe_objects(params):
    """Type information used to choose animations (draw-on for tubes, fades for fills ...)."""
    out = {}
    for n in params.get("names", []):
        o = bpy.data.objects.get(n)
        if o is None:
            continue
        info = {"type": o.type, "location": list(o.location), "scale": list(o.scale),
                "parent": o.parent.name if o.parent else None}
        if o.type == "CURVE":
            info["filled"] = o.data.dimensions == "2D" and o.data.fill_mode != "NONE"
            info["bevel"] = o.data.bevel_depth
            info["splines"] = len(o.data.splines)
        out[n] = info
    return out


def cmd_stroke_copy(params):
    """Duplicate a filled glyph curve as an outline tube (for handwriting-style 'write' animations)."""
    src = _get_obj(params["name"])
    new_name = params["new_name"]
    old = bpy.data.objects.get(new_name)
    if old is not None:
        _delete(old)
    data = src.data.copy()
    data.fill_mode = "NONE"
    data.bevel_depth = float(params.get("bevel_depth", 0.01))
    data.bevel_resolution = 2
    obj = bpy.data.objects.new(new_name, data)
    for c in src.users_collection:
        c.objects.link(obj)
    obj.parent = src.parent
    obj.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    obj.location = src.location
    obj.rotation_euler = src.rotation_euler
    obj.scale = src.scale
    obj.location.z += float(params.get("z_offset", 0.001))
    obj[META_KEY] = json.dumps({"node": params.get("node", "anim"), "part": "stroke", "anim_helper": True})
    _assign_material(obj, params.get("material"))
    return _obj_summary(obj)


def cmd_clear_animation(params):
    """Remove all keyframes from math objects (and their data/materials/shape keys) and delete helpers."""
    removed = 0
    for o in list(_math_objects()):
        meta = {}
        with contextlib.suppress(Exception):
            meta = json.loads(o.get(META_KEY, "{}"))
        if meta.get("anim_helper"):
            _delete(o)
            removed += 1
            continue
        o.animation_data_clear()
        o.hide_render = o.hide_viewport = False
        if o.data is not None and hasattr(o.data, "animation_data_clear"):
            o.data.animation_data_clear()
            sk = getattr(o.data, "shape_keys", None)
            if sk is not None:
                sk.animation_data_clear()
        for m, sock in _alpha_sockets(o, False):
            if m.node_tree:
                m.node_tree.animation_data_clear()
    cam = bpy.context.scene.camera
    if cam is not None:
        cam.animation_data_clear()
        cam.data.animation_data_clear()
    piv = bpy.data.objects.get("MathCameraPivot")
    if piv is not None:
        piv.animation_data_clear()
    return {"removed_helpers": removed}


COMMANDS = {
    "describe_objects": cmd_describe_objects,
    "stroke_copy": cmd_stroke_copy,
    "clear_animation": cmd_clear_animation,
    "store": cmd_store,
    "bounds": cmd_bounds,
    "keyframes": cmd_keyframes,
    "shape_keys": cmd_shape_keys,
    "timeline": cmd_timeline,
    "camera_pivot": cmd_camera_pivot,
    "render_animation": cmd_render_animation,
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
            if box["response"]["status"] == "ok" and request.get("type") not in (
                    "ping", "get_scene_info", "get_object", "render", "bounds", "store", "render_animation"):
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
