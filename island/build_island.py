"""
Procedural tropical island scene for Blender (bpy).

Usage (headless, via the bpy pip module):
    python3 -I build_island.py --engine CYCLES --quality preview --frame 60
    python3 -I build_island.py --engine CYCLES --quality final --animation

Everything in the scene (terrain, water, palms, rocks, clouds, camera path)
is generated from code, so no external assets are required.
"""
import argparse
import math
import os
import random
import sys

import bpy
import numpy as np
from mathutils import Vector

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves,
                 bpy.data.lights, bpy.data.cameras, bpy.data.worlds):
        for block in list(coll):
            coll.remove(block)


def keyframe_interp(mode):
    """Set the interpolation used by subsequently inserted keyframes."""
    bpy.context.preferences.edit.keyframe_new_interpolation_type = mode


def smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_object(name, verts, faces, smooth_shading=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    link(ob)
    if smooth_shading:
        smooth(ob)
    return ob


def new_material(name, color=(0.8, 0.8, 0.8, 1), roughness=0.5, **kw):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    for k, v in kw.items():
        bsdf.inputs[k].default_value = v
    return mat


def set_material(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# --------------------------------------------------------------------------
# noise (numpy value-noise fbm, no external deps)
# --------------------------------------------------------------------------

def _hash_grid(ix, iy, seed):
    # deterministic pseudo random in [0,1) for integer lattice points
    h = (ix * 374761393 + iy * 668265263 + seed * 982451653) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) * 1274126177 & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0x1000000)


def value_noise(x, y, seed=0):
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    a = _hash_grid(x0, y0, seed)
    b = _hash_grid(x0 + 1, y0, seed)
    c = _hash_grid(x0, y0 + 1, seed)
    d = _hash_grid(x0 + 1, y0 + 1, seed)
    return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy


def fbm(x, y, octaves=5, lacunarity=2.0, gain=0.5, seed=0):
    total = np.zeros_like(x)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for i in range(octaves):
        total += amp * value_noise(x * freq, y * freq, seed + i * 17)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


# --------------------------------------------------------------------------
# terrain
# --------------------------------------------------------------------------

ISLAND_RADIUS = 26.0
SEA_LEVEL = 0.0


def island_height(x, y, seed=3):
    """Height field: a soft dome with ridges, dropping below sea level at the rim."""
    r = np.sqrt(x * x + y * y)
    # domain warp for an irregular coastline
    wx = x + 9.0 * (fbm(x * 0.035 + 10, y * 0.035, 3, seed=seed + 1) - 0.5)
    wy = y + 9.0 * (fbm(x * 0.035, y * 0.035 + 10, 3, seed=seed + 2) - 0.5)
    rw = np.sqrt(wx * wx + wy * wy)
    dome = np.clip(1.0 - (rw / ISLAND_RADIUS) ** 1.6, -1.0, 1.0)
    n = fbm(x * 0.07, y * 0.07, 6, seed=seed) - 0.5
    hills = fbm(x * 0.16 + 3, y * 0.16 + 7, 5, seed=seed + 5)
    # a ridge-like central peak
    peak = np.exp(-((x - 4) ** 2 + (y + 2) ** 2) / (2 * 9.0 ** 2))
    h = dome * 6.0 + n * 20.0 * np.clip(dome, 0, 1) + (hills - 0.5) * 10.0 * np.clip(dome, 0, 1) ** 2 + peak * 6.0
    # beach: squash heights around sea level into a wide gentle shelf
    beach = np.clip((h + 0.3) / 2.0, 0, 1)
    h = np.where(h > 0, h * (0.35 + 0.65 * beach ** 1.5), h)
    # shallow lagoon floor: keep the sea bed near the island bright and shallow
    h = np.where(h < -1.4, -1.4 + (h + 1.4) * 0.5, h)
    return h


def make_island(size=140.0, res=380):
    xs = np.linspace(-size / 2, size / 2, res)
    X, Y = np.meshgrid(xs, xs)
    Z = island_height(X, Y)
    # sea floor far from the island slopes down gently
    r = np.sqrt(X ** 2 + Y ** 2)
    floor = np.maximum(-6.0, -1.2 - 0.30 * np.clip(r - ISLAND_RADIUS - 18, 0, None))
    Z = np.where(Z < floor, floor, Z)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = []
    for j in range(res - 1):
        base = j * res
        for i in range(res - 1):
            a = base + i
            faces.append((a, a + 1, a + res + 1, a + res))
    ob = mesh_object("Island", verts.tolist(), faces)
    return ob, Z, xs


def island_material():
    mat = bpy.data.materials.new("IslandMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Specular IOR Level"].default_value = 0.2

    geo = nodes.new("ShaderNodeNewGeometry")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geo.outputs["Position"], sep.inputs[0])

    # small noise to break up the gradient
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 0.6
    noise.inputs["Detail"].default_value = 6
    nmath = nodes.new("ShaderNodeMath")
    nmath.operation = "MULTIPLY_ADD"
    nmath.inputs[1].default_value = 1.0
    nmath.inputs[2].default_value = -0.5
    links.new(noise.outputs["Fac"], nmath.inputs[0])
    hmix = nodes.new("ShaderNodeMath")
    hmix.operation = "ADD"
    links.new(sep.outputs["Z"], hmix.inputs[0])
    links.new(nmath.outputs[0], hmix.inputs[1])

    ramp = nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "EASE"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.06, 0.20, 0.32, 1)   # deep sea floor
    e = cr.elements.new(0.20); e.color = (0.06, 0.20, 0.32, 1)
    e = cr.elements.new(0.30); e.color = (0.90, 0.84, 0.62, 1)  # wet sand
    e = cr.elements.new(0.385); e.color = (1.0, 0.96, 0.82, 1)  # dry sand
    e = cr.elements.new(0.42); e.color = (0.75, 0.86, 0.34, 1)  # light grass
    e = cr.elements.new(0.52); e.color = (0.25, 0.62, 0.20, 1)  # lush green
    e = cr.elements.new(0.84); e.color = (0.12, 0.42, 0.17, 1)  # dense jungle
    cr.elements[-1].color = (0.10, 0.36, 0.15, 1)              # summit stays green
    # map Z in [-2, 7] to [0,1]
    rng = nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = -7.0
    rng.inputs["From Max"].default_value = 12.0
    links.new(hmix.outputs[0], rng.inputs["Value"])
    links.new(rng.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # fine bump for texture
    bump_noise = nodes.new("ShaderNodeTexNoise")
    bump_noise.inputs["Scale"].default_value = 6.0
    bump_noise.inputs["Detail"].default_value = 8
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    links.new(bump_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


# --------------------------------------------------------------------------
# water
# --------------------------------------------------------------------------

def make_water(size=1000.0):
    keyframe_interp("LINEAR")
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, SEA_LEVEL))
    water = bpy.context.active_object
    water.name = "Water"
    mod = water.modifiers.new("Ocean", "OCEAN")
    mod.geometry_mode = "DISPLACE"
    mod.resolution = 10
    mod.viewport_resolution = 8
    mod.size = 2.0
    mod.spatial_size = 50
    mod.repeat_x = 8
    mod.repeat_y = 8
    mod.wave_scale = 0.22
    mod.wave_scale_min = 0.02
    mod.choppiness = 1.1
    mod.wind_velocity = 7.0
    mod.wave_alignment = 0.6
    mod.depth = 200
    mod.use_normals = True
    mod.use_foam = True
    mod.foam_coverage = 0.08
    mod.foam_layer_name = "foam"
    # subdivide the base plane so displacement has geometry to push
    sub = water.modifiers.new("Subdiv", "SUBSURF")
    sub.subdivision_type = "SIMPLE"
    sub.levels = 6
    sub.render_levels = 9
    water.modifiers.move(1, 0)
    smooth(water)

    mat = bpy.data.materials.new("WaterMat")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.10, 0.70, 0.75, 1)
    bsdf.inputs["Roughness"].default_value = 0.10
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.35

    # volume absorption makes shallow water turquoise and deep water blue
    absorb = nodes.new("ShaderNodeVolumeAbsorption")
    absorb.inputs["Color"].default_value = (0.25, 0.80, 0.95, 1)
    absorb.inputs["Density"].default_value = 0.40
    out = nodes["Material Output"]
    links.new(absorb.outputs["Volume"], out.inputs["Volume"])

    # ripple bump on top of the ocean geometry
    tex = nodes.new("ShaderNodeTexNoise")
    tex.noise_dimensions = "4D"
    tex.inputs["Scale"].default_value = 3.0
    tex.inputs["Detail"].default_value = 6
    tex.inputs["Roughness"].default_value = 0.55
    tex.inputs["W"].default_value = 0.0
    tex.inputs["W"].keyframe_insert("default_value", frame=1)
    tex.inputs["W"].default_value = 2.5
    tex.inputs["W"].keyframe_insert("default_value", frame=bpy.context.scene.frame_end)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.1
    links.new(tex.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # foam: whiten crests via the ocean modifier's foam attribute
    attr = nodes.new("ShaderNodeAttribute")
    attr.attribute_name = "foam"
    foam_ramp = nodes.new("ShaderNodeValToRGB")
    foam_ramp.color_ramp.elements[0].position = 0.15
    foam_ramp.color_ramp.elements[1].position = 0.6
    links.new(attr.outputs["Fac"], foam_ramp.inputs["Fac"])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.05, 0.55, 0.62, 1)
    mix.inputs[7].default_value = (0.95, 0.98, 1.0, 1)
    links.new(foam_ramp.outputs["Color"], mix.inputs[0])
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    tmix = nodes.new("ShaderNodeMath")
    tmix.operation = "SUBTRACT"
    tmix.inputs[0].default_value = 1.0
    links.new(foam_ramp.outputs["Color"], tmix.inputs[1])
    links.new(tmix.outputs[0], bsdf.inputs["Transmission Weight"])

    # shadow rays pass straight through the surface (fake caustics)
    lp = nodes.new("ShaderNodeLightPath")
    transp = nodes.new("ShaderNodeBsdfTransparent")
    mixsh = nodes.new("ShaderNodeMixShader")
    links.new(lp.outputs["Is Shadow Ray"], mixsh.inputs[0])
    links.new(bsdf.outputs[0], mixsh.inputs[1])
    links.new(transp.outputs[0], mixsh.inputs[2])
    links.new(mixsh.outputs[0], out.inputs["Surface"])

    set_material(water, mat)
    # far water: a huge plane with a square hole where the detailed ocean sits
    R, r = 30000.0, size / 2 - 1.0
    verts = [(-R, -R, 0), (R, -R, 0), (R, R, 0), (-R, R, 0),
             (-r, -r, 0), (r, -r, 0), (r, r, 0), (-r, r, 0)]
    faces = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    far = mesh_object("FarWater", verts, faces, smooth_shading=False)
    far.location = (0, 0, SEA_LEVEL - 0.05)
    set_material(far, mat)
    # sea floor everywhere beyond the terrain mesh
    bpy.ops.mesh.primitive_plane_add(size=60000, location=(0, 0, -6.0))
    floor = bpy.context.active_object
    floor.name = "SeaFloor"
    set_material(floor, new_material("SeaFloorMat", (0.06, 0.20, 0.32, 1), 0.9))
    # animate ocean time
    mod.time = 0.0
    mod.keyframe_insert("time", frame=1)
    mod.time = 6.0
    mod.keyframe_insert("time", frame=bpy.context.scene.frame_end)
    return water


# --------------------------------------------------------------------------
# vegetation & props
# --------------------------------------------------------------------------

def make_frond(length=4.0, width=0.5, droop=2.8, segments=12):
    verts, faces = [], []
    for i in range(segments + 1):
        t = i / segments
        w = width * math.sin(math.pi * min(1.0, t * 1.1)) * (1 - 0.25 * t) + 0.02
        y = length * t
        z = 0.35 * t - droop * t * t
        verts.append((-w, y, z - 0.25 * w))
        verts.append((0.0, y, z))
        verts.append((w, y, z - 0.25 * w))
    for i in range(segments):
        a = i * 3
        faces.append((a, a + 3, a + 4, a + 1))
        faces.append((a + 1, a + 4, a + 5, a + 2))
    return verts, faces


def make_palm(loc, height=6.0, lean=0.25, rot=0.0, seed=0, trunk_mat=None, leaf_mat=None):
    rnd = random.Random(seed)
    # trunk: a bezier curve with a bevel
    cu = bpy.data.curves.new("PalmTrunk", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.16
    cu.bevel_resolution = 4
    cu.fill_mode = "FULL"
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(2)
    top = Vector((lean * height * math.cos(rot), lean * height * math.sin(rot), height))
    pts = [Vector((0, 0, -0.3)), top * 0.5 + Vector((0, 0, 0.3)), top]
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    sp.bezier_points[0].radius = 1.3
    sp.bezier_points[2].radius = 0.75
    trunk = bpy.data.objects.new("PalmTrunk", cu)
    link(trunk)
    trunk.location = loc
    trunk.data.materials.append(trunk_mat)

    # crown
    crown = bpy.data.objects.new("PalmCrown", None)
    link(crown)
    crown.location = Vector(loc) + top
    n = rnd.randint(12, 15)
    fv, ff = make_frond()
    for i in range(n):
        f = mesh_object("Frond", fv, ff)
        f.data.materials.append(leaf_mat)
        f.parent = crown
        ang = i * 2 * math.pi / n + rnd.uniform(-0.2, 0.2)
        tilt = rnd.uniform(0.05, 0.3) if i % 2 == 0 else rnd.uniform(0.5, 0.85)
        f.rotation_euler = (tilt, rnd.uniform(-0.15, 0.15), ang)
        s = rnd.uniform(0.85, 1.15)
        f.scale = (s, s, s)
    # coconuts
    for i in range(3):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.16, segments=12, ring_count=8)
        c = bpy.context.active_object
        c.parent = crown
        a = i * 2.1
        c.location = (0.25 * math.cos(a), 0.25 * math.sin(a), -0.15)
        c.data.materials.append(trunk_mat)
        smooth(c)
    return trunk


def make_rock(loc, radius=1.0, seed=0, mat=None):
    rnd = random.Random(seed)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=radius, location=loc)
    rock = bpy.context.active_object
    me = rock.data
    for v in me.vertices:
        v.co *= rnd.uniform(0.75, 1.25)
    rock.scale = (rnd.uniform(0.8, 1.4), rnd.uniform(0.8, 1.4), rnd.uniform(0.5, 0.8))
    rock.rotation_euler = (0, 0, rnd.uniform(0, 6.28))
    smooth(rock)
    rock.data.materials.append(mat)
    return rock


def make_bush(loc, radius=1.0, seed=0, mat=None):
    rnd = random.Random(seed)
    for i in range(rnd.randint(3, 5)):
        r = radius * rnd.uniform(0.5, 1.0)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=r,
            location=(loc[0] + rnd.uniform(-radius, radius) * 0.7,
                      loc[1] + rnd.uniform(-radius, radius) * 0.7,
                      loc[2] + r * 0.45))
        b = bpy.context.active_object
        b.scale = (1.0, 1.0, 0.7)
        smooth(b)
        b.data.materials.append(mat)


def make_cloud(loc, scale=1.0, seed=0, mat=None):
    rnd = random.Random(seed)
    parts = []
    n = rnd.randint(5, 8)
    for i in range(n):
        r = rnd.uniform(1.6, 3.2) * scale
        off = (rnd.uniform(-3.5, 3.5) * scale, rnd.uniform(-1.5, 1.5) * scale, rnd.uniform(-0.3, 0.9) * scale)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=24, ring_count=12,
                                             location=(loc[0] + off[0], loc[1] + off[1], loc[2] + off[2]))
        s = bpy.context.active_object
        s.scale = (1.0, 1.0, 0.55)
        s.visible_shadow = False
        s.visible_glossy = False
        s.visible_transmission = False
        smooth(s)
        s.data.materials.append(mat)
        parts.append(s)
    return parts


# --------------------------------------------------------------------------
# world, light, camera
# --------------------------------------------------------------------------

def make_world(sun_elev_deg=52.0, sun_rot_deg=215.0):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes, links = nt.nodes, nt.links
    bg = nodes["Background"]
    # view-direction based gradient: cyan-white horizon -> deep saturated blue zenith
    texco = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(texco.outputs["Generated"], sep.inputs[0])
    rng = nodes.new("ShaderNodeMapRange")
    rng.inputs["From Min"].default_value = -0.02
    rng.inputs["From Max"].default_value = 0.5
    links.new(sep.outputs["Z"], rng.inputs["Value"])
    ramp = nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.interpolation = "EASE"
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.55, 0.80, 1.0, 1)     # bright horizon haze
    e = cr.elements.new(0.2); e.color = (0.40, 0.72, 1.0, 1)
    e = cr.elements.new(0.55); e.color = (0.15, 0.48, 0.98, 1)
    cr.elements[-1].position = 1.0
    cr.elements[-1].color = (0.05, 0.28, 0.82, 1)   # zenith
    links.new(rng.outputs["Result"], ramp.inputs["Fac"])
    # below the horizon (only visible in reflections) keep a sea-blue
    below = nodes.new("ShaderNodeMix")
    below.data_type = "RGBA"
    below.inputs[6].default_value = (0.08, 0.30, 0.55, 1)
    links.new(ramp.outputs["Color"], below.inputs[7])
    step = nodes.new("ShaderNodeMath")
    step.operation = "GREATER_THAN"
    step.inputs[1].default_value = -0.02
    links.new(sep.outputs["Z"], step.inputs[0])
    links.new(step.outputs[0], below.inputs[0])
    links.new(below.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0

    sun = bpy.data.lights.new("Sun", "SUN")
    sun.energy = 4.5
    sun.angle = math.radians(2.0)
    sun.color = (1.0, 0.96, 0.9)
    sun_ob = bpy.data.objects.new("Sun", sun)
    link(sun_ob)
    # point the sun to match the sky texture
    el = math.radians(sun_elev_deg)
    az = math.radians(sun_rot_deg)
    d = Vector((math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)))
    sun_ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    return world


def make_camera(frame_end):
    keyframe_interp("BEZIER")
    cam = bpy.data.cameras.new("Camera")
    cam.lens = 32
    cam.sensor_width = 36
    cam.dof.use_dof = False
    cam_ob = bpy.data.objects.new("Camera", cam)
    link(cam_ob)
    bpy.context.scene.camera = cam_ob

    target = bpy.data.objects.new("CamTarget", None)
    link(target)
    target.location = (0, 0, 2.0)
    con = cam_ob.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"

    # path: start low and far off the coast, sweep around and rise to a reveal
    keys = [
        (1,         (-96, -62, 20.0)),
        (frame_end, (52, -62, 56.0)),
    ]
    for f, loc in keys:
        cam_ob.location = loc
        cam_ob.keyframe_insert("location", frame=f)
    # bend the path into an arc
    mid = frame_end // 2
    cam_ob.location = (-28, -100, 38.0)
    cam_ob.keyframe_insert("location", frame=mid)
    # slight ease-in/out already from bezier; make the middle key smooth
    target.location = (0, 0, 2.0)
    target.keyframe_insert("location", frame=1)
    target.location = (0, 0, 4.0)
    target.keyframe_insert("location", frame=frame_end)
    return cam_ob


# --------------------------------------------------------------------------
# scene assembly
# --------------------------------------------------------------------------

def build(frame_end=192, fps=24):
    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_end
    scene.render.fps = fps

    # ---- terrain
    island, Z, xs = make_island()
    set_material(island, island_material())

    # ---- water
    make_water()

    # ---- materials
    trunk_mat = new_material("Trunk", (0.36, 0.24, 0.12, 1), 0.8)
    leaf_mat = new_material("Leaf", (0.16, 0.55, 0.18, 1), 0.55)
    leaf_mat.node_tree.nodes["Principled BSDF"].inputs["Sheen Weight"].default_value = 0.3
    rock_mat = new_material("Rock", (0.17, 0.16, 0.15, 1), 0.9)
    cloud_mat = new_material("Cloud", (1.0, 1.0, 1.0, 1), 1.0)
    cbsdf = cloud_mat.node_tree.nodes["Principled BSDF"]
    cbsdf.inputs["Emission Color"].default_value = (1.0, 1.0, 1.0, 1)
    cbsdf.inputs["Emission Strength"].default_value = 0.55

    # sample terrain height for placement
    def h_at(x, y):
        return float(island_height(np.array([x]), np.array([y]))[0])

    rnd = random.Random(7)
    # ---- palms: near the beach ring, on sand/low grass
    placed = 0
    tries = 0
    while placed < 34 and tries < 8000:
        tries += 1
        a = rnd.uniform(0, 2 * math.pi)
        r = rnd.uniform(10, 30)
        x, y = r * math.cos(a), r * math.sin(a)
        h = h_at(x, y)
        if 0.3 < h < 3.0:
            lean_dir = a + rnd.uniform(-0.6, 0.6)  # lean towards the sea
            make_palm((x, y, h - 0.1), height=rnd.uniform(4.5, 7.5), lean=rnd.uniform(0.15, 0.4),
                      rot=lean_dir, seed=placed, trunk_mat=trunk_mat, leaf_mat=leaf_mat)
            placed += 1

    # ---- bushes on the slopes
    bush_mats = [new_material("Bush%d" % i, c, 0.6) for i, c in enumerate(
        [(0.10, 0.45, 0.14, 1), (0.18, 0.56, 0.16, 1), (0.30, 0.62, 0.18, 1)])]
    for i in range(140):
        a = rnd.uniform(0, 2 * math.pi)
        r = rnd.uniform(0, 30)
        x, y = r * math.cos(a), r * math.sin(a)
        h = h_at(x, y)
        if 1.2 < h < 16:
            make_bush((x, y, h - 0.2), radius=rnd.uniform(0.8, 2.2), seed=i, mat=rnd.choice(bush_mats))

    # ---- rocks along the shore and in the water
    for i in range(22):
        a = rnd.uniform(0, 2 * math.pi)
        r = rnd.uniform(14, 36)
        x, y = r * math.cos(a), r * math.sin(a)
        h = h_at(x, y)
        make_rock((x, y, max(h, -0.6) + 0.1), radius=rnd.uniform(0.5, 1.6), seed=i, mat=rock_mat)

    # ---- clouds
    for i, (x, y, z, s) in enumerate([(-80, 190, 40, 3.0), (60, 230, 46, 3.6), (200, 90, 42, 2.8),
                                       (-220, 60, 44, 3.2), (240, -60, 50, 3.2), (-30, 280, 52, 4.0),
                                       (140, 200, 48, 3.0), (-160, 160, 38, 2.4), (300, 40, 56, 3.8)]):
        make_cloud((x, y, z), scale=s, seed=i, mat=cloud_mat)

    # ---- world + camera
    make_world()
    make_camera(frame_end)

    # ---- color management: bright, punchy, clean
    scene.view_settings.view_transform = os.environ.get("VIEW", "AgX")
    scene.view_settings.look = os.environ.get("LOOK", "AgX - Punchy")
    scene.view_settings.exposure = float(os.environ.get("EXPOSURE", "0.25"))
    scene.view_settings.gamma = 1.0
    return scene


def configure_render(scene, engine, quality, out_dir):
    scene.render.engine = engine
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    if quality == "preview":
        scene.render.resolution_x, scene.render.resolution_y = 640, 360
        samples = 24
    elif quality == "medium":
        scene.render.resolution_x, scene.render.resolution_y = 1280, 720
        samples = 32
    else:
        scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
        samples = 64
    scene.render.resolution_percentage = 100
    if engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.08
        scene.cycles.time_limit = float(os.environ.get("FRAME_TIME_LIMIT", "0"))
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.max_bounces = 5
        scene.cycles.transmission_bounces = 4
        scene.cycles.volume_bounces = 1
        scene.cycles.caustics_reflective = False
        scene.cycles.caustics_refractive = False
        scene.cycles.blur_glossy = 1.0
    else:
        scene.eevee.taa_render_samples = max(16, samples)
        try:
            scene.eevee.use_raytracing = True
        except AttributeError:
            pass
    scene.render.filepath = os.path.join(out_dir, "frame_")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    ap.add_argument("--quality", default="preview", choices=["preview", "medium", "final"])
    ap.add_argument("--frame", type=int, default=None, help="render a single frame")
    ap.add_argument("--animation", action="store_true")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--frames", type=int, default=192)
    ap.add_argument("--out", default="renders")
    ap.add_argument("--save-blend", default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    scene = build(frame_end=args.frames)
    configure_render(scene, args.engine, args.quality, args.out)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    if args.frame is not None:
        scene.frame_set(args.frame)
        scene.render.filepath = os.path.join(args.out, f"frame_{args.frame:04d}.png")
        bpy.ops.render.render(write_still=True)
    elif args.animation:
        if args.start: scene.frame_start = args.start
        if args.end: scene.frame_end = args.end
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
