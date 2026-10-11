import bpy
import numpy as np
import mathutils
import os
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# Path to the NEW .npz file to import
file_path = r"D:\AI\MegaSam\mega-sam\outputs_cvd\zepp_run0_sgd_cvd_hr.npz"

# Frame on the Blender timeline where this sequence should START
START_FRAME_OFFSET = 0

# ── ALIGNMENT ────────────────────────────────────────────────
# Set ALIGN_TO_OBJECT_NAME to the CAMERA name from the previous import.
# Leave "" to import without any alignment.
ALIGN_TO_OBJECT_NAME = ""

# (RECOMMENDED) Path to the PREVIOUS sequence's .npz file.
# When provided, the script automatically estimates the scale by comparing
# the median depth at the overlap point (last frame of previous / first of new).
# This is the most accurate way to match two sequences.
ALIGN_TO_NPZ_FILE = ""

# ── MANUAL OVERRIDES ──────────────────────────────────────────
# If you need to fine-tune after automatic alignment:
MANUAL_SCALE         = 1.0             # multiplier applied on top of auto-scale
MANUAL_LOC_OFFSET    = (0.0, 0.0, 0.0) # (X, Y, Z) in Blender units
MANUAL_ROT_OFFSET    = (0.0, 0.0, 0.0) # (X, Y, Z) Euler degrees

# ── DEBUG ────────────────────────────────────────────────────
# Set to True to place Arrow-empties at both alignment anchor points
DEBUG_ALIGNMENT = False

# ── RENDERING SETTINGS ───────────────────────────────────────
frame_step       = 1        # Process every Nth frame of the data
pixel_step       = 6        # Sample every Nth pixel (higher = fewer points, faster)
FILTER_MODE      = "ALL"    # "STATIC", "DYNAMIC", or "ALL"
motion_threshold = 0.1      # Uncertainty threshold for STATIC / DYNAMIC filter
ANIMATE          = True     # Animate the point cloud frame-by-frame

# ============================================================
# INTERNALS – do not edit below unless you know what you're doing
# ============================================================

# OpenCV → Blender world axis flip (-90° around X)
WORLD_XFORM = mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')
# OpenCV camera local → Blender camera local (flip Y and Z)
CV_TO_BL_LOCAL = mathutils.Matrix(((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1)))


def load_cvd_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    d = np.load(path)
    return {
        'images':      d['images'],
        'depths':      d['depths'],
        'uncertainty': d.get('uncertainty', None),
        'intrinsic':   d['intrinsic'],
        'cam_c2w':     d['cam_c2w'],
    }


def _raw_cam_pose(c2w_np, scale):
    """Return the Blender-space camera 4x4 matrix for a single frame,
    with translation scaled by *scale*."""
    m = mathutils.Matrix(c2w_np)
    m.translation *= scale
    return WORLD_XFORM @ m @ CV_TO_BL_LOCAL


def _create_debug_empty(name, matrix):
    if not DEBUG_ALIGNMENT:
        return
    e = bpy.data.objects.new(name, None)
    e.empty_display_size  = 0.25
    e.empty_display_type  = 'ARROWS'
    e.matrix_world        = matrix
    bpy.context.collection.objects.link(e)


def estimate_scale(new_data, ref_npz_path):
    """Estimate scale by comparing median depths at the overlap point.

    The last frame of the reference sequence and the first frame of the new
    sequence should be (approximately) the same viewpoint.  The ratio of
    median depths gives the scale factor needed to bring the new sequence
    into the same metric space as the reference.
    """
    ref_d = np.load(ref_npz_path)
    ref_depths = ref_d['depths']
    ref_frame  = ref_depths[-1]          # last frame of reference
    new_frame  = new_data['depths'][0]   # first frame of new sequence

    def median_valid(depth_img):
        v = depth_img[(depth_img > 0.05) & (depth_img < 200.0)]
        return float(np.median(v)) if len(v) > 0 else None

    ref_med = median_valid(ref_frame)
    new_med = median_valid(new_frame)

    if ref_med is None or new_med is None or new_med < 1e-6:
        print("WARNING: Could not estimate scale from depth – falling back to MANUAL_SCALE")
        return MANUAL_SCALE

    auto_scale = (ref_med / new_med)
    print(f"  Depth-based scale  ref={ref_med:.4f}  new={new_med:.4f}  auto={auto_scale:.4f}")
    return auto_scale * MANUAL_SCALE


def get_alignment(new_data):
    """Returns (alignment_4x4, scale_float).

    alignment_4x4 is an additional world-space transform applied on top of
    the regular WORLD_XFORM so that frame-0 of the new sequence starts
    exactly where the reference camera ended.
    scale_float is the uniform scale to apply to translations and depths.
    """
    scale = MANUAL_SCALE

    if not ALIGN_TO_OBJECT_NAME:
        return mathutils.Matrix.Identity(4), scale

    if ALIGN_TO_OBJECT_NAME not in bpy.data.objects:
        print(f"WARNING: '{ALIGN_TO_OBJECT_NAME}' not found in scene – skipping alignment")
        return mathutils.Matrix.Identity(4), scale

    ref_obj = bpy.data.objects[ALIGN_TO_OBJECT_NAME]

    # ── Find the last keyframe of the reference camera ──────────────
    ref_frame = 0
    if ref_obj.animation_data and ref_obj.animation_data.action:
        ref_frame = int(ref_obj.animation_data.action.frame_range[1])
    print(f"  Reference object '{ALIGN_TO_OBJECT_NAME}'  last keyframe = {ref_frame}")

    # ── Auto scale from previous npz ────────────────────────────────
    if ALIGN_TO_NPZ_FILE and os.path.exists(ALIGN_TO_NPZ_FILE):
        print("  Estimating scale from previous .npz depth values …")
        scale = estimate_scale(new_data, ALIGN_TO_NPZ_FILE)
    print(f"  Final scale factor: {scale:.6f}")

    # ── Reference pose (last frame of previous camera) ───────────────
    saved = bpy.context.scene.frame_current
    bpy.context.scene.frame_set(ref_frame)
    m_ref = ref_obj.matrix_world.copy()
    bpy.context.scene.frame_set(saved)
    _create_debug_empty("DEBUG_Ref_End", m_ref)

    # ── New sequence pose at frame 0 (with scale applied) ────────────
    m_new_0 = _raw_cam_pose(new_data['cam_c2w'][0], scale)
    _create_debug_empty("DEBUG_New_Start", m_new_0)

    # ── Alignment: T such that  T @ m_new_0 = m_ref  →  T = m_ref @ m_new_0⁻¹ ──
    alignment = m_ref @ m_new_0.inverted()

    # ── Optional manual fine-tune ────────────────────────────────────
    if any(v != 0 for v in MANUAL_LOC_OFFSET) or any(v != 0 for v in MANUAL_ROT_OFFSET):
        loc_mat = mathutils.Matrix.Translation(MANUAL_LOC_OFFSET)
        rot_mat = mathutils.Euler(
            [np.radians(a) for a in MANUAL_ROT_OFFSET], 'XYZ'
        ).to_matrix().to_4x4()
        alignment = alignment @ loc_mat @ rot_mat
        print(f"  Manual fine-tune applied: loc={MANUAL_LOC_OFFSET}  rot={MANUAL_ROT_OFFSET}")

    return alignment, scale


def setup_camera(data, prefix, offset, alignment, scale):
    cam_name = f"{prefix}_Camera"
    cam_data = bpy.data.cameras.new(cam_name)
    cam_obj  = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.collection.objects.link(cam_obj)

    K, images         = data['intrinsic'], data['images']
    num_frames, H, W, _ = images.shape

    bpy.context.scene.render.resolution_x = W
    bpy.context.scene.render.resolution_y = H

    if bpy.context.scene.frame_start > offset:
        bpy.context.scene.frame_start = offset
    if bpy.context.scene.frame_end < (offset + num_frames - 1):
        bpy.context.scene.frame_end = offset + num_frames - 1

    cam_data.lens_unit = 'FOV'
    cam_data.angle     = 2.0 * np.arctan(W / (2.0 * K[0, 0]))

    print(f"  Keyframing '{cam_name}' frames {offset} → {offset + num_frames - 1} …")
    for f in range(num_frames):
        cam_obj.matrix_world = alignment @ _raw_cam_pose(data['cam_c2w'][f], scale)
        cam_obj.keyframe_insert(data_path="location",       frame=f + offset)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f + offset)

    return cam_obj


def create_animated_point_cloud(data, prefix, offset, alignment, scale):
    depths, images  = data['depths'], data['images']
    K, c2ws         = data['intrinsic'], data['cam_c2w']
    uncertainties   = data['uncertainty']
    num_frames, H, W = depths.shape
    fx, fy, cx, cy  = K[0,0], K[1,1], K[0,2], K[1,2]

    all_pts, all_cols, all_fids = [], [], []

    print(f"  Building point cloud (scale={scale:.4f}) …")
    for f in range(0, num_frames, frame_step):
        depth  = depths[f]
        image  = images[f]
        # Camera-to-world in Blender space WITH scale on translation
        m_c2w = WORLD_XFORM @ mathutils.Matrix(c2ws[f])
        m_c2w.translation *= scale
        # Combine with alignment
        full_mat = alignment @ m_c2w

        u, v = np.meshgrid(np.arange(0, W, pixel_step), np.arange(0, H, pixel_step))
        # Scale depth values to match the metric space of the reference
        z = depth[v, u].flatten() * scale

        mask = (z > 0.05 * scale) & (z < 200.0 * scale)
        if uncertainties is not None and FILTER_MODE != "ALL":
            vals  = uncertainties[f][v, u].flatten()
            mask &= (vals < motion_threshold) if FILTER_MODE == "STATIC" else (vals > motion_threshold)

        if not np.any(mask):
            continue

        z, uu, vv = z[mask], u.flatten()[mask], v.flatten()[mask]

        # Back-project to camera space (z already scaled)
        pts_cam = np.stack(
            [(uu - cx) * z / fx, (vv - cy) * z / fy, z, np.ones_like(z)],
            axis=-1
        )

        # Transform to aligned Blender world space
        # Note: NO cv_to_blender_local here – that only applies to camera orientation
        all_pts.append((pts_cam @ full_mat.transposed().to_3x3_translation())[:, :3]
                       if False else
                       np.array([(full_mat @ mathutils.Vector(p[:3])) for p in pts_cam]))

        all_cols.append(image[vv, uu] / 255.0)
        all_fids.append(np.full(len(z), f + offset, dtype=np.float32))

    if not all_pts:
        print("  ERROR: No valid points generated!")
        return None

    pc_name = f"{prefix}_PC_{FILTER_MODE}"
    all_pts_np = np.concatenate(all_pts)
    mesh = bpy.data.meshes.new(pc_name)
    obj  = bpy.data.objects.new(pc_name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(all_pts_np.tolist(), [], [])

    col_attr = mesh.attributes.new("Color", 'FLOAT_COLOR', 'POINT')
    all_cols_np = np.concatenate(all_cols)
    rgba = np.concatenate([all_cols_np, np.ones((len(mesh.vertices), 1))], axis=1)
    col_attr.data.foreach_set("color", rgba.flatten())

    if ANIMATE:
        fid_attr = mesh.attributes.new("FrameID", 'FLOAT', 'POINT')
        fid_attr.data.foreach_set("value", np.concatenate(all_fids))

    _setup_geo_nodes(obj, prefix)
    return obj


def _setup_geo_nodes(obj, prefix):
    mat_name = f"{prefix}_Mat_{FILTER_MODE}"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    attr = nodes.new('ShaderNodeAttribute')
    attr.attribute_name = "Color"
    mat.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'],  out.inputs['Surface'])

    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    mod        = obj.modifiers.new("PointVis", 'NODES')
    grp_name   = f"{prefix}_Nodes_{FILTER_MODE}"
    grp = (bpy.data.node_groups.get(grp_name) or
           bpy.data.node_groups.new(grp_name, 'GeometryNodeTree'))
    mod.node_group = grp
    grp.nodes.clear()

    g_in  = grp.nodes.new('NodeGroupInput')
    g_out = grp.nodes.new('NodeGroupOutput')
    if not grp.interface.items_tree:
        grp.interface.new_socket("Geometry", in_out='INPUT',  socket_type='NodeSocketGeometry')
        grp.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

    m2p = grp.nodes.new('GeometryNodeMeshToPoints')
    m2p.inputs['Radius'].default_value = 0.01

    set_mat = grp.nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs['Material'].default_value = mat

    if ANIMATE:
        sep   = grp.nodes.new('GeometryNodeSeparateGeometry')
        nattr = grp.nodes.new('GeometryNodeInputNamedAttribute')
        nattr.data_type = 'FLOAT'
        nattr.inputs[0].default_value = "FrameID"

        stime = grp.nodes.new('GeometryNodeInputSceneTime')
        cmp   = grp.nodes.new('FunctionNodeCompare')
        cmp.data_type = 'FLOAT'
        cmp.operation = 'EQUAL'

        grp.links.new(nattr.outputs[0],    cmp.inputs[0])
        grp.links.new(stime.outputs[1],    cmp.inputs[1])   # "Frame"
        grp.links.new(g_in.outputs[0],     sep.inputs[0])
        grp.links.new(cmp.outputs[0],      sep.inputs['Selection'])
        grp.links.new(sep.outputs['Selection'], m2p.inputs[0])
    else:
        grp.links.new(g_in.outputs[0], m2p.inputs[0])

    grp.links.new(m2p.outputs[0],   set_mat.inputs[0])
    grp.links.new(set_mat.outputs[0], g_out.inputs[0])


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    prefix = Path(file_path).stem
    print(f"\n{'='*60}")
    print(f"Importing: {prefix}")
    print(f"  START_FRAME_OFFSET = {START_FRAME_OFFSET}")
    print(f"  ALIGN_TO           = '{ALIGN_TO_OBJECT_NAME}'")
    print(f"  ALIGN_TO_NPZ       = '{ALIGN_TO_NPZ_FILE}'")

    # Remove previous objects with same prefix (and debug empties)
    bpy.ops.object.select_all(action='DESELECT')
    for o in list(bpy.data.objects):
        if o.name.startswith(prefix) or (DEBUG_ALIGNMENT and o.name.startswith("DEBUG_")):
            o.select_set(True)
    bpy.ops.object.delete()

    data = load_cvd_data(file_path)
    alignment, scale = get_alignment(data)

    setup_camera(data, prefix, START_FRAME_OFFSET, alignment, scale)
    create_animated_point_cloud(data, prefix, START_FRAME_OFFSET, alignment, scale)

    print(f"Import complete: {prefix}")
    print('='*60)
