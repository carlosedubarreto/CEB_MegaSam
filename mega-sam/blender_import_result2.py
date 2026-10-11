import bpy
import numpy as np
import mathutils
import os

# --- CONFIGURATION ---
file_path = r"D:\AI\MegaSam\mega-sam\outputs_cvd\swing_sgd_cvd_hr.npz"
frame_step = 1      # Set to 1 for moving objects to see the full motion trail
pixel_step = 4      
FILTER_MODE = "DYNAMIC" # Options: "STATIC" or "DYNAMIC"
motion_threshold = 0.15 # For DYNAMIC, higher values ensure you only get clear motion
# ---------------------

def load_cvd_data(path):
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    return {
        'images': data['images'], 
        'depths': data['depths'], 
        'uncertainty': data.get('uncertainty', None),
        'intrinsic': data['intrinsic'], 
        'cam_c2w': data['cam_c2w']
    }

world_transform = mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')

def setup_camera(data):
    # (Existing camera setup code...)
    cam_data = bpy.data.cameras.new("CVD_Camera")
    cam_obj = bpy.data.objects.new("CVD_Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    K, images = data['intrinsic'], data['images']
    num_frames, H, W, _ = images.shape
    bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = W, H
    bpy.context.scene.frame_start, bpy.context.scene.frame_end = 0, num_frames - 1
    cam_data.lens_unit, cam_data.angle = 'FOV', 2 * np.arctan(W / (2 * K[0, 0]))
    cv_to_blender_local = mathutils.Matrix(((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1)))
    for f in range(num_frames):
        c2w = mathutils.Matrix(data['cam_c2w'][f])
        cam_obj.matrix_world = world_transform @ c2w @ cv_to_blender_local
        cam_obj.keyframe_insert(data_path="location", frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)
    return cam_obj

def create_global_point_cloud(data):
    depths, images, K, c2ws = data['depths'], data['images'], data['intrinsic'], data['cam_c2w']
    uncertainties = data['uncertainty']
    num_frames, H, W = depths.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    if uncertainties is None:
        print("WARNING: 'uncertainty' data not found. Filtering will not work.")
    
    all_pts, all_cols = [], []
    for f in range(0, num_frames, frame_step):
        depth, image, c2w = depths[f], images[f], mathutils.Matrix(c2ws[f])
        u, v = np.meshgrid(np.arange(0, W, pixel_step), np.arange(0, H, pixel_step))
        z = depth[v, u].flatten()
        mask = (z > 0.1) & (z < 50.0)
        
        # --- INVERTED MOTION FILTERING ---
        if uncertainties is not None:
            vals = uncertainties[f][v, u].flatten()
            if FILTER_MODE == "STATIC":
                motion_mask = vals < motion_threshold
            else: # DYNAMIC
                motion_mask = vals > motion_threshold
            mask &= motion_mask
        
        if not np.any(mask): continue
        
        z, uu, vv = z[mask], u.flatten()[mask], v.flatten()[mask]
        pts_cam = np.stack([(uu-cx)*z/fx, (vv-cy)*z/fy, z, np.ones_like(z)], axis=-1)
        all_pts.append((pts_cam @ (world_transform @ c2w).transposed())[:, :3])
        all_cols.append(image[vv, uu] / 255.0)

    if not all_pts:
        print("No points found with the current threshold!")
        return None

    mesh = bpy.data.meshes.new(f"PC_{FILTER_MODE}")
    obj = bpy.data.objects.new(f"PC_{FILTER_MODE}", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(np.concatenate(all_pts).tolist(), [], [])
    
    color_attr = mesh.attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
    flat_colors = np.concatenate(all_cols)
    rgba_colors = np.concatenate([flat_colors, np.ones((len(flat_colors), 1))], axis=1)
    color_attr.data.foreach_set("color", rgba_colors.flatten())
    
    setup_point_visualization(obj)
    return obj

def setup_point_visualization(obj):
    mat_name = "PointColorMat_Dynamic" if FILTER_MODE == "DYNAMIC" else "PointColorMat_Static"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial'); bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    attr = nodes.new('ShaderNodeAttribute'); attr.attribute_name = "Color"
    mat.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    if not obj.data.materials: obj.data.materials.append(mat)

    mod = obj.modifiers.new(name="PointVis", type='NODES')
    group = bpy.data.node_groups.new(name=f"PointGroup_{FILTER_MODE}", type='GeometryNodeTree')
    mod.node_group = group
    in_node = group.nodes.new('NodeGroupInput'); out_node = group.nodes.new('NodeGroupOutput')
    group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    m2p = group.nodes.new('GeometryNodeMeshToPoints')
    m2p.inputs['Radius'].default_value = 0.01
    set_mat = group.nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs['Material'].default_value = mat
    group.links.new(in_node.outputs[0], m2p.inputs[0])
    group.links.new(m2p.outputs[0], set_mat.inputs[0])
    group.links.new(set_mat.outputs[0], out_node.inputs[0])

# Cleanup
bpy.ops.object.select_all(action='DESELECT')
for n in [f"PC_{FILTER_MODE}", "CVD_Camera"]:
    if n in bpy.data.objects: bpy.data.objects[n].select_set(True)
bpy.ops.object.delete()

data = load_cvd_data(file_path)
setup_camera(data)
create_global_point_cloud(data)
