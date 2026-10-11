import bpy
import numpy as np
import mathutils
import os

# --- CONFIGURATION ---
file_path = r"D:\AI\MegaSam\mega-sam\outputs_cvd\swing_sgd_cvd_hr.npz"
frame_step = 1      
pixel_step = 6      # Higher for animation to keep it smooth
FILTER_MODE = "STATIC" # "STATIC", "DYNAMIC", or "ALL"
motion_threshold = 0.1
ANIMATE = True      # Set to True for an animated point cloud
# ---------------------

def load_cvd_data(path):
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    return {
        'images': data['images'], 'depths': data['depths'], 
        'uncertainty': data.get('uncertainty', None),
        'intrinsic': data['intrinsic'], 'cam_c2w': data['cam_c2w']
    }

world_transform = mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')

def setup_camera(data):
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

def create_animated_point_cloud(data, cam_obj):
    depths, images, K, c2ws = data['depths'], data['images'], data['intrinsic'], data['cam_c2w']
    uncertainties = data['uncertainty']
    if uncertainties is None:
        print("WARNING: 'uncertainty' data not found in .npz. Filtering (Static/Dynamic Only) will not work.")
    num_frames, H, W = depths.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    all_pts, all_cols, all_f_ids = [], [], []
    
    for f in range(0, num_frames, frame_step):
        depth, image, c2w = depths[f], images[f], mathutils.Matrix(c2ws[f])
        u, v = np.meshgrid(np.arange(0, W, pixel_step), np.arange(0, H, pixel_step))
        z = depth[v, u].flatten()
        mask = (z > 0.1) & (z < 50.0)
        
        if uncertainties is not None and FILTER_MODE != "ALL":
            vals = uncertainties[f][v, u].flatten()
            mask &= (vals < motion_threshold) if FILTER_MODE == "STATIC" else (vals > motion_threshold)
        
        if not np.any(mask): continue
        
        z, uu, vv = z[mask], u.flatten()[mask], v.flatten()[mask]
        pts_cam = np.stack([(uu-cx)*z/fx, (vv-cy)*z/fy, z, np.ones_like(z)], axis=-1)
        all_pts.append((pts_cam @ (world_transform @ c2w).transposed())[:, :3])
        all_cols.append(image[vv, uu] / 255.0)
        all_f_ids.append(np.full(len(z), f, dtype=np.float32))

    mesh = bpy.data.meshes.new(f"PC_{FILTER_MODE}")
    obj = bpy.data.objects.new(f"PC_{FILTER_MODE}", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(np.concatenate(all_pts).tolist(), [], [])
    
    # Add Color Attribute
    color_attr = mesh.attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
    color_attr.data.foreach_set("color", np.concatenate([np.concatenate(all_cols), np.ones((len(mesh.vertices), 1))], axis=1).flatten())
    
    # Add Frame ID Attribute
    if ANIMATE:
        frame_attr = mesh.attributes.new(name="FrameID", type='FLOAT', domain='POINT')
        frame_attr.data.foreach_set("value", np.concatenate(all_f_ids))
    
    setup_point_visualization(obj, cam_obj)
    return obj

def setup_point_visualization(obj, cam_obj):
    mat_name = f"PointColorMat_{FILTER_MODE}"
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
    
    # --- DISTANCE BASED SCALING ---
    # 1. Get Camera Info
    obj_info = group.nodes.new('GeometryNodeObjectInfo')
    obj_info.inputs['Object'].default_value = cam_obj
    
    # 2. Get Point Position
    pos = group.nodes.new('GeometryNodeInputPosition')
    
    # 3. Calculate Distance
    dist = group.nodes.new('ShaderNodeVectorMath')
    dist.operation = 'DISTANCE'
    group.links.new(obj_info.outputs['Location'], dist.inputs[0])
    group.links.new(pos.outputs[0], dist.inputs[1])
    
    # 4. Map Distance to Radius (Near smaller, Far bigger)
    map_range = group.nodes.new('ShaderNodeMapRange')
    map_range.inputs['From Min'].default_value = 0.5   # Meters (Near distance)
    map_range.inputs['From Max'].default_value = 10.0  # Meters (Far distance)
    map_range.inputs['To Min'].default_value = 0.005  # Radius Near
    map_range.inputs['To Max'].default_value = 0.04   # Radius Far (Increased for visibility)
    map_range.interpolation_type = 'LINEAR'
    
    group.links.new(dist.outputs['Value'], map_range.inputs['Value'])
    group.links.new(map_range.outputs[0], m2p.inputs['Radius'])
    
    set_mat = group.nodes.new('GeometryNodeSetMaterial')
    set_mat.inputs['Material'].default_value = mat
    
    final_geo = m2p.outputs[0]
    
    # --- ANIMATION LOGIC: Filter by FrameID ---
    if ANIMATE:
        sep_geo = group.nodes.new('GeometryNodeSeparateGeometry')
        named_attr = group.nodes.new('GeometryNodeInputNamedAttribute')
        named_attr.data_type = 'FLOAT'
        named_attr.inputs[0].default_value = "FrameID"
        
        scene_time = group.nodes.new('GeometryNodeInputSceneTime')
        compare = group.nodes.new('FunctionNodeCompare')
        compare.data_type = 'FLOAT'
        compare.operation = 'EQUAL'
        
        group.links.new(named_attr.outputs[0], compare.inputs[0])
        group.links.new(scene_time.outputs[1], compare.inputs[1]) # Scene Frame
        
        group.links.new(in_node.outputs[0], sep_geo.inputs[0])
        group.links.new(compare.outputs[0], sep_geo.inputs['Selection'])
        group.links.new(sep_geo.outputs['Selection'], m2p.inputs[0])

    group.links.new(final_geo, set_mat.inputs[0])
    group.links.new(set_mat.outputs[0], out_node.inputs[0])

# Main
bpy.ops.object.select_all(action='DESELECT')
for n in [f"PC_{FILTER_MODE}", "CVD_Camera"]:
    if n in bpy.data.objects: bpy.data.objects[n].select_set(True)
bpy.ops.object.delete()

data = load_cvd_data(file_path)
cam_obj = setup_camera(data)
create_animated_point_cloud(data, cam_obj)
