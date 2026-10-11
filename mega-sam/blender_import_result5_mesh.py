import bpy
import numpy as np
import mathutils
import os

# --- CONFIGURATION ---
file_path = r"D:\AI\MegaSam\mega-sam\outputs_cvd\swing_sgd_cvd_hr.npz"
pixel_step = 2      # Use every 2nd pixel (1 = full res, higher = faster/lighter)
frame_step = 1      
# ---------------------

def load_cvd_data(path):
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    return {'images': data['images'], 'depths': data['depths'], 'intrinsic': data['intrinsic'], 'cam_c2w': data['cam_c2w']}

world_transform = mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')

def create_mesh_for_frame(data, f, collection):
    depth = data['depths'][f]
    image = data['images'][f]
    K = data['intrinsic']
    c2w = mathutils.Matrix(data['cam_c2w'][f])
    
    H, W = depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    # 1. Generate Grid of Vertices
    u, v = np.meshgrid(np.arange(0, W, pixel_step), np.arange(0, H, pixel_step))
    z = depth[v, u].flatten()
    uu, vv = u.flatten(), v.flatten()
    
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    
    # Transform to World
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
    pts_world = (pts_cam @ (world_transform @ c2w).transposed())[:, :3]
    
    # 2. Generate Faces (triangles)
    h_small, w_small = u.shape
    faces = []
    for r in range(h_small - 1):
        for c in range(w_small - 1):
            idx = r * w_small + c
            # Only create faces if depth is valid (not 0)
            if z[idx] > 0.1 and z[idx+1] > 0.1 and z[idx+w_small] > 0.1:
                faces.append((idx, idx + 1, idx + w_small))
            if z[idx+1] > 0.1 and z[idx+w_small+1] > 0.1 and z[idx+w_small] > 0.1:
                faces.append((idx + 1, idx + w_small + 1, idx + w_small))

    # 3. Create Mesh Object
    mesh = bpy.data.meshes.new(f"Frame_{f:03d}")
    obj = bpy.data.objects.new(f"Frame_{f:03d}", mesh)
    collection.objects.link(obj)
    mesh.from_pydata(pts_world.tolist(), [], faces)
    
    # 4. Add Color
    color_attr = mesh.attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
    colors = image[vv, uu] / 255.0
    rgba = np.concatenate([colors, np.ones((len(colors), 1))], axis=1)
    color_attr.data.foreach_set("color", rgba.flatten())
    
    # 5. Animate Visibility
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert("hide_viewport", frame=f-1)
    obj.keyframe_insert("hide_render", frame=f-1)
    
    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert("hide_viewport", frame=f)
    obj.keyframe_insert("hide_render", frame=f)
    
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert("hide_viewport", frame=f+1)
    obj.keyframe_insert("hide_render", frame=f+1)

    # 6. Material Setup
    mat = bpy.data.materials.get("MeshFrameMat") or bpy.data.materials.new("MeshFrameMat")
    mat.use_nodes = True
    if not mat.node_tree.nodes.get("Attribute"):
        nodes = mat.node_tree.nodes
        attr = nodes.new('ShaderNodeAttribute'); attr.attribute_name = "Color"
        bsdf = nodes.get("Principled BSDF")
        mat.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
    obj.data.materials.append(mat)
    
    return obj

# Main Execution
print("Loading data...")
data = load_cvd_data(file_path)
num_frames = data['depths'].shape[0]

# Cleanup
if "MegaSam_Meshes" in bpy.data.collections:
    col = bpy.data.collections["MegaSam_Meshes"]
    for o in col.objects: bpy.data.objects.remove(o, do_unlink=True)
else:
    col = bpy.data.collections.new("MegaSam_Meshes")
    bpy.context.scene.collection.children.link(col)

print(f"Creating meshes for {num_frames} frames...")
for f in range(0, num_frames, frame_step):
    print(f"Processing Frame {f}")
    create_mesh_for_frame(data, f, col)

print("Done! Press Play to see the animated mesh.")
