import numpy as np
import os
from pathlib import Path

# --- CONFIGURATION ---
input_file = r"D:\AI\MegaSam\mega-sam\outputs_cvd\swing_sgd_cvd_hr.npz"
output_dir = r"D:\AI\MegaSam\mega-sam\outputs_cvd\meshes_ply"
pixel_step = 2      # Use every 2nd pixel (1 = high res, higher = faster)
# ---------------------

def get_world_transform():
    # Equivalent to mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0], # Y becomes Z
        [0, -1, 0, 0], # Z becomes -Y
        [0, 0, 0, 1]
    ])

def save_ply(filename, points, colors, faces):
    """Saves a PLY file with vertex colors."""
    num_verts = len(points)
    num_faces = len(faces)
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices (x y z r g b)
        for i in range(num_verts):
            f.write(f"{points[i,0]} {points[i,1]} {points[i,2]} {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")
            
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def process_frames():
    print(f"Loading {input_file}...")
    data = np.load(input_file)
    
    images = data['images']
    depths = data['depths']
    K = data['intrinsic']
    c2ws = data['cam_c2w']
    num_frames, H, W = depths.shape
    
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    world_rot = get_world_transform()
    
    # Coordinate conversion for camera local (CV to Blender-style)
    cv_to_blender_local = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Starting conversion of {num_frames} frames...")
    
    for f in range(num_frames):
        depth = depths[f]
        image = images[f]
        c2w = c2ws[f]
        
        # 1. Grid of pixels
        u, v = np.meshgrid(np.arange(0, W, pixel_step), np.arange(0, H, pixel_step))
        z = depth[v, u].flatten()
        uu, vv = u.flatten(), v.flatten()
        
        # 2. Camera space points
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)
        
        # 3. Transform to World
        # Apply World Rotation @ Camera2World @ Local Correction
        full_mat = world_rot @ c2w @ cv_to_blender_local
        pts_world = (pts_cam @ full_mat.T)[:, :3]
        
        # 4. Generate Faces
        h_small, w_small = u.shape
        faces = []
        for r in range(h_small - 1):
            for c in range(w_small - 1):
                idx = r * w_small + c
                # Filter out faces with invalid depth
                if z[idx] > 0.1 and z[idx+1] > 0.1 and z[idx+w_small] > 0.1:
                    faces.append((idx, idx + 1, idx + w_small))
                if z[idx+1] > 0.1 and z[idx+w_small+1] > 0.1 and z[idx+w_small] > 0.1:
                    faces.append((idx + 1, idx + w_small + 1, idx + w_small))
        
        # 5. Save PLY
        out_path = os.path.join(output_dir, f"frame_{f:04d}.ply")
        save_ply(out_path, pts_world, image[vv, uu], faces)
        
        if f % 10 == 0:
            print(f"Saved {f}/{num_frames} frames")

    print(f"Success! All frames saved to: {output_dir}")

if __name__ == "__main__":
    process_frames()
