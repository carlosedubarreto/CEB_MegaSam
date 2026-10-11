"""
generate_mesh_from_npz.py

High-quality MoGe-style mesh generation from a MegaSam CVD NPZ file.

Key features (matching MoGe's quality):
  - UV-mapped PNG texture  (vastly better than per-vertex colors in Blender)
  - Smooth vertex normals  (area-weighted from adjacent face normals)
  - Relative edge filtering: threshold * avg_depth  (the core MoGe technique)
  - Zero-depth vertex removal
  - Optional Gaussian depth smoothing to reduce noise
  - Downscale support (INTER_AREA) as well as upscale (INTER_CUBIC)
  - Back-face culling (removes camera-facing reversed triangles)
  - Optional 180° rotation around the X axis (--rotate_x)

Usage:
  python generate_mesh_from_npz.py --npz scene.npz --frame 0 --threshold 0.05 --upscale 1.5 --smooth 0.5
  python generate_mesh_from_npz.py --npz scene.npz --output_dir ./meshes
"""

import numpy as np
import argparse
import cv2
from pathlib import Path
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Core mesh generation
# ---------------------------------------------------------------------------

def generate_mesh(
    npz_path,
    frame_idx   = 0,
    threshold   = 0.05,   # relative edge threshold (MoGe default ~0.01-0.05)
    upscale     = 1.0,    # resolution multiplier (1.5-2.0 = more detail)
    smooth      = 0.0,    # gaussian depth smoothing sigma (0 = off)
    cull_back   = True,   # remove triangles facing away from camera
    # rotate_x    = False,  # rotate 180° around the X axis
    output_path = None,
    output_dir  = None,
    fmt         = "obj",  # "obj" or "glb"
    filter_mode = "ALL",  # "STATIC", "DYNAMIC", or "ALL"
    motion_threshold = 0.1,
):
    """
    Generate a textured mesh from one frame of a MegaSam NPZ file.
    
    Args:
        npz_path: Path to .npz file
        frame_idx: Frame index to extract
        threshold: Relative edge threshold
        upscale: Resolution multiplier
        smooth: Gaussian depth smoothing sigma
        cull_back: Whether to remove back-facing triangles
        output_path: Base filename for output (default: npz stem + frame)
        output_dir: Optional destination directory
        fmt: "obj" or "glb"
    """

    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)

    # ------------------------------------------------------------------
    # 1. Load and sanitise
    # ------------------------------------------------------------------
    depth     = data["depths"][frame_idx].astype(np.float32)
    image     = data["images"][frame_idx].astype(np.uint8)   # (H, W, 3) RGB
    intrinsic = data["intrinsic"].astype(np.float64).copy()  # (3, 3)

    depth = np.clip(depth, 0, None)
    depth[~np.isfinite(depth)] = 0.0

    # ------------------------------------------------------------------
    # 1.5 Motion Filtering
    # ------------------------------------------------------------------
    if filter_mode != "ALL":
        uncertainty = data.get("uncertainty")
        if uncertainty is not None:
            print(f"Applying filter: {filter_mode} (threshold={motion_threshold})")
            u_vals = uncertainty[frame_idx]
            if filter_mode == "STATIC":
                mask = u_vals < motion_threshold
            else:
                mask = u_vals > motion_threshold
            depth[~mask] = 0.0
        else:
            print("WARNING: Uncertainty data missing. Skipping motion filter.")

    # ------------------------------------------------------------------
    # 2. Optional Gaussian smoothing
    # ------------------------------------------------------------------
    if smooth > 0:
        print(f"Smoothing depth (sigma={smooth}) ...")
        valid = depth > 0
        depth = gaussian_filter(depth.astype(np.float64), sigma=smooth).astype(np.float32)
        depth[~valid] = 0.0                     # restore invalid pixels

    # ------------------------------------------------------------------
    # 3. Resize / upscale
    # ------------------------------------------------------------------
    orig_h, orig_w = depth.shape
    if upscale != 1.0 and upscale > 0:
        new_w = max(2, int(orig_w * upscale))
        new_h = max(2, int(orig_h * upscale))
        interp = cv2.INTER_CUBIC if upscale > 1.0 else cv2.INTER_AREA
        print(f"Resizing {orig_w}x{orig_h} → {new_w}x{new_h} ...")
        depth = cv2.resize(depth.astype(np.float32), (new_w, new_h), interpolation=interp)
        image = cv2.resize(image,                    (new_w, new_h), interpolation=interp)
        intrinsic[0, 0] *= upscale              # fx
        intrinsic[1, 1] *= upscale              # fy
        intrinsic[0, 2] = (intrinsic[0, 2] + 0.5) * upscale - 0.5   # cx
        intrinsic[1, 2] = (intrinsic[1, 2] + 0.5) * upscale - 0.5   # cy

    H, W = depth.shape
    fx = intrinsic[0, 0];  fy = intrinsic[1, 1]
    cx = intrinsic[0, 2];  cy = intrinsic[1, 2]
    print(f"Working resolution: {W}x{H}  |  fx={fx:.1f}  fy={fy:.1f}")

    # ------------------------------------------------------------------
    # 4. Back-project to 3-D (OpenCV / camera space: X right, Y down, Z forward)
    # ------------------------------------------------------------------
    print("Back-projecting ...")
    uu = np.arange(W, dtype=np.float64)
    vv = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(uu, vv)               # both (H, W)

    z = depth.astype(np.float64)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)         # (H, W, 3)

    vertices = pts.reshape(-1, 3)              # (N, 3)

    # ------------------------------------------------------------------
    # 5. UV coordinates  (pixel centre → [0, 1], V flipped for OBJ)
    # ------------------------------------------------------------------
    u_coords = ((uu + 0.5) / W).ravel()         # (N,)
    v_coords = (1.0 - (vv + 0.5) / H).ravel()  # (N,)  flipped for OBJ/OpenGL
    uvs = np.stack([u_coords, v_coords], axis=1)  # (N, 2)

    # ------------------------------------------------------------------
    # 6. Vectorised triangulation of the regular pixel grid
    # ------------------------------------------------------------------
    print("Triangulating ...")
    idx = np.arange(H * W, dtype=np.int32).reshape(H, W)

    tl = idx[:-1, :-1].ravel()   # top-left
    tr = idx[:-1, 1: ].ravel()   # top-right
    bl = idx[1:,  :-1].ravel()   # bottom-left
    br = idx[1:,  1: ].ravel()   # bottom-right

    # Two CCW triangles per quad when viewed from -Z (camera direction)
    f1 = np.stack([tl, bl, tr], axis=1)   # (Q, 3)
    f2 = np.stack([tr, bl, br], axis=1)   # (Q, 3)
    faces = np.vstack([f1, f2])            # (2Q, 3)

    # ------------------------------------------------------------------
    # 7. Edge-length filter  (MoGe relative threshold)
    # ------------------------------------------------------------------
    print(f"Filtering edges (threshold={threshold}) ...")

    depth_flat = z.ravel()

    # Remove faces that have any vertex with zero depth
    d0 = depth_flat[faces[:, 0]]
    d1 = depth_flat[faces[:, 1]]
    d2 = depth_flat[faces[:, 2]]
    valid_depth = (d0 > 0) & (d1 > 0) & (d2 > 0)

    # Relative edge threshold: longest edge < threshold * avg_depth
    pa = vertices[faces[:, 0]]
    pb = vertices[faces[:, 1]]
    pc = vertices[faces[:, 2]]

    avg_z       = (pa[:, 2] + pb[:, 2] + pc[:, 2]) / 3.0
    dyn_thresh2 = (threshold * np.maximum(avg_z, 1e-6)) ** 2

    max_e2 = np.maximum(
        np.sum((pa - pb) ** 2, axis=1),
        np.maximum(
            np.sum((pb - pc) ** 2, axis=1),
            np.sum((pc - pa) ** 2, axis=1),
        ),
    )

    keep = valid_depth & (max_e2 < dyn_thresh2)

    # ------------------------------------------------------------------
    # 8. Optional back-face culling (remove tris facing +Z = behind camera)
    # ------------------------------------------------------------------
    if cull_back:
        pa2 = pa[keep];  pb2 = pb[keep];  pc2 = pc[keep]
        normals_tmp = np.cross(pb2 - pa2, pc2 - pa2)
        # In camera space the camera looks along +Z, so face normal Z < 0 is visible
        front_facing = normals_tmp[:, 2] < 0
        idxs = np.where(keep)[0][front_facing]
        keep_final = np.zeros(len(faces), dtype=bool)
        keep_final[idxs] = True
        keep = keep_final

    valid_faces = faces[keep]
    print(f"  Kept {len(valid_faces):,} / {len(faces):,} faces "
          f"({100*len(valid_faces)/len(faces):.1f}%)")

    # ------------------------------------------------------------------
    # 9. Transform to World Space (Matches raw NPZ c2w)
    print("Transforming to World Space ...")
    c2w = data["cam_c2w"][frame_idx].astype(np.float64)
    
    # 1. Base world transform from the animation script (-90 on X)
    world_transform = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)
    
    # 2. User-requested additional -90 on X (Total = -180 on X)
    extra_x_rot = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],
        [0, -1,  0,  0],
        [0,  0,  0,  1]
    ], dtype=np.float64)
    
    full_matrix = extra_x_rot @ world_transform @ c2w
    
    verts_hom = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    vertices = (verts_hom @ full_matrix.T)[:, :3]

    # ------------------------------------------------------------------
    # 10. Smooth vertex normals  (area-weighted average of adjacent faces)
    # ------------------------------------------------------------------
    print("Computing smooth normals ...")
    vn = np.zeros_like(vertices)              # (N, 3) accumulator

    pa_v = vertices[valid_faces[:, 0]]
    pb_v = vertices[valid_faces[:, 1]]
    pc_v = vertices[valid_faces[:, 2]]

    fn = np.cross(pb_v - pa_v, pc_v - pa_v)   # (F, 3)  area-weighted already
    for col in range(3):
        np.add.at(vn, valid_faces[:, col], fn)

    nlen = np.linalg.norm(vn, axis=1, keepdims=True)
    vn /= np.maximum(nlen, 1e-12)

    # ------------------------------------------------------------------
    # 11. Save
    # ------------------------------------------------------------------
    # Resolve default output name (suffix will be overridden by save functions)
    if output_path is None:
        output_path = Path(npz_path).stem + f"_f{frame_idx:04d}_mesh"
    
    out_path = Path(output_path)
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_path.name

    output_path = out_path.with_suffix("")  # strip any suffix; each saver adds its own

    if fmt == "glb":
        return _save_glb(output_path, vertices, valid_faces, vn, uvs, image)
    else:
        return _save_obj(output_path, vertices, valid_faces, vn, uvs, image)


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_obj(base: Path, vertices, faces, normals, uvs, image) -> str:
    """Write OBJ + MTL + PNG texture (3 files, works in every tool)."""
    obj_path = base.with_suffix(".obj")
    tex_path = base.with_suffix(".png")
    mtl_path = base.with_suffix(".mtl")

    # Texture PNG (NPZ stores RGB; cv2.imwrite expects BGR)
    cv2.imwrite(str(tex_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # MTL
    with open(mtl_path, "w") as f:
        f.write("newmtl mat0\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write("illum 1\n")
        f.write(f"map_Kd {tex_path.name}\n")

    # OBJ
    print(f"Writing OBJ ({len(vertices):,} verts, {len(faces):,} faces) ...")
    with open(obj_path, "w") as f:
        f.write("# MoGe-style mesh\n")
        f.write(f"mtllib {mtl_path.name}\n\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("\n")
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        f.write("\n")
        for n in normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f.write("\nusemtl mat0\n")
        for fc in faces:
            a, b, c = fc[0]+1, fc[1]+1, fc[2]+1
            f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

    print("\n✓ Done!")
    print(f"  OBJ     : {obj_path}")
    print(f"  Texture : {tex_path}")
    print(f"  Material: {mtl_path}")
    return str(obj_path)


def _save_glb(base: Path, vertices, faces, normals, uvs, image) -> str:
    """
    Write a self-contained GLB (binary glTF) using trimesh + Pillow.
    Install if needed:  pip install trimesh pillow
    """
    try:
        import trimesh
        import trimesh.visual.texture as tvt
        from PIL import Image as PILImage
    except ImportError as e:
        print(f"[ERROR] Missing dependency for GLB export: {e}")
        print("  Install with:  pip install trimesh pillow")
        print("  Falling back to OBJ export.")
        return _save_obj(base, vertices, faces, normals, uvs, image)

    glb_path = base.with_suffix(".glb")

    print(f"Building GLB ({len(vertices):,} verts, {len(faces):,} faces) ...")

    # Build trimesh mesh
    mesh = trimesh.Trimesh(
        vertices       = vertices.astype(np.float32),
        faces          = faces.astype(np.int32),
        vertex_normals = normals.astype(np.float32),
        process        = False,
    )

    # Attach UV-mapped PBR texture
    pil_img  = PILImage.fromarray(image)          # image is already RGB uint8
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture = pil_img,
        metallicFactor   = 0.0,
        roughnessFactor  = 1.0,
    )
    # Use the same UVs as OBJ. Trimesh handles the internal glTF flip (0,0 is top-left)
    # during the export process automatically.
    mesh.visual = tvt.TextureVisuals(uv=uvs.astype(np.float32), material=material)

    mesh.export(str(glb_path))

    print("\n✓ Done!")
    print(f"  GLB: {glb_path}")
    return str(glb_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-quality MoGe-style textured mesh from a MegaSam NPZ frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--npz",       type=str,   required=True)
    parser.add_argument("--frame",     type=int,   default=0,
                        help="Frame index (default 0)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Relative edge threshold. Lower = cleaner gaps, "
                             "higher = more connected. (default 0.05)")
    parser.add_argument("--upscale",   type=float, default=1.0,
                        help="Resolution multiplier. 1.5-2.0 gives much more "
                             "detail. (default 1.0, no change)")
    parser.add_argument("--smooth",    type=float, default=0.0,
                        help="Gaussian depth smoothing sigma. 0.5-1.5 helps "
                             "noisy depth. (default 0 = off)")
    parser.add_argument("--no_cull",   action="store_true",
                        help="Disable back-face culling")
    # parser.add_argument("--rotate_x",  action="store_true",
    #                     help="Rotate the mesh 180° around the X axis "
    #                          "(flips Y and Z: useful to match Blender/glTF up-axis conventions)")
    parser.add_argument("--format",    type=str,   default="obj",
                        choices=["obj", "glb"],
                        help="Output format: 'obj' (default) or 'glb' (requires trimesh + pillow)")
    parser.add_argument("--output",    type=str,   default=None,
                        help="Output file base path (extension added automatically)")
    parser.add_argument("--output_dir", type=str,   default=None,
                        help="Output directory (created if missing)")
    parser.add_argument("--filter_mode", type=str, default="ALL",
                        choices=["STATIC", "DYNAMIC", "ALL"],
                        help="Motion filter mode (default: ALL)")
    parser.add_argument("--motion_threshold", type=float, default=0.1,
                        help="Uncertainty threshold for motion filtering (default: 0.1)")

    args = parser.parse_args()
    generate_mesh(
        npz_path    = args.npz,
        frame_idx   = args.frame,
        threshold   = args.threshold,
        upscale     = args.upscale,
        smooth      = args.smooth,
        cull_back   = not args.no_cull,
        # rotate_x    = args.rotate_x,
        output_path = args.output,
        output_dir  = args.output_dir,
        fmt         = args.format,
        filter_mode = args.filter_mode,
        motion_threshold = args.motion_threshold,
    )
