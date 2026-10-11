#!/usr/bin/env python3
"""
stitch_npz_batches.py

Stitches multiple MegaSam batch NPZ files (e.g. zepp_b000_sgd_cvd_hr.npz …
zepp_b007_sgd_cvd_hr.npz) into a single, unified, correctly aligned NPZ.

Alignment strategy
------------------
For each batch N (N > 0):
  1. SCALE  — Compare the median valid depth of the LAST frame of the
              previous (already-aligned) batch with the FIRST frame of the
              current batch.  The ratio gives the scale factor needed to
              bring the current batch into the same metric space.

  2. ROTATE / TRANSLATE — Compute an alignment transform T in OpenCV
              camera-to-world space such that:
                  T @ c2w_curr[0]_scaled  =  c2w_prev[-1]
              Apply T to every camera in the current batch so the
              trajectory is continuous.

  3. DEPTHS  — Multiply every depth value by the scale factor so the
              point-cloud geometry also matches.

All operations happen entirely in NPZ (OpenCV) coordinate space, so the
result can be loaded by any downstream tool without further conversion.

Usage examples
--------------
# Auto-discover files from a directory given the base scene name:
python stitch_npz_batches.py --input_dir outputs_cvd --scene_name zepp

# Explicit list of files:
python stitch_npz_batches.py \\
    --files outputs_cvd/zepp_b000_sgd_cvd_hr.npz \\
            outputs_cvd/zepp_b001_sgd_cvd_hr.npz \\
            outputs_cvd/zepp_b002_sgd_cvd_hr.npz \\
    --output outputs_cvd/zepp_stitched_sgd_cvd_hr.npz
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def load_npz(path: str) -> dict:
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    d = np.load(path)
    return {
        "images":      d["images"],        # (N, H, W, 3) uint8
        "depths":      d["depths"].astype(np.float32),  # (N, H, W)
        "uncertainty": d["uncertainty"] if "uncertainty" in d else None,  # (N, H, W) or None
        "intrinsic":   d["intrinsic"],     # (3, 3)
        "cam_c2w":     d["cam_c2w"].astype(np.float64), # (N, 4, 4)
    }


# ---------------------------------------------------------------------------
# Scale estimation
# ---------------------------------------------------------------------------

def estimate_scale(prev_depth_frame: np.ndarray,
                   curr_depth_frame: np.ndarray,
                   depth_min: float = 0.05,
                   depth_max: float = 200.0) -> float:
    """Return scale factor = prev_median_depth / curr_median_depth.

    Both frames should be at the 'seam' (last of prev, first of curr).
    Pixels outside [depth_min, depth_max] are treated as invalid.
    Returns 1.0 if estimation is not reliable.
    """
    def valid_median(d):
        v = d[(d > depth_min) & (d < depth_max)].ravel()
        if len(v) < 100:
            return None
        return float(np.median(v))

    prev_med = valid_median(prev_depth_frame)
    curr_med = valid_median(curr_depth_frame)

    if prev_med is None or curr_med is None or curr_med < 1e-6:
        print("    WARNING: Could not estimate scale reliably – using 1.0")
        return 1.0

    scale = prev_med / curr_med
    print(f"    Depth scale:  prev_median={prev_med:.4f}  "
          f"curr_median={curr_med:.4f}  → scale={scale:.6f}")
    return scale


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_batch_to_prev(curr_data: dict,
                         prev_c2w_last: np.ndarray,
                         scale: float) -> dict:
    """Return a copy of curr_data with cameras and depths aligned to prev.

    Steps
    -----
    1. Scale all camera translations and all depth values by *scale*.
    2. Compute alignment transform T (4×4, OpenCV space) such that
           T @ c2w_scaled[0]  =  prev_c2w_last
    3. Apply T to every camera in the batch.

    The intrinsic and image data are left unchanged.
    """
    c2w    = curr_data["cam_c2w"].copy()   # (N, 4, 4)
    depths = curr_data["depths"].copy()    # (N, H, W)

    # --- Step 1: scale translations and depths --------------------------------
    c2w[:, :3, 3] *= scale          # only the translation column
    depths         *= scale

    # --- Step 2: alignment transform ------------------------------------------
    # T such that T @ c2w_scaled[0] = prev_c2w_last
    # => T = prev_c2w_last @ inv(c2w_scaled[0])
    T_align = prev_c2w_last @ np.linalg.inv(c2w[0])

    # --- Step 3: apply T to all cameras ---------------------------------------
    # Correct batch matrix multiply: aligned_c2w[n] = T_align @ c2w[n]
    # Using einsum to be unambiguous: 'ij,njk->nik'
    #   i,j = T_align row/col   |   n = batch   |   j,k = c2w row/col
    # NOTE: the previous transpose trick was computing c2w[n] @ T_align.T
    # (wrong!), causing cameras to jump at every seam.
    aligned_c2w = np.einsum('ij,njk->nik', T_align, c2w)

    result = dict(curr_data)
    result["cam_c2w"] = aligned_c2w
    result["depths"]  = depths

    # Carry aligned uncertainty if present (geometry changes with scale in 3-D,
    # but the per-pixel motion mask value itself is unchanged)
    return result


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def stitch_batches(npz_files: list, output_path: str) -> str:
    """Stitch *npz_files* in order and save to *output_path*.

    Returns the resolved output path as a string.
    """
    if len(npz_files) < 2:
        raise ValueError(f"Need at least 2 files to stitch, got {len(npz_files)}")

    print(f"\n{'='*60}")
    print(f"Stitching {len(npz_files)} batches → {output_path}")
    print(f"{'='*60}")
    for i, f in enumerate(npz_files):
        print(f"  [{i:03d}] {Path(f).name}")

    # ------------------------------------------------------------------
    # Batch 0: reference – no transformation
    # ------------------------------------------------------------------
    print(f"\n[0] Loading reference batch: {Path(npz_files[0]).name}")
    prev_data = load_npz(npz_files[0])
    n0 = prev_data["images"].shape[0]
    print(f"    Frames: {n0}")

    # Accumulate lists
    all_images  = [prev_data["images"]]
    all_depths  = [prev_data["depths"]]
    all_c2ws    = [prev_data["cam_c2w"]]
    intrinsic   = prev_data["intrinsic"]
    has_uncert  = prev_data["uncertainty"] is not None
    all_uncerts = [prev_data["uncertainty"]] if has_uncert else None

    # ------------------------------------------------------------------
    # Batches 1 … N-1
    # ------------------------------------------------------------------
    for i in range(1, len(npz_files)):
        print(f"\n[{i}] Aligning batch: {Path(npz_files[i]).name}")
        curr_data = load_npz(npz_files[i])
        n_curr = curr_data["images"].shape[0]
        print(f"    Frames: {n_curr}")

        # Scale from depth at the seam point
        scale = estimate_scale(
            prev_data["depths"][-1],  # last frame of (already aligned) prev
            curr_data["depths"][0],   # first frame of current (unaligned)
        )

        # Last camera pose from the accumulated sequence
        prev_c2w_last = all_c2ws[-1][-1]

        # Align
        aligned = align_batch_to_prev(curr_data, prev_c2w_last, scale)

        # Accumulate
        all_images.append(aligned["images"])
        all_depths.append(aligned["depths"])
        all_c2ws.append(aligned["cam_c2w"])
        if has_uncert and aligned["uncertainty"] is not None:
            all_uncerts.append(aligned["uncertainty"])

        # Use the aligned data as prev for the next iteration
        # (depth comparison needs depths in the same metric space)
        prev_data = aligned

    # ------------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------------
    print(f"\nConcatenating all batches …")
    merged_images = np.concatenate(all_images, axis=0)
    merged_depths = np.concatenate(all_depths, axis=0)
    merged_c2ws   = np.concatenate(all_c2ws,   axis=0)
    total_frames  = merged_images.shape[0]
    print(f"    Total frames: {total_frames}")
    print(f"    images shape: {merged_images.shape}")
    print(f"    depths shape: {merged_depths.shape}")
    print(f"    cam_c2w shape: {merged_c2ws.shape}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_dict = {
        "images":    merged_images,
        "depths":    merged_depths,
        "intrinsic": intrinsic,
        "cam_c2w":   merged_c2ws,
    }
    if has_uncert and all_uncerts:
        save_dict["uncertainty"] = np.concatenate(all_uncerts, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving → {output_path} …")
    np.savez_compressed(str(output_path), **save_dict)
    print(f"Done!")
    print(f"{'='*60}\n")
    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_batch_files(input_dir: str, scene_name: str, suffix: str) -> list:
    """Auto-discover sorted batch files matching <scene_name>_b*<suffix>."""
    pattern = os.path.join(input_dir, f"{scene_name}_b*{suffix}")
    files = sorted(glob.glob(pattern))
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Stitch MegaSam batch NPZ files into a single aligned sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Auto-discovery mode ---
    parser.add_argument(
        "--input_dir", "-i", type=str, default="outputs_cvd",
        help="Directory containing the batch NPZ files (default: outputs_cvd)")
    parser.add_argument(
        "--scene_name", "-s", type=str, default=None,
        help="Base scene name, e.g. 'zepp'. Files like zepp_b000_sgd_cvd_hr.npz "
             "are discovered automatically.")
    parser.add_argument(
        "--suffix", type=str, default="_sgd_cvd_hr.npz",
        help="File suffix to match (default: _sgd_cvd_hr.npz)")

    # --- Explicit file list mode ---
    parser.add_argument(
        "--files", nargs="+", type=str, default=None,
        help="Explicit ordered list of NPZ files. Overrides --scene_name discovery.")

    # --- Output ---
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output NPZ path. Defaults to <input_dir>/<scene_name>_stitched<suffix>")

    args = parser.parse_args()

    # Resolve file list
    if args.files:
        npz_files = args.files
    elif args.scene_name:
        npz_files = find_batch_files(args.input_dir, args.scene_name, args.suffix)
        if not npz_files:
            print(f"[ERROR] No files found matching: "
                  f"{args.input_dir}/{args.scene_name}_b*{args.suffix}")
            return
    else:
        print("[ERROR] Provide either --files or --scene_name.")
        return

    print(f"Found {len(npz_files)} batch file(s) to stitch.")

    if len(npz_files) < 2:
        print("[ERROR] Need at least 2 files to stitch.")
        return

    # Resolve output path
    if args.output:
        output_path = args.output
    elif args.scene_name:
        stem = f"{args.scene_name}_stitched{args.suffix}"
        output_path = os.path.join(args.input_dir, stem)
    else:
        output_path = "stitched_output.npz"

    stitch_batches(npz_files, output_path)


if __name__ == "__main__":
    main()
