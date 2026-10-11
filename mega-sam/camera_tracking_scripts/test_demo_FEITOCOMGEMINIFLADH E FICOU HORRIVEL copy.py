# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Test camera tracking on a single scene."""

# pylint: disable=invalid-name
# pylint: disable=g-importing-member
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=redefined-outer-name
# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import sys

sys.path.append("base/droid_slam")

from tqdm import tqdm
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from lietorch import SE3
from scipy.spatial.transform import Rotation as Rot

import torch.nn.functional as F
from droid import Droid


def umeyama_alignment(x, y, with_scale=True):
  """
  Computes the transformation Y = sRX + t using the Umeyama algorithm.
  x, y: (N, 3) numpy arrays.
  Returns: s, R, t
  """
  # Ensure we are working with (N, 3)
  x = x[:, :3]
  y = y[:, :3]
  
  mu_x = x.mean(axis=0)
  mu_y = y.mean(axis=0)
  x_centered = x - mu_x
  y_centered = y - mu_y
  n = x.shape[0]

  sigma2_x = (x_centered**2).sum() / n
  cov_xy = (y_centered.T @ x_centered) / n

  u, d, vh = np.linalg.svd(cov_xy)
  S = np.eye(3)
  if np.linalg.det(u) * np.linalg.det(vh) < 0:
    S[2, 2] = -1

  r = u @ S @ vh
  if with_scale:
    scale = (d @ S.diagonal()) / (sigma2_x + 1e-8)
  else:
    scale = 1.0

  t = mu_y - scale * r @ mu_x
  return scale, r, t


def image_stream(
    image_list,
    mono_disp_paths,
    scene_name,
    use_depth=False,
    aligns=None,
    K=None,
    stride=1,
):
  """image generator."""
  del scene_name, stride

  fx, fy, cx, cy = (
      K[0, 0],
      K[1, 1],
      K[0, 2],
      K[1, 2],
  )

  for t, (image_file) in enumerate(image_list):
    image = cv2.imread(image_file)
    mono_disp = np.float32(np.load(mono_disp_paths[t]))

    h_m, w_m = mono_disp.shape
    # Ensure mono_disp matches K expectations or resize if needed
    # (Actually in test_demo it was already resized in the calibration loop,
    # but here we load raw, so we need to resize it to match the expected resolution)

    depth = np.clip(
        1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
        1e-4,
        1e4,
    )
    depth[depth < 1e-2] = 0.0

    h0, w0, _ = image.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
    image = image[: h1 - h1 % 8, : w1 - w1 % 8]

    image = torch.as_tensor(image).permute(2, 0, 1)

    depth = torch.as_tensor(depth)
    depth = F.interpolate(
        depth[None, None], (h1, w1), mode="nearest-exact"
    ).squeeze()
    depth = depth[: h1 - h1 % 8, : w1 - w1 % 8]

    mask = torch.ones_like(depth)

    intrinsics = torch.as_tensor([fx, fy, cx, cy])
    intrinsics[0::2] *= w1 / w0
    intrinsics[1::2] *= h1 / h0

    if use_depth:
      yield t, image[None], depth, intrinsics, mask
    else:
      yield t, image[None], intrinsics, mask


def save_full_reconstruction(
    intrinsics_all,
    full_traj,
    image_list,
    mono_disp_paths,
    aligns,
    K,
    motion_prob,
    scene_name,
):
  """Save full reconstruction."""
  from pathlib import Path
  t = full_traj.shape[0]

  images_list = []
  depths_list = []

  # Re-process images and depths for saving
  print("Processing frames for saving...")
  for i in tqdm(range(t)):
    img = cv2.imread(image_list[i])
    mono_disp = np.float32(np.load(mono_disp_paths[i]))
    depth = np.clip(
        1.0 / ((1.0 / aligns[2]) * (aligns[0] * mono_disp + aligns[1])),
        1e-4,
        1e4,
    )
    depth[depth < 1e-2] = 0.0

    h0, w0, _ = img.shape
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

    img_res = cv2.resize(img, (w1, h1), interpolation=cv2.INTER_AREA)
    img_res = img_res[: h1 - h1 % 8, : w1 - w1 % 8]
    images_list.append(img_res.transpose(2, 0, 1))

    depth_th = torch.as_tensor(depth)
    depth_res = F.interpolate(
        depth_th[None, None], (h1, w1), mode="nearest-exact"
    ).squeeze()
    depth_res = depth_res[: h1 - h1 % 8, : w1 - w1 % 8]
    depths_list.append(depth_res.numpy())

  images = np.array(images_list)
  disps = 1.0 / (np.array(depths_list) + 1e-6)

  poses = full_traj
  intrinsics = intrinsics_all[:t]

  Path("reconstructions/{}".format(scene_name)).mkdir(
      parents=True, exist_ok=True
  )
  np.save("reconstructions/{}/images.npy".format(scene_name), images)
  np.save("reconstructions/{}/disps.npy".format(scene_name), disps)
  np.save("reconstructions/{}/poses.npy".format(scene_name), poses)
  np.save(
      "reconstructions/{}/intrinsics.npy".format(scene_name), intrinsics * 8.0
  )
  np.save("reconstructions/{}/motion_prob.npy".format(scene_name), motion_prob)

  intrinsics = intrinsics[0] * 8.0
  poses_th = torch.as_tensor(poses, device="cpu")
  cam_c2w = SE3(poses_th).inv().matrix().numpy()

  K_out = np.eye(3)
  K_out[0, 0] = intrinsics[0]
  K_out[1, 1] = intrinsics[1]
  K_out[0, 2] = intrinsics[2]
  K_out[1, 2] = intrinsics[3]
  print("K ", K_out)
  print("img_data ", images.shape)
  print("disp_data ", disps.shape)

  # max_frames = min(1000, images.shape[0]) # Removed the limit for sliding window
  print("outputs/%s_droid.npz" % scene_name)
  Path("outputs").mkdir(parents=True, exist_ok=True)

  np.savez(
      "outputs/%s_droid.npz" % scene_name,
      images=np.uint8(images[:, ::-1, ...].transpose(0, 2, 3, 1)),
      depths=np.float32(1.0 / disps),
      intrinsic=K_out,
      cam_c2w=cam_c2w,
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--datapath")
  parser.add_argument("--weights", default="droid.pth")
  parser.add_argument("--buffer", type=int, default=1024)
  parser.add_argument("--image_size", default=[240, 320])
  parser.add_argument("--disable_vis", action="store_true")

  parser.add_argument("--beta", type=float, default=0.3)
  parser.add_argument(
      "--filter_thresh", type=float, default=2.0
  )  # motion threhold for keyframe
  parser.add_argument("--warmup", type=int, default=8)
  parser.add_argument("--keyframe_thresh", type=float, default=2.0)
  parser.add_argument("--frontend_thresh", type=float, default=12.0)
  parser.add_argument("--frontend_window", type=int, default=25)
  parser.add_argument("--frontend_radius", type=int, default=2)
  parser.add_argument("--frontend_nms", type=int, default=1)

  parser.add_argument("--stereo", action="store_true")
  parser.add_argument("--depth", action="store_true")
  parser.add_argument("--upsample", action="store_true")
  parser.add_argument("--scene_name", help="scene_name")

  parser.add_argument("--backend_thresh", type=float, default=16.0)
  parser.add_argument("--backend_radius", type=int, default=2)
  parser.add_argument("--backend_nms", type=int, default=3)

  parser.add_argument(
      "--mono_depth_path", default="Depth-Anything/video_visualization"
  )
  parser.add_argument("--metric_depth_path", default="UniDepth/outputs ")

  # Sliding window arguments
  parser.add_argument("--window_size", type=int, default=30)
  parser.add_argument("--stride", type=int, default=10)

  args = parser.parse_args()

  print("Running evaluation on {}".format(args.datapath))
  print(args)

  scene_name = args.scene_name.split("/")[-1]

  image_list = sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.jpg")))
  image_list += sorted(glob.glob(os.path.join("%s" % (args.datapath), "*.png")))

  # NOTE Mono is inverse depth, but metric-depth is depth!
  mono_disp_paths = sorted(
      glob.glob(
          os.path.join("%s/%s" % (args.mono_depth_path, scene_name), "*.npy")
      )
  )
  metric_depth_paths = sorted(
      glob.glob(
          os.path.join("%s/%s" % (args.metric_depth_path, scene_name), "*.npz")
      )
  )

  num_frames = len(image_list)
  print(f"Total frames: {num_frames}")

  # --- Optimized Calibration Sampling ---
  print("Calculating alignment parameters (sampling frames)...")
  img_0 = cv2.imread(image_list[0])
  scales = []
  shifts = []
  fovs = []
  sample_indices = np.linspace(
      0, num_frames - 1, min(100, num_frames), dtype=int
  )
  mono_disp_samples = []

  for idx in tqdm(sample_indices):
    da_disp = np.float32(np.load(mono_disp_paths[idx]))
    uni_data = np.load(metric_depth_paths[idx])
    metric_depth = uni_data["depth"]
    fovs.append(uni_data["fov"])

    da_disp = cv2.resize(
        da_disp,
        (metric_depth.shape[1], metric_depth.shape[0]),
        interpolation=cv2.INTER_NEAREST_EXACT,
    )
    mono_disp_samples.append(da_disp)
    gt_disp = 1.0 / (metric_depth + 1e-8)

    valid_mask = (metric_depth < 2.0) & (da_disp < 0.02)
    gt_disp[valid_mask] = 1e-2

    sky_ratio = np.sum(da_disp < 0.01) / (da_disp.shape[0] * da_disp.shape[1])
    if sky_ratio > 0.5:
      non_sky_mask = da_disp > 0.01
      gt_disp_ms = gt_disp[non_sky_mask] - np.median(gt_disp[non_sky_mask]) + 1e-8
      da_disp_ms = da_disp[non_sky_mask] - np.median(da_disp[non_sky_mask]) + 1e-8
      scale = np.median(gt_disp_ms / da_disp_ms)
      shift = np.median(gt_disp[non_sky_mask] - scale * da_disp[non_sky_mask])
    else:
      gt_disp_ms = gt_disp - np.median(gt_disp) + 1e-8
      da_disp_ms = da_disp - np.median(da_disp) + 1e-8
      scale = np.median(gt_disp_ms / da_disp_ms)
      shift = np.median(gt_disp - scale * da_disp)

    scales.append(scale)
    shifts.append(shift)

  print("************** UNIDEPTH FOV ", np.median(fovs))
  ff = img_0.shape[1] / (2 * np.tan(np.radians(np.median(fovs) / 2.0)))
  K = np.eye(3)
  K[0, 0] = ff
  K[1, 1] = ff
  K[0, 2] = img_0.shape[1] / 2.0
  K[1, 2] = img_0.shape[0] / 2.0

  med_idx = np.argmin(
      np.abs(np.array(scales) * np.array(shifts) - np.median(np.array(scales) * np.array(shifts)))
  )
  align_scale = scales[med_idx]
  align_shift = shifts[med_idx]
  normalize_scale = (
      np.percentile(
          (align_scale * np.array(mono_disp_samples) + align_shift), 98
      )
      / 2.0
  )
  aligns = (align_scale, align_shift, normalize_scale)
  # -------------------------------------

  # --- Calculate expected image size ---
  h0, w0, _ = img_0.shape
  h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
  w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
  args.image_size = [h1 - h1 % 8, w1 - w1 % 8]
  print(f"Internal image size: {args.image_size}")

  # --- Sliding Window Processing ---
  final_poses = np.zeros((num_frames, 7))
  final_motion_prob = np.zeros((num_frames, args.image_size[0] // 8, args.image_size[1] // 8))
  final_intrinsics = np.zeros((num_frames, 4))
  processed_mask = np.zeros(num_frames, dtype=bool)

  window_size = args.window_size
  stride = args.stride
  
  last_end_idx = 0

  print(f"Starting sliding window processing (window={window_size}, stride={stride})...")

  for start_idx in range(0, num_frames, stride):
    end_idx = min(start_idx + window_size, num_frames)
    
    # Skip if this window is already covered
    if end_idx <= last_end_idx:
      continue
      
    if end_idx - start_idx < 5 and start_idx > 0: # Too small chunk
      break
    
    print(f"\nProcessing window: [{start_idx} : {end_idx}]")
    
    win_image_list = image_list[start_idx:end_idx]
    win_mono_paths = mono_disp_paths[start_idx:end_idx]

    # Initialize DROID for this window
    args.buffer = window_size + 10
    droid = Droid(args)

    # Track window
    for t, image, depth, intrinsics, mask in tqdm(
        image_stream(
            win_image_list,
            win_mono_paths,
            scene_name,
            use_depth=True,
            aligns=aligns,
            K=K,
        ),
        total=len(win_image_list),
        desc="Tracking"
    ):
      droid.track(t, image, depth, intrinsics=intrinsics, mask=mask)

    droid.track_final(t, image, depth, intrinsics=intrinsics, mask=mask)

    # Refine window
    traj_est, depth_est, motion_prob = droid.terminate(
        image_stream(
            win_image_list,
            win_mono_paths,
            scene_name,
            use_depth=True,
            aligns=aligns,
            K=K,
        ),
        _opt_intr=True,
        full_ba=True,
        scene_name=scene_name,
    )

    # Extract optimized intrinsics
    win_intrinsics = droid.video.intrinsics[:droid.video.counter.value].cpu().numpy()

    # Stitch trajectory
    if last_end_idx == 0:
      final_poses[:end_idx] = traj_est
      final_motion_prob[:end_idx] = motion_prob
      final_intrinsics[:end_idx] = win_intrinsics
    else:
      # Use actual overlap frames to align
      actual_overlap = last_end_idx - start_idx
      if actual_overlap < 3:
         print("Warning: very small overlap for stitching!")
         actual_overlap = max(1, actual_overlap)

      # Convert to camera-to-world to get camera centers for alignment
      # local: traj_est (world-to-camera)
      # global: final_poses (world-to-camera)
      
      # Extract local overlap poses and convert to c2w
      local_overlap_vecs = torch.as_tensor(traj_est[:actual_overlap])
      local_overlap_se3 = SE3(local_overlap_vecs)
      local_overlap_c2w = local_overlap_se3.inv()
      x_local = local_overlap_c2w.translation().numpy() # [N, 3]

      # Extract global overlap poses and convert to c2w
      global_overlap_vecs = torch.as_tensor(final_poses[start_idx:last_end_idx])
      global_overlap_se3 = SE3(global_overlap_vecs)
      global_overlap_c2w = global_overlap_se3.inv()
      y_global = global_overlap_c2w.translation().numpy() # [N, 3]

      # Align camera centers
      s, R_a, t_a = umeyama_alignment(x_local, y_global, with_scale=True)
      r_quat_a = Rot.from_matrix(R_a).as_quat() # x,y,z,w
      
      # Transform the whole window
      # 1. Convert all window poses to c2w
      win_vecs = torch.as_tensor(traj_est)
      win_se3_c2w = SE3(win_vecs).inv()
      
      # Use .vec() to avoid internal NameError in lietorch's .quaternion()
      win_vecs_c2w = win_se3_c2w.vec()
      win_centers = win_vecs_c2w[:, :3].cpu().numpy()
      win_quats = win_vecs_c2w[:, 3:].cpu().numpy() # x,y,z,w
      
      # 2. Apply similarity transform: C' = s*R_a*C + t_a, R' = R_a*R
      new_centers = s * (win_centers @ R_a.T) + t_a
      
      new_quats = []
      rot_a = Rot.from_quat(r_quat_a)
      for q in win_quats:
        new_q = (rot_a * Rot.from_quat(q)).as_quat()
        new_quats.append(new_q)
      new_quats = np.array(new_quats)
      
      # 3. Convert back to world-to-camera vectors
      new_win_data = torch.cat([torch.as_tensor(new_centers, dtype=torch.float32), 
                               torch.as_tensor(new_quats, dtype=torch.float32)], dim=-1)
      new_win_c2w = SE3(new_win_data)
      new_window_poses = new_win_c2w.inv().vec().cpu().numpy()

      # Fill in final_poses (taking only the non-overlapping part)
      global_copy_start = last_end_idx
      copy_start_local = actual_overlap
      
      if end_idx > global_copy_start:
        final_poses[global_copy_start:end_idx] = new_window_poses[copy_start_local:]
        final_motion_prob[global_copy_start:end_idx] = motion_prob[copy_start_local:]
        final_intrinsics[global_copy_start:end_idx] = win_intrinsics[copy_start_local:]

    last_end_idx = end_idx
    processed_mask[start_idx:end_idx] = True
    
    # Clean up GPU memory
    del droid
    torch.cuda.empty_cache()

  # Final Save
  if args.scene_name is not None:
    save_full_reconstruction(
        final_intrinsics,
        final_poses,
        image_list,
        mono_disp_paths,
        aligns,
        K,
        final_motion_prob,
        args.scene_name,
    )
