# CEB MegaSam (Blender Addon)



**CEB MegaSam** is a Blender add-on that integrates the **MegaSaM** pipeline (*Accurate, Fast and Robust Casual Structure and Motion from Casual Dynamic Videos*) directly into Blender's 3D Viewport.

It provides a complete visual interface for camera tracking, dense monocular depth estimation, point cloud reconstruction, and textured 3D mesh extraction from standard videos—running asynchronously in an external Python console without freezing Blender.

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Pretrained Checkpoints](#pretrained-checkpoints)
- [Installation](#installation)
- [How to Use](#how-to-use)
  - [1. Selecting Video & Frame Range](#1-selecting-video--frame-range)
  - [2. Choosing a VRAM Profile](#2-choosing-a-vram-profile)
  - [3. Running the Pipeline](#3-running-the-pipeline)
  - [4. Importing Results](#4-importing-results)
  - [5. Generating 3D Surface Meshes](#5-generating-3d-surface-meshes)
- [Optimization & Advanced Controls](#optimization--advanced-controls)
- [Licenses & Attribution](#licenses--attribution)

---

## Features

- **Embedded Pipeline**: MegaSam core scripts (`run_pipeline.py`) reside directly inside the add-on folder (`CEB_MegaSam/mega-sam`).
- **Zero-Freeze Asynchronous Execution**: Launches processing in a dedicated terminal window with live timers, progress bars, and diagnostics.

- **Smart Project Naming**: Automatically renames scene prefixes (e.g. `walk` &rarr; `walk_001` &rarr; `walk_002`) to prevent overwriting previous runs.
- **VRAM Presets**:
  - **LOWEST (~3–4 GB VRAM)**: 384px width, ViT-Small depth encoder, Tracking-only mode (skips CVD & RAFT).
  - **MEDIUM (~6–8 GB VRAM)**: 480px width, ViT-Base depth encoder, balanced CVD optimization.
  - **HIGH (12+ GB VRAM)**: 540px width, ViT-Large depth encoder, full CVD iterations.
- **Timeline & VSE Integration**:
  - Load video directly into Blender's Video Sequence Editor (VSE).
  - Sync processing frame range to Blender's timeline playback range with a single click.
- **Batching & Chunk Stitching**: Automatically splits long video sequences into overlapping chunks to run within low VRAM limits, stitching them into a unified camera trajectory.
- **Point Cloud & Camera Importer**: Imports solved camera motion and animated 3D point clouds with customizable point size, step rate, motion thresholding, and parenting to an empty.
- **3D Mesh Reconstruction**: Generates textured 3D surface meshes (`.obj`) for any chosen frame directly from reconstructed depth.

---

## Folder Structure

The add-on is designed to work alongside a portable Python environment:

```text
D:/_Code/Meu/CEB_MegaSam_prj/
├── CEB_MegaSam/                                 # Blender Add-on directory
│   ├── __init__.py                              # Registration, preferences & properties
│   ├── operators.py                             # Pipeline runner, importers, mesh generator
│   ├── panel.py                                 # 3D Viewport N-Panel UI (CEB tab)
│   ├── utils.py                                 # Auto-detection helpers
│   └── mega-sam/                                # MegaSam core repository
│       ├── run_pipeline.py                      # Pipeline master script
│       ├── stitch_npz_batches.py                # Batch stitching script
│       ├── generate_mesh_from_npz_FINAL.py      # Mesh extraction script
│       ├── camera_tracking_scripts/             # DROID-SLAM tracking scripts
│       ├── cvd_opt/                             # CVD & RAFT optimization
│       ├── Depth-Anything/                      # Depth-Anything modules
│       ├── UniDepth/                            # UniDepth metric depth modules
│       └── checkpoints/
│           └── megasam_final.pth
│
└── Portable_torch210_cu128/                     # Portable Python environment
    ├── python-3.10.11-embed-amd64/
    │   └── python.exe                           # Portable Python executable
    ├── DepthAnything_CKPT/
    │   └── checkpoints/
    │       ├── depth_anything_vits14.pth        # ViT-Small checkpoint
    │       ├── depth_anything_vitb14.pth        # ViT-Base checkpoint
    │       └── depth_anything_vitl14.pth        # ViT-Large checkpoint
    └── outputs/                                 # Saved results & project folders
        └── <scene_name>/
            ├── <scene_name>_sgd_cvd_hr.npz      # Final reconstructed result
            ├── outputs/                         # DROID tracking results
            └── outputs_cvd/                     # CVD optimized results
```

---

## Prerequisites & Requirements

1. **Blender**: Version **4.4+** or **5.2+** (Windows 64-bit).
2. **Python Environment**:
   - Python 3.10 with CUDA-enabled PyTorch (e.g. PyTorch 2.10 with CUDA 12.8 or PyTorch 2.0.1 with CUDA 11.8).
   - Pre-compiled `droid_backends` wheel.
   - `torch_scatter`.
   - `opencv-python`, `imageio`, `tqdm`, `einops`, `scipy`, `trimesh`, `open3d`.
3. **NVIDIA GPU**:
   - Minimum: 4 GB VRAM (using `LOWEST` profile).
   - Recommended: 8 GB–12+ GB VRAM.

---

## Pretrained Checkpoints

Ensure the following weights are in place:

| Model | Path | Source |
|---|---|---|
| **MegaSam** | `CEB_MegaSam/mega-sam/checkpoints/megasam_final.pth` | [MegaSam Release](https://github.com/mega-sam/mega-sam) |
| **RAFT Optical Flow** | `CEB_MegaSam/mega-sam/cvd_opt/raft-things.pth` | [RAFT Pretrained](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) |
| **Depth-Anything ViT-S** | `Portable_torch210_cu128/DepthAnything_CKPT/checkpoints/depth_anything_vits14.pth` | [Depth-Anything HuggingFace](https://huggingface.co/spaces/LiheYoung/Depth-Anything) |
| **Depth-Anything ViT-B** | `Portable_torch210_cu128/DepthAnything_CKPT/checkpoints/depth_anything_vitb14.pth` | [Depth-Anything HuggingFace](https://huggingface.co/spaces/LiheYoung/Depth-Anything) |
| **Depth-Anything ViT-L** | `Portable_torch210_cu128/DepthAnything_CKPT/checkpoints/depth_anything_vitl14.pth` | [Depth-Anything HuggingFace](https://huggingface.co/spaces/LiheYoung/Depth-Anything) |

---

## Installation

1. Copy or create a symbolic link / junction of the `CEB_MegaSam` folder into your Blender addons directory:
   ```text
   C:\Users\<YourUser>\AppData\Roaming\Blender Foundation\Blender\<Version>\scripts\addons\CEB_MegaSam
   ```
2. Open Blender and go to **Edit &rarr; Preferences &rarr; Add-ons**.
3. Search for **CEB MegaSam** and enable the checkbox.
4. Expand the add-on preferences:
   - **Python Path**: Leave blank to auto-detect the portable Python, or specify a custom `python.exe`.
   - **MegaSam Root**: Leave blank to auto-detect the `mega-sam` folder inside the add-on.
   - Verify the **DepthAnything CKPT** path shows a checkmark (`CHECKMARK`).

---

## How to Use

Open the 3D Viewport and press `N` to open the sidebar. Select the **CEB** tab.

### 1. Selecting Video & Frame Range
- **Video File**: Select your input video (`.mp4`, `.mov`, `.avi`).
- **Scene Name**: Automatically populated from the video file name. Auto-increments if a project with the same name already exists.
- **Open in VSE**: Loads the video into Blender's Video Sequence Editor to scrub frames and verify cuts.
- **Frame Range**:
  - Click **Current** to lock the range to Blender's timeline `[Start, End]`.
  - Or enable **Trim** to enter custom start and end frame numbers.

### 2. Choosing a VRAM Profile
- Choose between **LOWEST**, **MEDIUM**, or **HIGH**:
  - `LOWEST`: Fast camera tracking only (~3–4 GB VRAM).
  - `MEDIUM`: Balanced depth optimization (~6–8 GB VRAM).
  - `HIGH`: Maximum quality dense depth and camera reconstruction (12+ GB VRAM).

### 3. Running the Pipeline
- Click **Run MegaSam Pipeline**.
- A new command console opens displaying step-by-step progress:
  1. *Frame Extraction*
  2. *Depth-Anything Estimation* (using `DepthAnything_CKPT`)
  3. *UniDepth Metric Depth Estimation*
  4. *DROID Camera Tracking*
  5. *RAFT Optical Flow* (skipped if Tracking Only)
  6. *Consistent Video Depth (CVD) Optimization* (skipped if Tracking Only)
  7. *Batch Stitching* (if batching is enabled)

### 4. Importing Results
- Under **Results**, select your processed `.npz` file from the dropdown.
- Click **Import Result**:
  - Creates an animated camera with focal length and keyframes.
  - Generates the animated or static 3D point cloud colored by video frames.
  - Optionally parents all imported elements to an Empty named after the video.

### 5. Generating 3D Surface Meshes
- Scrub the timeline to the desired frame.
- Under **Mesh Generation (3D Surface)**:
  - Adjust **Mesh Threshold**, **Upscale**, and **Smoothing**.
  - Click **Generate Mesh (Current Frame)** to generate and import an `.obj` textured 3D mesh directly into the scene.

---

## Optimization & Advanced Controls

Accessible via the sub-panels in the sidebar:

- **Tracking Stride**: Set to `2` or `3` for 50–65% faster tracking with lower memory footprint.
- **Frontend / Backend BA Iterations**: Fine-tune bundle adjustment steps for speed vs. convergence.
- **Keyframe Buffer**: Lower buffer size (e.g. `256`) for limited VRAM.
- **CUDA Memory Allocator Opt**: Enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid memory fragmentation.
- **Batch Processing**: Set batch size (e.g. `100` frames) to track long videos without running out of VRAM. Overlaps and relative SE(3) alignment are automatically solved.

---

## Licenses & Attribution

### Original MegaSaM Codebase
This add-on embeds and interfaces with **MegaSaM**:
- **Repository**: [https://github.com/mega-sam/mega-sam](https://github.com/mega-sam/mega-sam)
- **Project Page**: [https://mega-sam.github.io/index.html](https://mega-sam.github.io/index.html)
- **Paper**: *MegaSaM: Accurate, Fast and Robust Casual Structure and Motion from Casual Dynamic Videos* (CVPR 2025)  
  *Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye, Angjoo Kanazawa, Aleksander Holynski, Noah Snavely.*
- **License**: The software in MegaSaM is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0). Materials are licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/). *MegaSaM is not an officially supported Google product.*

### Third-Party Models and Dependencies
Please review and adhere to the respective licenses of the original repositories:
- **Depth-Anything**: [Apache License 2.0](https://github.com/LiheYoung/Depth-Anything/blob/main/LICENSE)
- **UniDepth**: [UniDepth License](https://github.com/lpiccinelli-eth/UniDepth)
- **DROID-SLAM**: [DROID-SLAM License](https://github.com/princeton-vl/DROID-SLAM)
- **RAFT**: [BSD 3-Clause License](https://github.com/princeton-vl/RAFT)

---

## Citation

If you use MegaSaM in academic work, please cite the original authors:

```bibtex
@inproceedings{li2025megasam,
  title     = {{MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos}},
  author    = {Li, Zhengqi and Tucker, Richard and Cole, Forrester and Wang, Qianqian and Jin, Linyi and Ye, Vickie and Kanazawa, Angjoo and Holynski, Aleksander and Snavely, Noah},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```
