import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def convert_video(video_path, output_root, scene_name, target_width=None, start_frame=0, end_frame=None):
    # 1. Setup paths
    scene_dir = Path(output_root) / scene_name
    img_dir = scene_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}")
    print(f"Total Frames in Video: {total_frames}")

    # Set start frame
    if start_frame > 0:
        print(f"Starting from frame {start_frame}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    count = start_frame
    processed_count = 0
    
    print(f"Extracting frames from {start_frame} to {end_frame if end_frame else 'end'}...")
    while cap.isOpened():
        if end_frame is not None and count >= end_frame:
            print(f"Reached end frame {end_frame}. Stopping.")
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize if requested
        if target_width:
            h, w = frame.shape[:2]
            ratio = target_width / w
            frame = cv2.resize(frame, (target_width, int(h * ratio)))

        # Save individual image
        img_path = img_dir / f"frame_{count:04d}.jpg"
        cv2.imwrite(str(img_path), frame)
        
        count += 1
        processed_count += 1
        if processed_count % 50 == 0:
            print(f"Processed {processed_count} frames (Current frame: {count})...")

    cap.release()
    
    print("\n" + "="*30)
    print(f"Success! Processed {processed_count} frames.")
    print(f"Images saved to: {img_dir}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to image sequence.")
    parser.add_argument("--video_path", "-v", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output_root", "-o", type=str, default=".", help="Root directory for output (default: current dir)")
    parser.add_argument("--scene_name", "-s", type=str, required=True, help="Name of the scene (folder name)")
    parser.add_argument("--target_width", "-w", type=int, default=None, help="Resize frames to this width (optional)")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame number to start extraction from (default: 0)")
    parser.add_argument("--end_frame", type=int, default=None, help="Frame number to end extraction at (optional)")
    
    args = parser.parse_args()
    
    convert_video(
        video_path=args.video_path,
        output_root=args.output_root,
        scene_name=args.scene_name,
        target_width=args.target_width,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )

