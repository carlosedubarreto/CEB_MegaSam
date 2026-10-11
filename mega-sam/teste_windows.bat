@REM Arquivo mono_depth_demo.sh
@REM run depth anything

python Depth-Anything/run_videos.py --encoder vitl --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth --img-path D:/AI/MegaSam/DAVIS/swing --outdir Depth-Anything/video_visualization/swing


@REM Run Unidepth
@REM set PYTHONPATH=%PYTHONPATH%;D:\AI\MegaSam\mega-sam\UniDepth
set PYTHONPATH=D:\AI\MegaSam\mega-sam\UniDepth

python UniDepth/scripts/demo_mega-sam.py --scene-name swing --img-path D:\AI\MegaSam\DAVIS\swing --outdir UniDepth\outputs

@REM ------------------------------------------------------

@REM Evaluate demo

python camera_tracking_scripts/test_demo.py --datapath=D:/AI/MegaSam/DAVIS/swing --weights=D:/AI/MegaSam/mega-sam/checkpoints/megasam_final.pth --scene_name swing --mono_depth_path D:/AI/MegaSam/mega-sam/Depth-Anything/video_visualization --metric_depth_path D:/AI/MegaSam/mega-sam/UniDepth/outputs --disable_vis 

@REM ------------------------------------------------------

@REM cvd_opt_demo

@REM run raft optical flows
python cvd_opt/preprocess_flow.py --datapath=D:/AI/MegaSam/DAVIS/swing --model=cvd_opt/raft-things.pth --scene_name swing --mixed_precision

@REM run CVD optimization
python cvd_opt/cvd_opt.py --scene_name swing --w_grad 2.0 --w_normal 5.0