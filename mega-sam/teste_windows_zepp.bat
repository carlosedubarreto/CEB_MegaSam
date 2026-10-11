@REM Arquivo mono_depth_demo.sh
@REM run depth anything

python Depth-Anything/run_videos.py --encoder vitl --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth --img-path D:\AI\MegaSam\Custom_vid\zepp --outdir Depth-Anything/video_visualization/zepp


@REM Run Unidepth
@REM set PYTHONPATH=%PYTHONPATH%;D:\AI\MegaSam\mega-sam\UniDepth
@REM set PYTHONPATH=D:\AI\MegaSam\mega-sam\UniDepth

python UniDepth/scripts/demo_mega-sam.py --scene-name zepp --img-path D:\AI\MegaSam\Custom_vid\zepp --outdir UniDepth\outputs

@REM ------------------------------------------------------

@REM Evaluate demo
@REM --- esse aqui noa funciona com muitos frames, ele estoura a memoria da gpu


python camera_tracking_scripts/test_demo.py --datapath=D:\AI\MegaSam\Custom_vid\zepp --weights=D:/AI/MegaSam/mega-sam/checkpoints/megasam_final.pth --scene_name zepp --mono_depth_path D:/AI/MegaSam/mega-sam/Depth-Anything/video_visualization --metric_depth_path D:/AI/MegaSam/mega-sam/UniDepth/outputs --disable_vis --window_size 30 --stride 10

@REM ------------------------------------------------------

@REM cvd_opt_demo

@REM run raft optical flows
python cvd_opt/preprocess_flow.py --datapath=D:\AI\MegaSam\Custom_vid\zepp --model=cvd_opt/raft-things.pth --scene_name zepp --mixed_precision

@REM run CVD optimization
python cvd_opt/cvd_opt.py --scene_name zepp --w_grad 2.0 --w_normal 5.0