import bpy
import os
from .utils import get_python_path, get_megasam_root

class VIEW3D_PT_megasam_panel(bpy.types.Panel):
    bl_label = "CEB MegaSam"
    bl_idname = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Access addon preferences
        addon_name = __package__.split('.')[0]
        prefs = context.preferences.addons[addon_name].preferences

        # 1. Environment Section
        box = layout.box()
        box.label(text="Environment", icon='PREFERENCES')
        
        col = box.column(align=True)
        col.prop(prefs, "python_path", text="Python")
        if not prefs.python_path:
            auto_py = get_python_path(context)
            if auto_py and os.path.exists(auto_py):
                col.label(text=f"Auto: {os.path.basename(os.path.dirname(auto_py))}\\python.exe", icon='CHECKMARK')
            else:
                col.label(text="Python not found", icon='ERROR')
                
        col.separator()
        col.prop(prefs, "megasam_root_path", text="MegaSam")
        if not prefs.megasam_root_path:
            auto_root = get_megasam_root(context)
            if auto_root and os.path.exists(auto_root):
                col.label(text=f"Auto: {os.path.basename(auto_root)}", icon='CHECKMARK')
            else:
                col.label(text="MegaSam root not found", icon='ERROR')
        
        layout.separator()
        
        # 2. Input Video Section
        box = layout.box()
        box.label(text="Input Video", icon='FILE_MOVIE')
        box.prop(scene, "megasam_video_path", text="Video File")
        box.prop(scene, "megasam_scene_name", text="Scene Name")
        
        if scene.megasam_video_frames > 0:
            row = box.column(align=True)
            row.label(text=f"Original: {scene.megasam_video_width}x{scene.megasam_video_height} | {scene.megasam_video_frames} Frames", icon='IMAGE_DATA')
        
        # VSE Controls
        vse_sub = box.box()
        vse_sub.label(text="Video Sequence Editor (VSE)", icon='SEQUENCE')
        vcol = vse_sub.column(align=True)
        vcol.prop(scene, "megasam_vse_view_mode", text="VSE Mode")
        vcol.prop(scene, "megasam_vse_clear_existing", text="Clear VSE Strips before Load")
        
        vrow = vcol.row(align=True)
        vrow.operator("megasam.open_vse", icon='WINDOW', text="Open VSE")
        vrow.operator("megasam.load_video_vse", icon='IMPORT', text="Open Video in VSE")
        
        # 3. Performance & Speed Optimization Controls
        sub = box.box()
        sub.label(text="Performance & Speed Optimization", icon='MODIFIER')
        
        col = sub.column(align=True)
        col.prop(scene, "megasam_vram_preset", text="VRAM Profile")
        if scene.megasam_vram_preset == 'LOWEST':
            col.label(text="Profile: ~3-4GB (384p, ViT-S, Tracking Only)", icon='INFO')
        elif scene.megasam_vram_preset == 'MEDIUM':
            col.label(text="Profile: ~6-8GB (480p, ViT-B, Balanced CVD)", icon='INFO')
        elif scene.megasam_vram_preset == 'HIGH':
            col.label(text="Profile: 12GB+ (540p, ViT-L, Full CVD)", icon='INFO')
        
        col.separator()
        col.prop(scene, "megasam_res_preset", text="Resolution")
        if scene.megasam_res_preset == 'CUSTOM':
            col.prop(scene, "megasam_target_width", text="Width (px)")
        else:
            col.label(text=f"Processing Width: {scene.megasam_target_width} px", icon='DRIVER')
        
        col.separator()
        col.prop(scene, "megasam_depth_encoder", text="Depth Model")
        
        col.separator()
        # CVD / Tracking-only mode
        col.prop(scene, "megasam_skip_cvd", text="Tracking Only (Fast / Skip CVD)")
        if not scene.megasam_skip_cvd:
            col.prop(scene, "megasam_cvd_iters", text="CVD Iterations")
        else:
            col.label(text="Skips RAFT flow & dense CVD optimization", icon='INFO')
        
        sub.separator()
        
        # Frame Range
        col = sub.column(align=True)
        col.prop(scene, "megasam_use_frame_range", text="Trim Frame Range")
        if scene.megasam_use_frame_range:
            row = col.row(align=True)
            row.prop(scene, "megasam_start_frame", text="Start")
            row.prop(scene, "megasam_end_frame", text="End")
            col.operator("megasam.set_range_from_timeline", icon='TIME', text="From Blender Timeline")
            
            if scene.megasam_end_frame > scene.megasam_start_frame:
                frame_count = scene.megasam_end_frame - scene.megasam_start_frame
                col.label(text=f"Processing {frame_count} frames", icon='INFO')
        
        sub.separator()
        
        # Batching
        col = sub.column(align=True)
        col.prop(scene, "megasam_use_batch", text="Enable Batching (Low VRAM)")
        if scene.megasam_use_batch:
            col.prop(scene, "megasam_batch_size", text="Batch Size")
            
            total_f = scene.megasam_video_frames
            if scene.megasam_use_frame_range and scene.megasam_end_frame > scene.megasam_start_frame:
                total_f = scene.megasam_end_frame - scene.megasam_start_frame
                
            if total_f > 0 and scene.megasam_batch_size > 0:
                batch_count = (total_f + scene.megasam_batch_size - 1) // scene.megasam_batch_size
                col.label(text=f"Splits into ~{batch_count} batches (auto-stitched)", icon='FORCE_CHARGE')
        
        sub.separator()
        sub.prop(scene, "megasam_low_vram", text="CUDA Memory Allocator Opt")
        
        # 4. Advanced SLAM Backend Controls
        sub.separator()
        adv_box = sub.box()
        adv_box.prop(scene, "megasam_show_advanced", icon='TRIA_DOWN' if scene.megasam_show_advanced else 'TRIA_RIGHT', text="Advanced SLAM (DROID) Settings", emboss=False)
        if scene.megasam_show_advanced:
            acol = adv_box.column(align=True)
            acol.prop(scene, "megasam_stride", text="Tracking Stride")
            if scene.megasam_stride > 1:
                acol.label(text=f"Tracks every {scene.megasam_stride} frames (50-65% faster)", icon='INFO')
            acol.prop(scene, "megasam_frontend_iters", text="Frontend BA Iters")
            acol.prop(scene, "megasam_backend_iters", text="Backend BA Iters")
            acol.prop(scene, "megasam_buffer", text="Keyframe Buffer")
            acol.prop(scene, "megasam_keyframe_thresh", text="Keyframe Thresh")

        layout.separator()
        
        # 5. Run Button
        layout.operator("megasam.run_pipeline", icon='PLAY', text="Run MegaSam Pipeline")
        
        layout.separator()
        
        # 6. Results Section
        box = layout.box()
        box.label(text="Results", icon='FILE_TICK')
        box.prop(scene, "megasam_output_filter", expand=True)
        box.prop(scene, "megasam_result_list", text="")
        box.prop(scene, "megasam_point_radius", icon='PARTICLE_POINT')
        
        # Import Config
        sub = box.box()
        sub.label(text="Import Settings", icon='SETTINGS')
        col = sub.column(align=True)
        col.prop(scene, "megasam_import_frame_step")
        col.prop(scene, "megasam_import_pixel_step")
        col.prop(scene, "megasam_import_filter_mode")
        if scene.megasam_import_filter_mode != 'ALL':
            col.prop(scene, "megasam_import_motion_threshold")
        col.prop(scene, "megasam_import_animate")
        col.prop(scene, "megasam_import_clean_scene")
        col.prop(scene, "megasam_import_parent_to_empty")
        
        row = box.row(align=True)
        row.operator("megasam.import_result", icon='IMPORT', text="Import Selected Result")
        row.operator("megasam.open_outputs_folder", icon='FILE_FOLDER', text="Open Project Folder")
        
        # 7. Mesh Generation
        sub = box.box()
        sub.label(text="Mesh Generation", icon='MESH_DATA')
        col = sub.column(align=True)
        col.prop(scene, "megasam_mesh_use_import_settings")
        col.prop(scene, "megasam_mesh_threshold")
        col.prop(scene, "megasam_mesh_upscale")
        col.prop(scene, "megasam_mesh_smooth")
        col.prop(scene, "megasam_mesh_output_dir", text="Output Folder")
        sub.operator("megasam.generate_mesh", icon='MOD_MESHDEFORM', text="Generate Mesh (Current Frame)")

def register():
    try:
        bpy.utils.register_class(VIEW3D_PT_megasam_panel)
    except ValueError:
        pass

def unregister():
    try:
        bpy.utils.unregister_class(VIEW3D_PT_megasam_panel)
    except Exception:
        pass
