import bpy
import os
from .utils import get_python_path, get_megasam_root, get_depth_anything_ckpt

class VIEW3D_PT_megasam_panel(bpy.types.Panel):
    """Main panel for CEB MegaSam containing primary workflow controls."""
    bl_label = "CEB MegaSam"
    bl_idname = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        # 1. Input Video Section
        box = layout.box()
        box.label(text="Input Video", icon='FILE_MOVIE')
        box.prop(scene, "megasam_video_path", text="Video File")
        box.prop(scene, "megasam_scene_name", text="Scene Name")

        if scene.megasam_video_frames > 0:
            row = box.column(align=True)
            row.label(
                text=f"Original: {scene.megasam_video_width}x{scene.megasam_video_height} | {scene.megasam_video_frames} Frames",
                icon='IMAGE_DATA'
            )

        vrow = box.row(align=True)
        vrow.operator("megasam.load_video_vse", icon='IMPORT', text="Open in VSE")
        vrow.operator("megasam.open_vse", icon='WINDOW', text="VSE Editor")

        layout.separator()

        # 2. VRAM Profile Quick Selector
        pbox = layout.box()
        pbox.label(text="GPU / Performance Profile", icon='MODIFIER')
        pbox.prop(scene, "megasam_vram_preset", text="Profile")
        if scene.megasam_vram_preset == 'LOWEST':
            pbox.label(text="Profile: ~3-4GB (384p, ViT-S, Tracking Only)", icon='INFO')
        elif scene.megasam_vram_preset == 'MEDIUM':
            pbox.label(text="Profile: ~6-8GB (480p, ViT-B, Balanced CVD)", icon='INFO')
        elif scene.megasam_vram_preset == 'HIGH':
            pbox.label(text="Profile: 12GB+ (540p, ViT-L, Full CVD)", icon='INFO')

        layout.separator()

        # 3. Frame Range Section
        tbox = layout.box()
        tbox.label(text="Frame Range", icon='TIME')

        row = tbox.row(align=True)
        sub_trim = row.row(align=True)
        if scene.megasam_use_timeline_range:
            sub_trim.scale_x = 0.75
        sub_trim.prop(scene, "megasam_use_frame_range", text="Trim", toggle=True)

        if scene.megasam_use_timeline_range:
            current_btn_text = f"Current ({scene.frame_start} - {scene.frame_end})"
        else:
            current_btn_text = "Current"
        row.prop(scene, "megasam_use_timeline_range", text=current_btn_text, toggle=True)

        if scene.megasam_use_frame_range:
            r = tbox.row(align=True)
            r.prop(scene, "megasam_start_frame", text="Start")
            r.prop(scene, "megasam_end_frame", text="End")
            if scene.megasam_end_frame > scene.megasam_start_frame:
                frame_count = scene.megasam_end_frame - scene.megasam_start_frame
                tbox.label(text=f"{frame_count} frames", icon='INFO')
        elif scene.megasam_use_timeline_range:
            timeline_frames = max(0, scene.frame_end - scene.frame_start + 1) if scene.frame_end >= scene.frame_start else 0
            tbox.label(text=f"{timeline_frames} frames", icon='INFO')

        layout.separator()

        # 4. Main Run Pipeline Button
        col = layout.column(align=True)
        col.scale_y = 1.4
        col.operator("megasam.run_pipeline", icon='PLAY', text="Run MegaSam Pipeline")

        layout.separator()

        # 5. Results & Quick Import Section
        rbox = layout.box()
        rbox.label(text="Results", icon='FILE_TICK')
        rbox.prop(scene, "megasam_output_filter", expand=True)
        rbox.prop(scene, "megasam_result_list", text="")

        r_btns = rbox.row(align=True)
        r_btns.scale_y = 1.2
        r_btns.operator("megasam.import_result", icon='IMPORT', text="Import Result")
        r_btns.operator("megasam.open_outputs_folder", icon='FILE_FOLDER', text="Outputs Folder")


class VIEW3D_PT_megasam_optimization(bpy.types.Panel):
    """Sub-panel for detailed speed, resolution, depth model, and SLAM settings."""
    bl_label = "Performance & Optimization"
    bl_idname = "VIEW3D_PT_megasam_optimization"
    bl_parent_id = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)

        # Resolution controls
        col.prop(scene, "megasam_res_preset", text="Resolution")
        if scene.megasam_res_preset == 'CUSTOM':
            col.prop(scene, "megasam_target_width", text="Width (px)")
        else:
            col.label(text=f"Processing Width: {scene.megasam_target_width} px", icon='DRIVER')

        col.separator()
        # Depth Backbone
        col.prop(scene, "megasam_depth_encoder", text="Depth Model")

        col.separator()
        # CVD / Tracking-only mode
        col.prop(scene, "megasam_skip_cvd", text="Tracking Only (Fast / Skip CVD)")
        if not scene.megasam_skip_cvd:
            col.prop(scene, "megasam_cvd_iters", text="CVD Iterations")
        else:
            col.label(text="Skips RAFT optical flow & dense CVD optimization", icon='INFO')

        layout.separator()

        # Batch Processing
        col = layout.column(align=True)
        col.prop(scene, "megasam_use_batch", text="Enable Batching (Low VRAM)")
        if scene.megasam_use_batch:
            col.prop(scene, "megasam_batch_size", text="Batch Size")
            total_f = scene.megasam_video_frames
            if scene.megasam_use_timeline_range and scene.frame_end > scene.frame_start:
                total_f = scene.frame_end - scene.frame_start + 1
            elif scene.megasam_use_frame_range and scene.megasam_end_frame > scene.megasam_start_frame:
                total_f = scene.megasam_end_frame - scene.megasam_start_frame

            if total_f > 0 and scene.megasam_batch_size > 0:
                batch_count = (total_f + scene.megasam_batch_size - 1) // scene.megasam_batch_size
                col.label(text=f"Splits into ~{batch_count} batches (auto-stitched)", icon='FORCE_CHARGE')

        layout.separator()
        layout.prop(scene, "megasam_low_vram", text="CUDA Memory Allocator Opt")

        # Advanced SLAM (DROID) Settings
        layout.separator()
        adv_box = layout.box()
        adv_box.label(text="Advanced SLAM (DROID) Settings", icon='TRACKER')
        acol = adv_box.column(align=True)
        acol.prop(scene, "megasam_stride", text="Tracking Stride")
        if scene.megasam_stride > 1:
            acol.label(text=f"Tracks every {scene.megasam_stride} frames (50-65% faster)", icon='INFO')
        acol.prop(scene, "megasam_frontend_iters", text="Frontend BA Iters")
        acol.prop(scene, "megasam_backend_iters", text="Backend BA Iters")
        acol.prop(scene, "megasam_buffer", text="Keyframe Buffer")
        acol.prop(scene, "megasam_keyframe_thresh", text="Keyframe Thresh")


class VIEW3D_PT_megasam_import_settings(bpy.types.Panel):
    """Sub-panel for Point Cloud visualization and import options."""
    bl_label = "Import & Point Cloud Settings"
    bl_idname = "VIEW3D_PT_megasam_import_settings"
    bl_parent_id = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.prop(scene, "megasam_point_radius", text="Point Radius", icon='PARTICLE_POINT')
        col.prop(scene, "megasam_import_frame_step", text="Frame Step")
        col.prop(scene, "megasam_import_pixel_step", text="Pixel Step")

        col.separator()
        col.prop(scene, "megasam_import_filter_mode", text="Filter Mode")
        if scene.megasam_import_filter_mode != 'ALL':
            col.prop(scene, "megasam_import_motion_threshold", text="Motion Thresh")

        col.separator()
        col.prop(scene, "megasam_import_animate", text="Animate Point Cloud")
        col.prop(scene, "megasam_import_clean_scene", text="Clean Previous Import")
        col.prop(scene, "megasam_import_parent_to_empty", text="Parent to Empty")


class VIEW3D_PT_megasam_mesh(bpy.types.Panel):
    """Sub-panel for generating 3D textured mesh from point cloud."""
    bl_label = "Mesh Generation (3D Surface)"
    bl_idname = "VIEW3D_PT_megasam_mesh"
    bl_parent_id = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.prop(scene, "megasam_mesh_use_import_settings", text="Use Import Filter")
        col.prop(scene, "megasam_mesh_threshold", text="Mesh Threshold")
        col.prop(scene, "megasam_mesh_upscale", text="Mesh Upscale")
        col.prop(scene, "megasam_mesh_smooth", text="Mesh Smooth")
        col.prop(scene, "megasam_mesh_output_dir", text="Output Folder")

        layout.separator()
        layout.operator("megasam.generate_mesh", icon='MOD_MESHDEFORM', text="Generate Mesh (Current Frame)")


class VIEW3D_PT_megasam_vse(bpy.types.Panel):
    """Sub-panel for Video Sequence Editor configuration."""
    bl_label = "VSE Settings"
    bl_idname = "VIEW3D_PT_megasam_vse"
    bl_parent_id = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.prop(scene, "megasam_vse_view_mode", text="View Mode")
        col.prop(scene, "megasam_vse_clear_existing", text="Clear Strips before Load")


class VIEW3D_PT_megasam_environment(bpy.types.Panel):
    """Sub-panel for Python executable and MegaSam repository paths."""
    bl_label = "Environment & Paths"
    bl_idname = "VIEW3D_PT_megasam_environment"
    bl_parent_id = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        addon_name = __package__.split('.')[0] if __package__ else "CEB_MegaSam"
        addon_entry = context.preferences.addons.get(addon_name)
        prefs = addon_entry.preferences if addon_entry else None

        col = layout.column(align=True)
        if prefs:
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
        else:
            auto_py = get_python_path(context)
            if auto_py and os.path.exists(auto_py):
                col.label(text=f"Python: {os.path.basename(os.path.dirname(auto_py))}\\python.exe", icon='CHECKMARK')
            else:
                col.label(text="Python not found", icon='ERROR')

            auto_root = get_megasam_root(context)
            if auto_root and os.path.exists(auto_root):
                col.label(text=f"MegaSam: {os.path.basename(auto_root)}", icon='CHECKMARK')
            else:
                col.label(text="MegaSam root not found", icon='ERROR')

        col.separator()
        enc = context.scene.megasam_depth_encoder if hasattr(context.scene, "megasam_depth_encoder") else "vits"
        depth_ckpt = get_depth_anything_ckpt(context, enc)
        if depth_ckpt and os.path.exists(depth_ckpt):
            col.label(text=f"Depth CKPT: {os.path.basename(depth_ckpt)}", icon='CHECKMARK')
        else:
            col.label(text="Depth CKPT not found", icon='ERROR')


classes = (
    VIEW3D_PT_megasam_panel,
    VIEW3D_PT_megasam_optimization,
    VIEW3D_PT_megasam_import_settings,
    VIEW3D_PT_megasam_mesh,
    VIEW3D_PT_megasam_vse,
    VIEW3D_PT_megasam_environment,
)

def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except Exception:
            pass

def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
