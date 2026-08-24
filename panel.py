import bpy

class VIEW3D_PT_megasam_panel(bpy.types.Panel):
    bl_label = "CEB MegaSam"
    bl_idname = "VIEW3D_PT_megasam_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'CEB'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Access the addon preferences
        addon_name = __package__.split('.')[0] # Get the top level package name
        prefs = context.preferences.addons[addon_name].preferences

        # Environment Section
        box = layout.box()
        box.label(text="Environment", icon='PROPERTIES')
        box.prop(prefs, "python_path", text="Python Path")
        box.prop(prefs, "megasam_root_path", text="MegaSam Root")
        
        layout.separator()
        
        # Processing Section
        box = layout.box()
        box.label(text="Processing")
        box.prop(scene, "megasam_video_path", text="Video File")
        box.prop(scene, "megasam_scene_name", text="Scene Name")
        
        if scene.megasam_video_frames > 0:
            row = box.column(align=True)
            row.label(text=f"Resolution: {scene.megasam_video_width} x {scene.megasam_video_height}", icon='SCENE_DATA')
            row.label(text=f"Frames: {scene.megasam_video_frames}", icon='IMAGE_DATA')
        
        row = box.row()
        row.prop(scene, "megasam_use_batch", text="Enable Batching")
        if scene.megasam_use_batch:
            row.prop(scene, "megasam_batch_size", text="Size")
        
        layout.separator()
        
        # Run Button
        layout.operator("megasam.run_pipeline", icon='PLAY', text="Run Pipeline")
        
        layout.separator()
        
        # Results Section
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
        
        box.operator("megasam.import_result", icon='IMPORT', text="Import Selected Result")
        
        # Mesh Generation
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
    bpy.utils.register_class(VIEW3D_PT_megasam_panel)

def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_megasam_panel)
