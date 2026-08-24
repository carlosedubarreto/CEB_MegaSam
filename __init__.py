import bpy
from . import operators
from . import panel

bl_info = {
    "name": "CEB MegaSam",
    "author": "Carlos Barreto, Antigravity",
    "version": (1, 0),
    "blender": (4, 4, 3),
    "location": "View3D > N-Panel > CEB",
    "description": "Run MegaSam pipeline from Blender",
    "category": "Pipeline",
}

class MegaSamPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    python_path: bpy.props.StringProperty(
        name="Python Path",
        description="Path to the portable Python executable",
        default="",
        subtype='FILE_PATH'
    )
    
    megasam_root_path: bpy.props.StringProperty(
        name="MegaSam Root",
        description="Root folder of the MegaSam project",
        default="",
        subtype='DIR_PATH'
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "python_path")
        layout.prop(self, "megasam_root_path")

def get_result_items(self, context):
    import os
    import glob
    import re
    
    addon_name = __package__.split('.')[0]
    prefs = context.preferences.addons[addon_name].preferences
    root = prefs.megasam_root_path
    if not root or not os.path.exists(root):
        return [('NONE', 'Root not set', 'Check preferences')]
    
    # Try multiple common locations
    search_paths = [
        os.path.join(root, "outputs_cvd"),
        os.path.join(root, "cvd_opt", "outputs_cvd"),
        os.path.join(root, "mega-sam", "cvd_opt", "outputs_cvd")
    ]
    
    output_dir = ""
    for p in search_paths:
        if os.path.exists(p):
            output_dir = p
            break
            
    if not output_dir:
        return [('NONE', 'Folder outputs_cvd not found', 'Check project root')]
        
    files = glob.glob(os.path.join(output_dir, "*.npz"))
    items = []
    
    filter_mode = context.scene.megasam_output_filter
    # Pattern for batches like _b000, _b001 etc
    batch_pattern = re.compile(r'_b\d{3}')
    
    for f in files:
        basename = os.path.basename(f)
        is_batch = bool(batch_pattern.search(basename))
        
        if filter_mode == 'BATCH' and is_batch:
            items.append((f, basename, f))
        elif filter_mode == 'FINAL' and not is_batch:
            items.append((f, basename, f))
            
    if not items:
        return [('NONE', 'No matches found', 'Check folder content')]
        
    return sorted(items, key=lambda x: x[1])

def update_video_info(self, context):
    import os
    path = bpy.path.abspath(self.megasam_video_path)
    if not path or not os.path.exists(path):
        self.megasam_video_width = 0
        self.megasam_video_height = 0
        self.megasam_video_frames = 0
        return

    # Auto-populate scene name if empty
    if not self.megasam_scene_name:
        self.megasam_scene_name = os.path.splitext(os.path.basename(path))[0]

    try:
        # Load as movieclip to get info
        clip = bpy.data.movieclips.load(path)
        self.megasam_video_width = clip.size[0]
        self.megasam_video_height = clip.size[1]
        self.megasam_video_frames = clip.frame_duration
        # Remove the clip from memory
        bpy.data.movieclips.remove(clip)
    except Exception as e:
        print(f"Error reading video info: {e}")

def update_point_radius(self, context):
    radius = context.scene.megasam_point_radius
    for group in bpy.data.node_groups:
        if group.name.startswith("PointGroup_"):
            for node in group.nodes:
                if node.type == 'MESH_TO_POINTS':
                    node.inputs['Radius'].default_value = radius

def register():
    bpy.utils.register_class(MegaSamPreferences)
    
    bpy.types.Scene.megasam_point_radius = bpy.props.FloatProperty(
        name="Point Radius",
        description="Size of the points in the 3D viewport",
        default=0.01,
        min=0.0001,
        max=1.0,
        precision=4,
        update=update_point_radius
    )
    
    bpy.types.Scene.megasam_video_path = bpy.props.StringProperty(
        name="Video File",
        description="Select video file for processing",
        default="",
        subtype='FILE_PATH',
        update=update_video_info
    )
    bpy.types.Scene.megasam_scene_name = bpy.props.StringProperty(
        name="Scene Name",
        description="Name for the processing output (defaults to video name)",
        default=""
    )
    bpy.types.Scene.megasam_video_width = bpy.props.IntProperty(name="Width", default=0)
    bpy.types.Scene.megasam_video_height = bpy.props.IntProperty(name="Height", default=0)
    bpy.types.Scene.megasam_video_frames = bpy.props.IntProperty(name="Total Frames", default=0)
    
    bpy.types.Scene.megasam_use_batch = bpy.props.BoolProperty(
        name="Use Batch Processing",
        description="Enable splitting the video into batches",
        default=False
    )
    bpy.types.Scene.megasam_batch_size = bpy.props.IntProperty(
        name="Batch Size",
        description="Number of frames per batch",
        default=100,
        min=1
    )
    
    bpy.types.Scene.megasam_output_filter = bpy.props.EnumProperty(
        name="Show Files",
        items=[
            ('FINAL', 'Final Stitched', 'Show stitched results'),
            ('BATCH', 'Intermediate Batches', 'Show individual batch files')
        ],
        default='FINAL'
    )
    
    bpy.types.Scene.megasam_result_list = bpy.props.EnumProperty(
        name="Results",
        items=get_result_items
    )
    
    # Import Settings
    bpy.types.Scene.megasam_import_frame_step = bpy.props.IntProperty(name="Frame Step", default=1, min=1)
    # Changed default to 6 to match original script
    bpy.types.Scene.megasam_import_pixel_step = bpy.props.IntProperty(name="Pixel Step", default=6, min=1)
    bpy.types.Scene.megasam_import_filter_mode = bpy.props.EnumProperty(
        name="Filter Mode",
        items=[
            ('STATIC', 'Static Only', 'Show static parts'),
            ('DYNAMIC', 'Dynamic Only', 'Show moving parts'),
            ('ALL', 'Show All', 'Show everything')
        ],
        default='ALL'
    )
    bpy.types.Scene.megasam_import_motion_threshold = bpy.props.FloatProperty(name="Motion Threshold", default=0.1, min=0.0)
    bpy.types.Scene.megasam_import_animate = bpy.props.BoolProperty(name="Animate", default=True)
    bpy.types.Scene.megasam_import_clean_scene = bpy.props.BoolProperty(
        name="Clean Scene",
        description="Remove previous point clouds and camera before importing",
        default=True
    )
    
    # Mesh Generation Settings
    bpy.types.Scene.megasam_mesh_use_import_settings = bpy.props.BoolProperty(
        name="Use Import Filter",
        description="Use the same Filter Mode and Motion Threshold as the Point Cloud import",
        default=True
    )
    bpy.types.Scene.megasam_mesh_threshold = bpy.props.FloatProperty(
        name="Mesh Threshold",
        description="Relative edge threshold (Lower = cleaner gaps)",
        default=0.05,
        min=0.0,
        max=0.5
    )
    bpy.types.Scene.megasam_mesh_upscale = bpy.props.FloatProperty(
        name="Mesh Upscale",
        description="Resolution multiplier (1.5-2.0 = more detail)",
        default=1.0,
        min=0.5,
        max=4.0
    )
    bpy.types.Scene.megasam_mesh_smooth = bpy.props.FloatProperty(
        name="Mesh Smooth",
        description="Gaussian depth smoothing sigma",
        default=0.0,
        min=0.0,
        max=5.0
    )
    
    bpy.types.Scene.megasam_mesh_output_dir = bpy.props.StringProperty(
        name="Mesh Output Folder",
        description="Folder where generated meshes will be saved",
        default="",
        subtype='DIR_PATH'
    )
    
    operators.register()
    panel.register()

def unregister():
    operators.unregister()
    panel.unregister()
    
    del bpy.types.Scene.megasam_point_radius
    del bpy.types.Scene.megasam_video_path
    del bpy.types.Scene.megasam_scene_name
    del bpy.types.Scene.megasam_video_width
    del bpy.types.Scene.megasam_video_height
    del bpy.types.Scene.megasam_video_frames
    del bpy.types.Scene.megasam_use_batch
    del bpy.types.Scene.megasam_batch_size
    del bpy.types.Scene.megasam_output_filter
    del bpy.types.Scene.megasam_result_list
    del bpy.types.Scene.megasam_import_frame_step
    del bpy.types.Scene.megasam_import_pixel_step
    del bpy.types.Scene.megasam_import_filter_mode
    del bpy.types.Scene.megasam_import_motion_threshold
    del bpy.types.Scene.megasam_import_animate
    del bpy.types.Scene.megasam_import_clean_scene
    del bpy.types.Scene.megasam_mesh_use_import_settings
    del bpy.types.Scene.megasam_mesh_threshold
    del bpy.types.Scene.megasam_mesh_upscale
    del bpy.types.Scene.megasam_mesh_smooth
    del bpy.types.Scene.megasam_mesh_output_dir
    
    bpy.utils.unregister_class(MegaSamPreferences)

if __name__ == "__main__":
    register()
