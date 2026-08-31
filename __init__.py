import os
import glob
import re
import bpy
from .utils import get_python_path, get_megasam_root
from . import operators
from . import panel

bl_info = {
    "name": "CEB MegaSam",
    "author": "Carlos Barreto, Antigravity",
    "version": (1, 2),
    "blender": (4, 4, 3),
    "location": "View3D > N-Panel > CEB",
    "description": "Run MegaSam pipeline from Blender with auto-detection, low-VRAM, and speed controls",
    "category": "Pipeline",
}

### 1.2 WIP - pulei o 1.1 para poder coincidir com o portable
## OK - Nao precisar colocar path pro megasam, apenas para python portable
## OK - processar apenas parte do video
## OK - scaled down the video for faster/less VRAM processing
## OK - load video starting at the correct frame, when running a trimmed frame range
## OK- formas de melhorar performance

## OK - colocar opçao para incluir empty com nome do video que pasa a ser parent dos objetos carregados
## OK - tirar padrao de limpar cena
## OK - colocar como padrao para mostrar nome do empty
## OK - por padrao cvd desativado

## - reduzir consumo de memoria
## - trocar para os outputs serem na pasta raiz do portable
## - reorganizar visual
## - oficial com novo portable python torch 2.10 CUDA 12.8

class MegaSamPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    python_path: bpy.props.StringProperty(
        name="Python Path",
        description="Path to Python executable (leave blank to auto-detect portable Python)",
        default="",
        subtype='FILE_PATH'
    )
    
    megasam_root_path: bpy.props.StringProperty(
        name="MegaSam Root",
        description="Root folder of MegaSam (leave blank to auto-detect in same parent folder as portable Python)",
        default="",
        subtype='DIR_PATH'
    )

    def draw(self, context):
        layout = self.layout
        
        # Python setting
        col = layout.column(align=True)
        col.prop(self, "python_path")
        detected_py = get_python_path(context)
        if not self.python_path:
            if detected_py and os.path.exists(detected_py):
                col.label(text=f"Auto-detected: {detected_py}", icon='CHECKMARK')
            else:
                col.label(text="Portable Python not found automatically", icon='ERROR')
        
        layout.separator()
        
        # MegaSam Root setting
        col = layout.column(align=True)
        col.prop(self, "megasam_root_path")
        detected_root = get_megasam_root(context)
        if not self.megasam_root_path:
            if detected_root and os.path.exists(detected_root):
                col.label(text=f"Auto-detected: {detected_root}", icon='CHECKMARK')
            else:
                col.label(text="MegaSam root folder not found automatically", icon='ERROR')

def get_result_items(self, context):
    root = get_megasam_root(context)
    if not root or not os.path.exists(root):
        return [('NONE', 'Root not set / found', 'Check preferences')]
    
    # Try multiple common locations
    search_paths = [
        os.path.join(root, "outputs_cvd"),
        os.path.join(root, "cvd_opt", "outputs_cvd"),
        os.path.join(root, "mega-sam", "cvd_opt", "outputs_cvd"),
        os.path.join(root, "mega-sam", "outputs_cvd"),
        os.path.join(root, "outputs"),
        os.path.join(root, "mega-sam", "outputs")
    ]
    
    output_dirs = []
    for p in search_paths:
        if os.path.exists(p) and p not in output_dirs:
            output_dirs.append(p)
            
    if not output_dirs:
        return [('NONE', 'Outputs folder not found', 'Check project root')]
        
    files = []
    for d in output_dirs:
        files.extend(glob.glob(os.path.join(d, "*.npz")))
        
    items = []
    filter_mode = context.scene.megasam_output_filter
    batch_pattern = re.compile(r'_b\d{3}')
    seen_names = set()
    
    for f in files:
        basename = os.path.basename(f)
        if basename in seen_names:
            continue
        seen_names.add(basename)
        
        is_batch = bool(batch_pattern.search(basename))
        
        if filter_mode == 'BATCH' and is_batch:
            items.append((f, basename, f))
        elif filter_mode == 'FINAL' and not is_batch:
            items.append((f, basename, f))
            
    if not items:
        return [('NONE', 'No matches found', 'Check folder content')]
        
    return sorted(items, key=lambda x: x[1])

def update_video_info(self, context):
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
        if self.megasam_end_frame == 0:
            self.megasam_end_frame = clip.frame_duration
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

def update_res_preset(self, context):
    if self.megasam_res_preset != 'CUSTOM':
        try:
            self.megasam_target_width = int(self.megasam_res_preset)
        except ValueError:
            pass

def register():
    try:
        bpy.utils.register_class(MegaSamPreferences)
    except ValueError:
        pass
    
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
    
    # Performance & Resolution settings
    bpy.types.Scene.megasam_res_preset = bpy.props.EnumProperty(
        name="Resolution Preset",
        description="Target width for processing. Lower resolution drastically speeds up all pipeline stages and reduces VRAM",
        items=[
            ('384', '384p (Fastest / Low VRAM)', 'Best for 6GB-8GB GPUs or rapid testing (3x-5x speedup)'),
            ('480', '480p (Fast / Balanced)', 'Good balance of speed and depth quality'),
            ('540', '540p (Default Balanced)', 'Standard MegaSam resolution'),
            ('720', '720p (High Detail)', 'Requires 12GB+ VRAM, slower processing'),
            ('CUSTOM', 'Custom Width', 'Manually specify custom target width')
        ],
        default='540',
        update=update_res_preset
    )
    bpy.types.Scene.megasam_target_width = bpy.props.IntProperty(
        name="Target Width",
        description="Width to resize frames to before processing (e.g. 384, 480, 540, 720). Lower = faster & less VRAM",
        default=540,
        min=128,
        max=3840
    )
    bpy.types.Scene.megasam_use_frame_range = bpy.props.BoolProperty(
        name="Process Frame Range",
        description="Process only a specific range of frames instead of the entire video",
        default=False
    )
    bpy.types.Scene.megasam_start_frame = bpy.props.IntProperty(
        name="Start Frame",
        description="First frame index to process",
        default=0,
        min=0
    )
    bpy.types.Scene.megasam_end_frame = bpy.props.IntProperty(
        name="End Frame",
        description="Last frame index to process (0 = process until end of video)",
        default=0,
        min=0
    )
    bpy.types.Scene.megasam_low_vram = bpy.props.BoolProperty(
        name="VRAM Allocator Optimization",
        description="Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent CUDA memory fragmentation and OOM errors",
        default=True
    )
    
    # 1. SLAM Backend Optimization Properties
    bpy.types.Scene.megasam_stride = bpy.props.IntProperty(
        name="Tracking Frame Stride",
        description="Frame stride for camera tracking. Setting stride 2 or 3 cuts tracking runtime by 50-65% on 30/60fps footage",
        default=1,
        min=1,
        max=10
    )
    bpy.types.Scene.megasam_frontend_iters = bpy.props.IntProperty(
        name="Frontend BA Iterations",
        description="Local bundle adjustment optimization iterations per frame (default 4). Lowering to 2 or 3 speeds up tracking",
        default=4,
        min=1,
        max=10
    )
    bpy.types.Scene.megasam_backend_iters = bpy.props.IntProperty(
        name="Backend BA Iterations",
        description="Global bundle adjustment optimization iterations (default 12). Lowering to 6 or 8 speeds up final tracking",
        default=12,
        min=1,
        max=24
    )
    bpy.types.Scene.megasam_buffer = bpy.props.IntProperty(
        name="Keyframe Buffer",
        description="Keyframe buffer size for tracking graph updates (default 1024). Lowering to 512 reduces VRAM usage",
        default=1024,
        min=128,
        max=2048
    )
    bpy.types.Scene.megasam_keyframe_thresh = bpy.props.FloatProperty(
        name="Keyframe Threshold",
        description="Motion threshold for keyframe creation (default 2.0). Higher values produce fewer keyframes and faster tracking",
        default=2.0,
        min=0.1,
        max=10.0
    )

    # 2. Monocular Depth Backbone (Depth-Anything)
    bpy.types.Scene.megasam_depth_encoder = bpy.props.EnumProperty(
        name="Depth Backbone",
        description="Depth-Anything model architecture. ViT-Small runs 4-5x faster with minimal accuracy impact on road scenes",
        items=[
            ('vitl', 'ViT-Large (Best Quality)', 'Highest depth accuracy, default model'),
            ('vitb', 'ViT-Base (Fast)', 'Faster execution with balanced quality'),
            ('vits', 'ViT-Small (Ultra Fast)', 'Runs 4-5x faster with minimal degradation')
        ],
        default='vitl'
    )

    # 3. CVD Optimization Controls
    bpy.types.Scene.megasam_skip_cvd = bpy.props.BoolProperty(
        name="Camera Tracking Only (Skip CVD)",
        description="Skip RAFT optical flow and CVD dense depth optimization. Ideal when you only need camera motion/poses quickly",
        default=True
    )
    bpy.types.Scene.megasam_cvd_iters = bpy.props.IntProperty(
        name="CVD Iterations",
        description="Number of CVD depth optimization iterations (default 400). Lowering to 100 or 200 cuts CVD time significantly",
        default=400,
        min=10,
        max=1000
    )
    
    # UI foldout toggle
    bpy.types.Scene.megasam_show_advanced = bpy.props.BoolProperty(
        name="Show Advanced Speed Options",
        description="Toggle advanced SLAM backend and depth optimization parameters",
        default=False
    )

    bpy.types.Scene.megasam_use_batch = bpy.props.BoolProperty(
        name="Use Batch Processing",
        description="Enable splitting the video into batches. Highly recommended for long videos to keep VRAM usage low and avoid OOM",
        default=False
    )
    bpy.types.Scene.megasam_batch_size = bpy.props.IntProperty(
        name="Batch Size",
        description="Number of frames per batch (50-100 recommended for low VRAM)",
        default=100,
        min=10
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
        default=False
    )
    bpy.types.Scene.megasam_import_parent_to_empty = bpy.props.BoolProperty(
        name="Parent to Empty",
        description="Create an Empty object named after the scene to parent the imported camera and point cloud",
        default=True
    )
    
    # VSE Settings
    bpy.types.Scene.megasam_vse_view_mode = bpy.props.EnumProperty(
        name="VSE View Mode",
        description="Display mode for Video Sequence Editor",
        items=[
            ('PREVIEW', 'Preview', 'Show video preview only'),
            ('SEQUENCER', 'Sequencer', 'Show sequence timeline strips only'),
            ('SEQUENCER_PREVIEW', 'Sequencer & Preview', 'Show both sequence timeline and video preview')
        ],
        default='PREVIEW'
    )
    bpy.types.Scene.megasam_vse_clear_existing = bpy.props.BoolProperty(
        name="Clear Existing VSE Strips",
        description="Clear existing video strips in VSE before loading selected video",
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
    
    props = [
        "megasam_point_radius", "megasam_video_path", "megasam_scene_name",
        "megasam_video_width", "megasam_video_height", "megasam_video_frames",
        "megasam_res_preset", "megasam_target_width", "megasam_use_frame_range",
        "megasam_start_frame", "megasam_end_frame", "megasam_low_vram",
        "megasam_stride", "megasam_frontend_iters", "megasam_backend_iters",
        "megasam_buffer", "megasam_keyframe_thresh", "megasam_depth_encoder",
        "megasam_skip_cvd", "megasam_cvd_iters", "megasam_show_advanced",
        "megasam_use_batch", "megasam_batch_size", "megasam_output_filter",
        "megasam_result_list", "megasam_import_frame_step", "megasam_import_pixel_step",
        "megasam_import_filter_mode", "megasam_import_motion_threshold",
        "megasam_import_animate", "megasam_import_clean_scene", "megasam_import_parent_to_empty",
        "megasam_vse_view_mode", "megasam_vse_clear_existing",
        "megasam_mesh_use_import_settings", "megasam_mesh_threshold",
        "megasam_mesh_upscale", "megasam_mesh_smooth", "megasam_mesh_output_dir"
    ]
    for p in props:
        if hasattr(bpy.types.Scene, p):
            try:
                delattr(bpy.types.Scene, p)
            except Exception:
                pass
    
    try:
        bpy.utils.unregister_class(MegaSamPreferences)
    except Exception:
        pass

if __name__ == "__main__":
    register()
