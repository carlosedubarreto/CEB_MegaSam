import bpy
import os
import subprocess
import numpy as np
import mathutils
from .utils import get_python_path, get_megasam_root, get_portable_root, get_depth_anything_ckpt, get_unique_scene_name

class MEGASAM_OT_set_range_from_timeline(bpy.types.Operator):
    bl_idname = "megasam.set_range_from_timeline"
    bl_label = "Use Timeline Range"
    bl_description = "Copy current Blender timeline Start and End frames to the MegaSam processing range"

    def execute(self, context):
        scene = context.scene
        scene.megasam_start_frame = max(0, scene.frame_start)
        scene.megasam_end_frame = max(scene.megasam_start_frame + 1, scene.frame_end)
        scene.megasam_use_frame_range = True
        self.report({'INFO'}, f"Frame range set to {scene.megasam_start_frame} - {scene.megasam_end_frame}")
        return {'FINISHED'}

class MEGASAM_OT_run_pipeline(bpy.types.Operator):
    bl_idname = "megasam.run_pipeline"
    bl_label = "Run MegaSam Pipeline"
    bl_description = "Execute the run_pipeline.py script in a new console"

    def execute(self, context):
        scene = context.scene
        
        # Resolve Python and MegaSam Root automatically if not set
        python_exe = get_python_path(context)
        repo_root = get_megasam_root(context)
        video_path = bpy.path.abspath(scene.megasam_video_path)
        
        # Validation and auto-correction
        if not python_exe or not os.path.exists(python_exe):
            self.report({'ERROR'}, f"Python executable not found. Please verify Python Path in Preferences. (Checked: '{python_exe}')")
            return {'CANCELLED'}
            
        if os.path.isdir(python_exe):
            potential_exe = os.path.join(python_exe, "python.exe")
            if os.path.exists(potential_exe):
                python_exe = potential_exe
            else:
                self.report({'ERROR'}, f"Python directory does not contain 'python.exe': {python_exe}")
                return {'CANCELLED'}
        
        if not os.path.isfile(python_exe):
            self.report({'ERROR'}, f"Python Path is not a file: {python_exe}")
            return {'CANCELLED'}
            
        if not repo_root or not os.path.exists(repo_root):
            self.report({'ERROR'}, f"MegaSam Root folder not found. Please check preferences or ensure 'mega-sam' is alongside portable Python.")
            return {'CANCELLED'}
            
        # Try to find run_pipeline.py
        potential_scripts = [
            os.path.join(repo_root, "run_pipeline.py"),
            os.path.join(repo_root, "mega-sam", "run_pipeline.py")
        ]
        
        script_path = ""
        for p in potential_scripts:
            if os.path.exists(p):
                script_path = p
                break
                
        if not script_path:
            self.report({'ERROR'}, f"Could not find 'run_pipeline.py' in {repo_root}")
            return {'CANCELLED'}
        
        if not video_path or not os.path.exists(video_path):
            self.report({'ERROR'}, f"Video file path is invalid or file does not exist: {video_path}")
            return {'CANCELLED'}
            
        portable_root = get_portable_root(context)
        output_root = os.path.join(portable_root, "outputs")
        
        # Use user-provided scene name or fallback to video filename
        raw_scene_name = scene.megasam_scene_name.strip()
        if not raw_scene_name:
            raw_scene_name = os.path.splitext(os.path.basename(video_path))[0]
            
        # Enforce unique project/scene name (e.g. chase -> chase_001 -> chase_002)
        scene_name = get_unique_scene_name(output_root, raw_scene_name)
        scene.megasam_scene_name = scene_name
        
        # Build command with performance & low-VRAM arguments
        command = [
            python_exe,
            script_path,
            "-v", video_path,
            "-s", scene_name,
            "-o", output_root,
            "--portable_root", portable_root,
            "-w", str(scene.megasam_target_width),
            "--depth_encoder", scene.megasam_depth_encoder,
            "--stride", str(scene.megasam_stride),
            "--buffer", str(scene.megasam_buffer),
            "--keyframe_thresh", str(scene.megasam_keyframe_thresh),
            "--frontend_iters", str(scene.megasam_frontend_iters),
            "--backend_iters", str(scene.megasam_backend_iters)
        ]

        # Explicitly pass DepthAnything checkpoint if resolved from DepthAnything_CKPT
        depth_ckpt = get_depth_anything_ckpt(context, scene.megasam_depth_encoder)
        if depth_ckpt:
            command.extend(["--depth_anything_weights", depth_ckpt])
        
        if scene.megasam_skip_cvd:
            command.append("--skip_cvd")
        else:
            command.extend(["--cvd_iters", str(scene.megasam_cvd_iters)])
        
        if scene.megasam_use_timeline_range:
            t_start = max(0, scene.frame_start)
            t_end = max(t_start, scene.frame_end)
            # Blender timeline is 1-based and inclusive [t_start, t_end].
            # When mapping to 0-based video frames [v_start, v_end):
            # If timeline starts at >= 1 (e.g. 1..200), video frames are [t_start - 1, t_end) -> [0, 200), exactly 200 frames.
            if scene.frame_start >= 1:
                v_start = scene.frame_start - 1
                v_end = scene.frame_end
            else:
                v_start = t_start
                v_end = t_end
            command.extend(["--start_frame", str(v_start)])
            if v_end >= v_start:
                command.extend(["--end_frame", str(v_end)])
        elif scene.megasam_use_frame_range:
            v_start = max(0, scene.megasam_start_frame)
            v_end = max(v_start, scene.megasam_end_frame)
            command.extend(["--start_frame", str(v_start)])
            if v_end >= v_start:
                command.extend(["--end_frame", str(v_end)])
        
        if scene.megasam_use_batch:
            command.extend(["--batch_size", str(scene.megasam_batch_size)])

        # Setup environment with PyTorch CUDA memory allocator optimization
        env = os.environ.copy()
        if scene.megasam_low_vram:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        print(f"Executing: {' '.join(command)}")
        print(f"Working Directory: {os.path.dirname(script_path)}")
        
        try:
            script_dir = os.path.dirname(script_path)
            
            # On Windows, CREATE_NEW_CONSOLE (0x00000010) opens a separate terminal window
            if os.name == 'nt':
                subprocess.Popen(command, cwd=script_dir, env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(command, cwd=script_dir, env=env)
                
            self.report({'INFO'}, f"MegaSam Pipeline launched for '{scene_name}' (Target Width: {scene.megasam_target_width}px)")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch pipeline: {str(e)}")
            return {'CANCELLED'}

class MEGASAM_OT_import_result(bpy.types.Operator):
    bl_idname = "megasam.import_result"
    bl_label = "Import Selected Result"
    bl_description = "Run the animation import script on the selected NPZ file"

    def execute(self, context):
        scene = context.scene
        
        selected_path = scene.megasam_result_list
        if not selected_path or selected_path == 'NONE':
            self.report({'ERROR'}, "No valid result selected")
            return {'CANCELLED'}
            
        try:
            # Extract scene name from filename
            filename = os.path.basename(selected_path)
            scene_name = filename.replace("_sgd_cvd_hr.npz", "").replace("_droid.npz", "").replace(".npz", "")
            if scene.megasam_use_timeline_range:
                start_frame = max(0, scene.frame_start)
                user_end_frame = scene.frame_end
            elif scene.megasam_use_frame_range:
                start_frame = scene.megasam_start_frame
                user_end_frame = scene.megasam_end_frame if scene.megasam_end_frame > 0 else None
            else:
                start_frame = 0
                user_end_frame = None

            # 1. Load Data
            data = self.load_cvd_data(selected_path)
            
            # Determine stride: check npz file first, then fall back to scene property
            file_stride = data.get('stride')
            if file_stride is not None and file_stride >= 1:
                stride = file_stride
            else:
                stride = scene.megasam_stride if scene.megasam_stride >= 1 else 1

            # 2. Cleanup old objects (if requested)
            if scene.megasam_import_clean_scene:
                self.cleanup_scene(scene, scene_name=scene_name)
            
            # 3. Setup Camera
            cam_obj = self.setup_camera(data, context, scene_name, start_frame=start_frame, stride=stride, user_end_frame=user_end_frame)
            
            # 4. Create Point Cloud
            pc_obj = self.create_animated_point_cloud(data, context, scene_name, start_frame=start_frame, stride=stride, user_end_frame=user_end_frame)
            
            # 5. Parent to Empty (if requested)
            if scene.megasam_import_parent_to_empty:
                empty_obj = bpy.data.objects.get(scene_name)
                if empty_obj is None or empty_obj.type != 'EMPTY':
                    empty_obj = bpy.data.objects.new(scene_name, None)
                    context.collection.objects.link(empty_obj)
                empty_obj.show_name = True
                
                if cam_obj:
                    cam_obj.parent = empty_obj
                    cam_obj.matrix_parent_inverse = empty_obj.matrix_world.inverted()
                if pc_obj:
                    pc_obj.parent = empty_obj
                    pc_obj.matrix_parent_inverse = empty_obj.matrix_world.inverted()
            
            num_f = data['images'].shape[0]
            calculated_end = start_frame + (num_f - 1) * stride
            end_f = min(calculated_end, user_end_frame) if (user_end_frame is not None and user_end_frame >= start_frame) else calculated_end
            self.report({'INFO'}, f"Imported: {os.path.basename(selected_path)} (Frames: {num_f}, Stride: {stride} -> Timeline: {start_frame}..{end_f})")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def load_cvd_data(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path)
        stride_val = int(data['stride']) if 'stride' in data else None
        return {
            'images': data['images'], 'depths': data['depths'], 
            'uncertainty': data.get('uncertainty', None),
            'intrinsic': data['intrinsic'], 'cam_c2w': data['cam_c2w'],
            'stride': stride_val
        }

    def cleanup_scene(self, scene, scene_name=None):
        to_delete = []
        for obj in bpy.data.objects:
            if obj.name.startswith("PC_") or obj.name.startswith("Cam_") or obj.name == "CVD_Camera":
                to_delete.append(obj)
            elif scene_name and obj.name == scene_name and obj.type == 'EMPTY':
                to_delete.append(obj)
        
        if to_delete:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in to_delete:
                obj.select_set(True)
            bpy.ops.object.delete()

    def get_world_transform(self):
        return mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')

    def setup_camera(self, data, context, scene_name, start_frame=0, stride=1, user_end_frame=None):
        world_transform = self.get_world_transform()
        cam_name = f"Cam_{scene_name}"
        cam_data = bpy.data.cameras.new(cam_name)
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
        context.collection.objects.link(cam_obj)
        
        K, images = data['intrinsic'], data['images']
        num_frames, H, W, _ = images.shape
        
        calculated_end = start_frame + (num_frames - 1) * stride
        effective_end = min(calculated_end, user_end_frame) if (user_end_frame is not None and user_end_frame >= start_frame) else calculated_end
        
        context.scene.render.resolution_x, context.scene.render.resolution_y = W, H
        context.scene.frame_start = start_frame
        context.scene.frame_end = effective_end
        
        cam_data.lens_unit = 'FOV'
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.angle = 2 * np.arctan(W / (2 * K[0, 0]))
        
        # Calculate optical center shift
        max_dim = max(W, H)
        cam_data.shift_x = -(W / 2.0 - K[0, 2]) / max_dim
        cam_data.shift_y = (H / 2.0 - K[1, 2]) / max_dim
        
        cv_to_blender_local = mathutils.Matrix(((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1)))
        for f in range(num_frames):
            timeline_frame = start_frame + f * stride
            if user_end_frame is not None and timeline_frame > user_end_frame:
                break
            c2w = mathutils.Matrix(data['cam_c2w'][f])
            cam_obj.matrix_world = world_transform @ c2w @ cv_to_blender_local
            cam_obj.keyframe_insert(data_path="location", frame=timeline_frame)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=timeline_frame)
        return cam_obj

    def create_animated_point_cloud(self, data, context, scene_name, start_frame=0, stride=1, user_end_frame=None):
        scene = context.scene
        world_transform = self.get_world_transform()
        
        depths, images, K, c2ws = data['depths'], data['images'], data['intrinsic'], data['cam_c2w']
        uncertainties = data['uncertainty']
        
        filter_mode = scene.megasam_import_filter_mode
        frame_step = scene.megasam_import_frame_step
        pixel_step = scene.megasam_import_pixel_step
        motion_threshold = scene.megasam_import_motion_threshold
        animate = scene.megasam_import_animate
        
        if uncertainties is None and filter_mode != 'ALL':
             self.report({'WARNING'}, "Uncertainty data missing. Filtering disabled.")
             filter_mode = 'ALL'
             
        num_frames, H, W = depths.shape
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        
        all_pts, all_cols, all_f_ids = [], [], []
        
        for f in range(0, num_frames, frame_step):
            timeline_frame = start_frame + f * stride
            if user_end_frame is not None and timeline_frame > user_end_frame:
                break
            depth, image, c2w = depths[f], images[f], mathutils.Matrix(c2ws[f])
            u, v = np.meshgrid(np.arange(0, W, pixel_step), np.arange(0, H, pixel_step))
            z = depth[v, u].flatten()
            mask = (z > 0.1) & (z < 50.0)
            
            if uncertainties is not None and filter_mode != "ALL":
                vals = uncertainties[f][v, u].flatten()
                mask &= (vals < motion_threshold) if filter_mode == "STATIC" else (vals > motion_threshold)
            
            if not np.any(mask): continue
            
            z, uu, vv = z[mask], u.flatten()[mask], v.flatten()[mask]
            pts_cam = np.stack([(uu-cx)*z/fx, (vv-cy)*z/fy, z, np.ones_like(z)], axis=-1)
            all_pts.append((pts_cam @ (world_transform @ c2w).transposed())[:, :3])
            all_cols.append(image[vv, uu] / 255.0)
            all_f_ids.append(np.full(len(z), timeline_frame, dtype=np.float32))

        if not all_pts:
            self.report({'WARNING'}, "No points generated with current settings.")
            return None

        obj_name = f"PC_{scene_name}_{filter_mode}"
        if animate:
            obj_name += "_Animated"

        mesh = bpy.data.meshes.new(obj_name)
        obj = bpy.data.objects.new(obj_name, mesh)
        context.collection.objects.link(obj)
        mesh.from_pydata(np.concatenate(all_pts).tolist(), [], [])
        
        # Add Color Attribute
        color_attr = mesh.attributes.new(name="Color", type='FLOAT_COLOR', domain='POINT')
        color_attr.data.foreach_set("color", np.concatenate([np.concatenate(all_cols), np.ones((len(mesh.vertices), 1))], axis=1).flatten())
        
        # Add Frame ID Attribute
        if animate:
            frame_attr = mesh.attributes.new(name="FrameID", type='FLOAT', domain='POINT')
            frame_attr.data.foreach_set("value", np.concatenate(all_f_ids))
        
        self.setup_point_visualization(obj, context, filter_mode, start_frame=start_frame, stride=stride)
        return obj

    def setup_point_visualization(self, obj, context, filter_mode, start_frame=0, stride=1):
        scene = context.scene
        frame_step = scene.megasam_import_frame_step
        effective_step = max(1, frame_step * stride)
        
        mat_name = f"PointColorMat_{filter_mode}"
        mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        out = nodes.new('ShaderNodeOutputMaterial'); bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        attr = nodes.new('ShaderNodeAttribute'); attr.attribute_name = "Color"
        mat.node_tree.links.new(attr.outputs['Color'], bsdf.inputs['Base Color'])
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        if not obj.data.materials: obj.data.materials.append(mat)

        mod = obj.modifiers.new(name="PointVis", type='NODES')
        group = bpy.data.node_groups.new(name=f"PointGroup_{filter_mode}", type='GeometryNodeTree')
        mod.node_group = group
        in_node = group.nodes.new('NodeGroupInput'); out_node = group.nodes.new('NodeGroupOutput')
        
        # Geometry Nodes Interface (Blender 4.0+)
        if hasattr(group, "interface"):
            group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
            group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        else:
            group.inputs.new('NodeSocketGeometry', "Geometry")
            group.outputs.new('NodeSocketGeometry', "Geometry")
        
        m2p = group.nodes.new('GeometryNodeMeshToPoints')
        m2p.inputs['Radius'].default_value = scene.megasam_point_radius
        
        set_mat = group.nodes.new('GeometryNodeSetMaterial')
        set_mat.inputs['Material'].default_value = mat
        
        final_geo = m2p.outputs[0]
        
        # --- ANIMATION LOGIC: Filter by FrameID ---
        if scene.megasam_import_animate:
            sep_geo = group.nodes.new('GeometryNodeSeparateGeometry')
            named_attr = group.nodes.new('GeometryNodeInputNamedAttribute')
            named_attr.data_type = 'FLOAT'
            named_attr.inputs[0].default_value = "FrameID"
            
            scene_time = group.nodes.new('GeometryNodeInputSceneTime')
            
            # Snap logic: floor((SceneFrame - start_frame) / effective_step) * effective_step + start_frame
            sub = group.nodes.new('ShaderNodeMath')
            sub.operation = 'SUBTRACT'
            sub.inputs[1].default_value = float(start_frame)
            
            div = group.nodes.new('ShaderNodeMath')
            div.operation = 'DIVIDE'
            div.inputs[1].default_value = float(effective_step)
            
            flr = group.nodes.new('ShaderNodeMath')
            flr.operation = 'FLOOR'
            
            mul = group.nodes.new('ShaderNodeMath')
            mul.operation = 'MULTIPLY'
            mul.inputs[1].default_value = float(effective_step)
            
            add = group.nodes.new('ShaderNodeMath')
            add.operation = 'ADD'
            add.inputs[1].default_value = float(start_frame)
            
            compare = group.nodes.new('FunctionNodeCompare')
            compare.data_type = 'FLOAT'
            compare.operation = 'EQUAL'
            
            # Chain snap logic
            group.links.new(scene_time.outputs[1], sub.inputs[0])
            group.links.new(sub.outputs[0], div.inputs[0])
            group.links.new(div.outputs[0], flr.inputs[0])
            group.links.new(flr.outputs[0], mul.inputs[0])
            group.links.new(mul.outputs[0], add.inputs[0])
            
            group.links.new(named_attr.outputs[0], compare.inputs[0]) # Attribute FrameID
            group.links.new(add.outputs[0], compare.inputs[1]) # Snapped Frame
            
            group.links.new(in_node.outputs[0], sep_geo.inputs[0])
            group.links.new(compare.outputs[0], sep_geo.inputs['Selection'])
            group.links.new(sep_geo.outputs['Selection'], m2p.inputs[0])
        else:
            group.links.new(in_node.outputs[0], m2p.inputs[0])

        group.links.new(final_geo, set_mat.inputs[0])
        group.links.new(set_mat.outputs[0], out_node.inputs[0])

class MEGASAM_OT_generate_mesh(bpy.types.Operator):
    bl_idname = "megasam.generate_mesh"
    bl_label = "Generate Mesh (Current Frame)"
    bl_description = "Generate a high-quality textured mesh from the current frame"

    def execute(self, context):
        scene = context.scene
        
        selected_path = scene.megasam_result_list
        if not selected_path or selected_path == 'NONE':
            self.report({'ERROR'}, "No valid result selected")
            return {'CANCELLED'}
            
        python_exe = get_python_path(context)
        repo_root = get_megasam_root(context)
        
        # 1. Python Path Validation/Correction
        if not python_exe or not os.path.exists(python_exe):
            self.report({'ERROR'}, f"Python executable not found at: {python_exe}")
            return {'CANCELLED'}
            
        if os.path.isdir(python_exe):
            python_exe = os.path.join(python_exe, "python.exe")
            
        if not os.path.exists(python_exe):
            self.report({'ERROR'}, f"Python executable not found at: {python_exe}")
            return {'CANCELLED'}
            
        # 2. Script Path Resolution
        script_name = "generate_mesh_from_npz_FINAL.py"
        script_path = os.path.join(repo_root, script_name)
        if not os.path.exists(script_path):
            script_path = os.path.join(repo_root, "mega-sam", script_name)
            
        if not os.path.exists(script_path):
            self.report({'ERROR'}, f"Mesh script '{script_name}' not found in {repo_root}")
            return {'CANCELLED'}
            
        # 3. Define output path and map timeline frame to NPZ frame index
        frame = context.scene.frame_current
        if scene.megasam_use_timeline_range:
            start_frame = max(0, scene.frame_start)
        elif scene.megasam_use_frame_range:
            start_frame = scene.megasam_start_frame
        else:
            start_frame = 0
        stride = scene.megasam_stride if scene.megasam_stride >= 1 else 1
        npz_frame = max(0, round((frame - start_frame) / stride))
        
        if scene.megasam_mesh_output_dir:
            out_dir = bpy.path.abspath(scene.megasam_mesh_output_dir)
        else:
            # Default to meshes/ inside project folder
            npz_dir = os.path.dirname(selected_path)
            if os.path.basename(npz_dir) in ("outputs_cvd", "outputs"):
                project_folder = os.path.dirname(npz_dir)
            else:
                project_folder = npz_dir
            out_dir = os.path.join(project_folder, "meshes")
            
        os.makedirs(out_dir, exist_ok=True)
        scene_prefix = scene.megasam_scene_name.strip() or "frame"
        output_base = os.path.join(out_dir, f"mesh_{scene_prefix}_F{frame:04d}")
        obj_path = output_base + ".obj"
        
        # Build command
        command = [
            python_exe, script_path,
            "--npz", selected_path,
            "--frame", str(npz_frame),
            "--threshold", str(scene.megasam_mesh_threshold),
            "--upscale", str(scene.megasam_mesh_upscale),
            "--smooth", str(scene.megasam_mesh_smooth),
            "--format", "obj",
            "--output", output_base,
            "--output_dir", out_dir
        ]
        
        # Add filtering arguments if requested
        if scene.megasam_mesh_use_import_settings:
            command.extend([
                "--filter_mode", scene.megasam_import_filter_mode,
                "--motion_threshold", str(scene.megasam_import_motion_threshold)
            ])
        
        try:
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            
            if not os.path.exists(obj_path):
                self.report({'ERROR'}, "Mesh generation failed: Output file not found")
                return {'CANCELLED'}
                
            # Import the OBJ into Blender
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=obj_path)
            else:
                bpy.ops.import_scene.obj(filepath=obj_path)
                
            self.report({'INFO'}, f"Mesh generated in '{out_dir}' for frame {context.scene.frame_current}")
            return {'FINISHED'}
            
        except subprocess.CalledProcessError:
            self.report({'ERROR'}, "Mesh generation script failed. Check console.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}

class MEGASAM_OT_open_outputs_folder(bpy.types.Operator):
    bl_idname = "megasam.open_outputs_folder"
    bl_label = "Open Outputs Folder"
    bl_description = "Open the project outputs folder in File Explorer"

    def execute(self, context):
        repo_root = get_megasam_root(context)
        if not repo_root or not os.path.exists(repo_root):
            self.report({'ERROR'}, "MegaSam root folder not found")
            return {'CANCELLED'}
            
        portable_root = get_portable_root(context)
        scene_name = context.scene.megasam_scene_name.strip()
        project_folder = os.path.join(portable_root, "outputs", scene_name) if scene_name else os.path.join(portable_root, "outputs")
        
        if not os.path.exists(project_folder):
            project_folder = os.path.join(portable_root, "outputs")
            
        os.makedirs(project_folder, exist_ok=True)
        try:
            if os.name == 'nt':
                os.startfile(project_folder)
            else:
                subprocess.Popen(['xdg-open', project_folder])
            self.report({'INFO'}, f"Opened outputs: {project_folder}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Could not open folder: {str(e)}")
            return {'CANCELLED'}

def helper_open_vse(context, view_mode='PREVIEW'):
    screen = context.screen
    if not screen:
        return None

    # 1. Assign active scene to workspace.sequencer_scene (Blender 5.2+)
    if hasattr(context, "workspace") and hasattr(context.workspace, "sequencer_scene"):
        try:
            context.workspace.sequencer_scene = context.scene
        except Exception as e:
            print("Error setting workspace.sequencer_scene:", e)

    # 2. Check if a SEQUENCE_EDITOR area already exists in current screen
    vse_area = None
    for area in screen.areas:
        if area.type == 'SEQUENCE_EDITOR':
            vse_area = area
            break

    # 3. If no VSE area exists, split 3D Viewport or main area to place VSE in lower portion
    if vse_area is None:
        main_area = None
        max_size = -1
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                main_area = area
                break
            size = area.width * area.height
            if size > max_size:
                max_size = size
                main_area = area

        if main_area:
            existing_areas = set(screen.areas)
            try:
                with context.temp_override(area=main_area):
                    bpy.ops.screen.area_split(direction='HORIZONTAL', factor=0.4)
                new_areas = set(screen.areas) - existing_areas
                if new_areas:
                    vse_area = min(list(new_areas) + [main_area], key=lambda a: a.y)
                else:
                    vse_area = main_area
            except Exception as e:
                print(f"Error splitting area for VSE: {e}")
                vse_area = main_area

    # 4. Configure area as SEQUENCE_EDITOR and set view_type & active scene
    if vse_area:
        vse_area.type = 'SEQUENCE_EDITOR'
        for space in vse_area.spaces:
            if space.type == 'SEQUENCE_EDITOR':
                space.view_type = view_mode
                if hasattr(space, "display_mode"):
                    space.display_mode = 'IMAGE'
                if hasattr(space, "display_channel"):
                    space.display_channel = 0
                if hasattr(space, "use_zoom_to_fit"):
                    space.use_zoom_to_fit = True
                
                # Bind active scene to space (Blender 5.x+)
                for scene_attr in ("scene", "displayed_scene", "target_scene", "editing_scene", "sequencer_scene"):
                    if hasattr(space, scene_attr):
                        try:
                            setattr(space, scene_attr, context.scene)
                        except Exception:
                            pass
                break
        vse_area.tag_redraw()

    if hasattr(context, "window") and hasattr(context.window, "scene"):
        try:
            context.window.scene = context.scene
        except Exception:
            pass

    return vse_area

def get_vse_strips_collection(seq_editor):
    if not seq_editor:
        return None
    if hasattr(seq_editor, "strips"):
        return seq_editor.strips
    if hasattr(seq_editor, "sequences"):
        return seq_editor.sequences
    return None

def helper_fit_vse_view(context, vse_area, strip=None):
    """
    Ensures the strip transform scale fits the render frame perfectly without overflow,
    enables Zoom to Fit on the preview space, and frames the preview and timeline views.
    """
    if not vse_area or vse_area.type != 'SEQUENCE_EDITOR':
        return

    scene = context.scene

    # 1. Enable Zoom to Fit on all VSE preview spaces
    for space in vse_area.spaces:
        if space.type == 'SEQUENCE_EDITOR':
            if hasattr(space, "use_zoom_to_fit"):
                space.use_zoom_to_fit = True

    # 2. Locate the MOVIE strip (ignoring SoundStrip if present)
    seq_editor = scene.sequence_editor
    seqs_coll = get_vse_strips_collection(seq_editor)
    movie_strip = strip if (strip and getattr(strip, "type", "") == 'MOVIE') else None

    if not movie_strip and seqs_coll:
        for s in reversed(list(seqs_coll)):
            if getattr(s, "type", "") == 'MOVIE':
                movie_strip = s
                break

    # 3. Ensure the movie strip transform fits the render resolution 1:1
    if movie_strip and hasattr(movie_strip, "transform"):
        orig_w = 0
        orig_h = 0
        if hasattr(movie_strip, "elements") and len(movie_strip.elements) > 0:
            orig_w = movie_strip.elements[0].orig_width
            orig_h = movie_strip.elements[0].orig_height

        if orig_w > 0 and orig_h > 0:
            if scene.render.resolution_x == orig_w and scene.render.resolution_y == orig_h:
                movie_strip.transform.scale_x = 1.0
                movie_strip.transform.scale_y = 1.0
            else:
                fit_scale = min(scene.render.resolution_x / orig_w, scene.render.resolution_y / orig_h)
                movie_strip.transform.scale_x = fit_scale
                movie_strip.transform.scale_y = fit_scale
        else:
            movie_strip.transform.scale_x = 1.0
            movie_strip.transform.scale_y = 1.0

        movie_strip.transform.offset_x = 0.0
        movie_strip.transform.offset_y = 0.0
        if hasattr(movie_strip.transform, "rotation"):
            movie_strip.transform.rotation = 0.0

        # Make movie_strip active and selected
        if seq_editor and hasattr(seq_editor, "active_strip"):
            seq_editor.active_strip = movie_strip
        if hasattr(movie_strip, "select"):
            movie_strip.select = True

    # 4. Zoom preview to fit inside view window (view_all_preview)
    # and zoom timeline to fit sequence (view_all)
    for r in vse_area.regions:
        if r.type in ('PREVIEW', 'WINDOW'):
            try:
                with context.temp_override(area=vse_area, region=r):
                    bpy.ops.sequencer.view_all_preview()
            except Exception:
                pass
        if r.type == 'WINDOW':
            try:
                with context.temp_override(area=vse_area, region=r):
                    bpy.ops.sequencer.view_all()
            except Exception:
                pass

    vse_area.tag_redraw()

class MEGASAM_OT_open_vse(bpy.types.Operator):
    bl_idname = "megasam.open_vse"
    bl_label = "Open VSE"
    bl_description = "Open the Video Sequence Editor in the lower part of the screen"

    def execute(self, context):
        scene = context.scene
        view_mode = scene.megasam_vse_view_mode
        area = helper_open_vse(context, view_mode=view_mode)
        if area:
            # Fit view to existing strip if any
            seq_editor = scene.sequence_editor
            seqs = get_vse_strips_collection(seq_editor)
            movie_strip = None
            if seqs:
                for s in reversed(list(seqs)):
                    if getattr(s, "type", "") == 'MOVIE':
                        movie_strip = s
                        break
            helper_fit_vse_view(context, area, strip=movie_strip)

            def deferred_open_fit():
                try:
                    if hasattr(context, "screen") and area in context.screen.areas and area.type == 'SEQUENCE_EDITOR':
                        helper_fit_vse_view(context, area, strip=movie_strip)
                except Exception:
                    pass
                return None

            if hasattr(bpy.app, "timers"):
                bpy.app.timers.register(deferred_open_fit, first_interval=0.05)

            mode_names = {'PREVIEW': 'Preview', 'SEQUENCER': 'Sequencer', 'SEQUENCER_PREVIEW': 'Sequencer & Preview'}
            self.report({'INFO'}, f"VSE opened in {mode_names.get(view_mode, view_mode)} mode")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Could not open Video Sequence Editor")
            return {'CANCELLED'}

class MEGASAM_OT_load_video_vse(bpy.types.Operator):
    bl_idname = "megasam.load_video_vse"
    bl_label = "Open Video in VSE"
    bl_description = "Load the selected video file into the Video Sequence Editor"

    def execute(self, context):
        scene = context.scene
        path = bpy.path.abspath(scene.megasam_video_path)
        
        if not path or not os.path.exists(path):
            self.report({'ERROR'}, "Please select a valid video file first")
            return {'CANCELLED'}
            
        try:
            # 0. Ensure workspace.sequencer_scene is set to active scene (Blender 5.2+)
            if hasattr(context, "workspace") and hasattr(context.workspace, "sequencer_scene"):
                try:
                    context.workspace.sequencer_scene = scene
                except Exception as e:
                    print("Error setting workspace.sequencer_scene:", e)

            # Ensure scene name is set
            if not scene.megasam_scene_name.strip():
                base_name = os.path.splitext(os.path.basename(path))[0]
                repo_root = get_megasam_root(context)
                if repo_root:
                    portable_root = get_portable_root(context)
                    output_root = os.path.join(portable_root, "outputs")
                    scene.megasam_scene_name = get_unique_scene_name(output_root, base_name)
                else:
                    scene.megasam_scene_name = base_name

            if scene.name.startswith("Scene"):
                try:
                    scene.name = scene.megasam_scene_name
                except Exception:
                    pass

            # Pre-set render resolution if known to prevent movie_strip_add from scaling up
            if scene.megasam_video_width > 0 and scene.megasam_video_height > 0:
                scene.render.resolution_x = scene.megasam_video_width
                scene.render.resolution_y = scene.megasam_video_height

            # 1. Ensure VSE area exists and is configured
            vse_area = helper_open_vse(context, view_mode=scene.megasam_vse_view_mode)

            # 2. Get or create Sequence Editor data
            seq_editor = scene.sequence_editor or scene.sequence_editor_create()
            seqs_coll = get_vse_strips_collection(seq_editor)
            
            # 3. Clear existing strips if requested
            if scene.megasam_vse_clear_existing and seqs_coll:
                for seq in list(seqs_coll):
                    try:
                        seqs_coll.remove(seq)
                    except Exception:
                        pass

            # 4. Load movie strip
            strip_name = os.path.basename(path)
            if scene.megasam_use_timeline_range:
                start_frame = max(1, scene.frame_start)
            elif scene.megasam_use_frame_range and scene.megasam_start_frame > 0:
                start_frame = scene.megasam_start_frame
            else:
                start_frame = 1
            
            added_movie_strip = None
            if vse_area:
                try:
                    with context.temp_override(area=vse_area):
                        bpy.ops.sequencer.movie_strip_add(filepath=path, frame_start=start_frame, channel=1)
                    seqs_coll = get_vse_strips_collection(seq_editor)
                    if seqs_coll:
                        for s in reversed(list(seqs_coll)):
                            if getattr(s, "type", "") == 'MOVIE':
                                added_movie_strip = s
                                break
                except Exception as e:
                    print("movie_strip_add operator failed, using fallback:", e)

            if not added_movie_strip:
                seqs_coll = get_vse_strips_collection(seq_editor)
                if seqs_coll is None:
                    self.report({'ERROR'}, "Could not access sequence editor strips container")
                    return {'CANCELLED'}
                added_movie_strip = seqs_coll.new_movie(
                    name=strip_name,
                    filepath=path,
                    channel=1,
                    frame_start=start_frame
                )

            # 5. Update render resolution to match exact video element dimensions
            if added_movie_strip and hasattr(added_movie_strip, "elements") and len(added_movie_strip.elements) > 0:
                elem_w = added_movie_strip.elements[0].orig_width
                elem_h = added_movie_strip.elements[0].orig_height
                if elem_w > 0 and elem_h > 0:
                    scene.render.resolution_x = elem_w
                    scene.render.resolution_y = elem_h
                    scene.megasam_video_width = elem_w
                    scene.megasam_video_height = elem_h
            elif scene.megasam_video_width > 0 and scene.megasam_video_height > 0:
                scene.render.resolution_x = scene.megasam_video_width
                scene.render.resolution_y = scene.megasam_video_height

            # 6. Ensure movie strip scale is 1.0 (matching resolution)
            if added_movie_strip and hasattr(added_movie_strip, "transform"):
                added_movie_strip.transform.scale_x = 1.0
                added_movie_strip.transform.scale_y = 1.0
                added_movie_strip.transform.offset_x = 0.0
                added_movie_strip.transform.offset_y = 0.0

            # 7. Sync timeline range & current playhead frame
            scene.frame_start = start_frame
            if added_movie_strip and hasattr(added_movie_strip, "frame_final_duration") and added_movie_strip.frame_final_duration > 0:
                scene.frame_end = start_frame + int(added_movie_strip.frame_final_duration) - 1
            elif scene.megasam_video_frames > 0:
                scene.frame_end = start_frame + scene.megasam_video_frames - 1
                
            # Set playhead frame to start_frame to immediately display video preview
            scene.frame_current = start_frame

            # 8. Fit strip transform and zoom VSE view to make video fully inside the view
            if vse_area:
                helper_fit_vse_view(context, vse_area, strip=added_movie_strip)

                # Also schedule a deferred fit in GUI mode once Blender has computed area/region layout
                def deferred_fit():
                    try:
                        if hasattr(context, "screen") and vse_area in context.screen.areas and vse_area.type == 'SEQUENCE_EDITOR':
                            helper_fit_vse_view(context, vse_area, strip=added_movie_strip)
                    except Exception:
                        pass
                    return None

                if hasattr(bpy.app, "timers"):
                    bpy.app.timers.register(deferred_fit, first_interval=0.05)

            # Refresh VSE area
            if vse_area:
                vse_area.tag_redraw()

            self.report({'INFO'}, f"Loaded video '{strip_name}' into VSE")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load video into VSE: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

classes = (
    MEGASAM_OT_set_range_from_timeline,
    MEGASAM_OT_run_pipeline,
    MEGASAM_OT_import_result,
    MEGASAM_OT_generate_mesh,
    MEGASAM_OT_open_outputs_folder,
    MEGASAM_OT_open_vse,
    MEGASAM_OT_load_video_vse,
)

def register():
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            pass

def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
