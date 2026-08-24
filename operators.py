import bpy
import subprocess
import os
import numpy as np
import mathutils

class MEGASAM_OT_run_pipeline(bpy.types.Operator):
    bl_idname = "megasam.run_pipeline"
    bl_label = "Run MegaSam Pipeline"
    bl_description = "Execute the run_pipeline.py script in a new console"

    def execute(self, context):
        scene = context.scene
        # Access preferences using the package name
        prefs = context.preferences.addons[__package__].preferences
        
        python_exe = prefs.python_path
        repo_root = prefs.megasam_root_path
        video_path = scene.megasam_video_path
        batch_size = scene.megasam_batch_size
        
        # Validation and auto-correction
        if not python_exe or not os.path.exists(python_exe):
            self.report({'ERROR'}, "Python Path is invalid or not set in Preferences")
            return {'CANCELLED'}
            
        if os.path.isdir(python_exe):
            # If a directory is selected, try to find python.exe inside
            potential_exe = os.path.join(python_exe, "python.exe")
            if os.path.exists(potential_exe):
                python_exe = potential_exe
            else:
                self.report({'ERROR'}, "Selected Python path is a directory and 'python.exe' was not found inside.")
                return {'CANCELLED'}
        
        if not os.path.isfile(python_exe):
            self.report({'ERROR'}, f"Python Path is not a file: {python_exe}")
            return {'CANCELLED'}
            
        if not repo_root or not os.path.exists(repo_root):
            self.report({'ERROR'}, "MegaSam Root Path is invalid or not set")
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
            self.report({'ERROR'}, f"Could not find 'run_pipeline.py' in {repo_root} or its 'mega-sam' subfolder")
            return {'CANCELLED'}
        
        if not video_path or not os.path.exists(video_path):
            self.report({'ERROR'}, "Video File path is invalid or not set")
            return {'CANCELLED'}
            
        # Use user-provided scene name or fallback to video filename
        scene_name = scene.megasam_scene_name
        if not scene_name:
            scene_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Build command
        command = [
            python_exe,
            script_path,
            "-v", video_path,
            "-s", scene_name
        ]
        
        if scene.megasam_use_batch:
            command.extend(["--batch_size", str(scene.megasam_batch_size)])

        
        print(f"Executing: {' '.join(command)}")
        
        try:
            # Set CWD to the script's directory so relative paths for weights work
            script_dir = os.path.dirname(script_path)
            
            # On Windows, CREATE_NEW_CONSOLE (0x00000010) opens a separate terminal
            if os.name == 'nt':
                subprocess.Popen(command, cwd=script_dir, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(command, cwd=script_dir)
                
            self.report({'INFO'}, f"MegaSam Pipeline started for: {scene_name}")
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
        prefs = context.preferences.addons[__package__].preferences
        
        selected_path = scene.megasam_result_list
        if not selected_path or selected_path == 'NONE':
            self.report({'ERROR'}, "No valid result selected")
            return {'CANCELLED'}
            
        try:
            # Extract scene name from filename
            filename = os.path.basename(selected_path)
            scene_name = filename.replace("_sgd_cvd_hr.npz", "").replace(".npz", "")

            # 1. Load Data
            data = self.load_cvd_data(selected_path)
            
            # 2. Cleanup old objects (if requested)
            if scene.megasam_import_clean_scene:
                self.cleanup_scene(scene)
            
            # 3. Setup Camera
            self.setup_camera(data, context, scene_name)
            
            # 4. Create Point Cloud
            self.create_animated_point_cloud(data, context, scene_name)
            
            self.report({'INFO'}, f"Imported: {os.path.basename(selected_path)}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

    def load_cvd_data(self, path):
        if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path)
        return {
            'images': data['images'], 'depths': data['depths'], 
            'uncertainty': data.get('uncertainty', None),
            'intrinsic': data['intrinsic'], 'cam_c2w': data['cam_c2w']
        }

    def cleanup_scene(self, scene):
        # Find objects to delete
        to_delete = []
        for obj in bpy.data.objects:
            if obj.name.startswith("PC_") or obj.name.startswith("Cam_") or obj.name == "CVD_Camera":
                to_delete.append(obj)
        
        # Delete them
        if to_delete:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in to_delete:
                obj.select_set(True)
            bpy.ops.object.delete()

    def get_world_transform(self):
        return mathutils.Matrix.Rotation(np.radians(-90), 4, 'X')

    def setup_camera(self, data, context, scene_name):
        world_transform = self.get_world_transform()
        cam_name = f"Cam_{scene_name}"
        cam_data = bpy.data.cameras.new(cam_name)
        cam_obj = bpy.data.objects.new(cam_name, cam_data)
        context.collection.objects.link(cam_obj)
        
        K, images = data['intrinsic'], data['images']
        num_frames, H, W, _ = images.shape
        
        context.scene.render.resolution_x, context.scene.render.resolution_y = W, H
        context.scene.frame_start, context.scene.frame_end = 0, num_frames - 1
        
        cam_data.lens_unit = 'FOV'
        cam_data.sensor_fit = 'HORIZONTAL'
        cam_data.angle = 2 * np.arctan(W / (2 * K[0, 0]))
        
        # Calculate optical center shift
        max_dim = max(W, H)
        cam_data.shift_x = -(W / 2.0 - K[0, 2]) / max_dim
        cam_data.shift_y = (H / 2.0 - K[1, 2]) / max_dim
        
        cv_to_blender_local = mathutils.Matrix(((1,0,0,0),(0,-1,0,0),(0,0,-1,0),(0,0,0,1)))
        for f in range(num_frames):
            c2w = mathutils.Matrix(data['cam_c2w'][f])
            cam_obj.matrix_world = world_transform @ c2w @ cv_to_blender_local
            cam_obj.keyframe_insert(data_path="location", frame=f)
            cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)
        return cam_obj

    def create_animated_point_cloud(self, data, context, scene_name):
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
            all_f_ids.append(np.full(len(z), f, dtype=np.float32))

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
        
        self.setup_point_visualization(obj, context, filter_mode)
        return obj

    def setup_point_visualization(self, obj, context, filter_mode):
        scene = context.scene
        frame_step = scene.megasam_import_frame_step
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
            
            # Snap logic: floor(SceneFrame / frame_step) * frame_step
            # This keeps the last frame visible until the next step is reached
            div = group.nodes.new('ShaderNodeMath')
            div.operation = 'DIVIDE'
            div.inputs[1].default_value = frame_step
            
            flr = group.nodes.new('ShaderNodeMath')
            flr.operation = 'FLOOR'
            
            mul = group.nodes.new('ShaderNodeMath')
            mul.operation = 'MULTIPLY'
            mul.inputs[1].default_value = frame_step
            
            compare = group.nodes.new('FunctionNodeCompare')
            compare.data_type = 'FLOAT'
            compare.operation = 'EQUAL'
            
            # Chain the snap logic
            group.links.new(scene_time.outputs[1], div.inputs[0]) # Scene Frame
            group.links.new(div.outputs[0], flr.inputs[0])
            group.links.new(flr.outputs[0], mul.inputs[0])
            
            group.links.new(named_attr.outputs[0], compare.inputs[0]) # Attribute FrameID
            group.links.new(mul.outputs[0], compare.inputs[1]) # Snapped Frame
            
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
        prefs = context.preferences.addons[__package__].preferences
        
        selected_path = scene.megasam_result_list
        if not selected_path or selected_path == 'NONE':
            self.report({'ERROR'}, "No valid result selected")
            return {'CANCELLED'}
            
        python_exe = prefs.python_path
        repo_root = prefs.megasam_root_path
        
        # 1. Python Path Validation/Correction
        if os.path.isdir(python_exe):
            python_exe = os.path.join(python_exe, "python.exe")
            
        if not os.path.exists(python_exe):
            self.report({'ERROR'}, f"Python executable not found at: {python_exe}")
            return {'CANCELLED'}
            
        # 2. Script Path Resolution (Search in root and mega-sam subfolder)
        script_name = "generate_mesh_from_npz_FINAL.py"
        script_path = os.path.join(repo_root, script_name)
        if not os.path.exists(script_path):
            script_path = os.path.join(repo_root, "mega-sam", script_name)
            
        if not os.path.exists(script_path):
            self.report({'ERROR'}, f"Mesh script not found in {repo_root}")
            return {'CANCELLED'}
            
        # 3. Define output path (inside mega-sam folder for consistency)
        project_dir = os.path.dirname(script_path)
        frame = context.scene.frame_current
        output_base = os.path.join(project_dir, f"temp_mesh_F{frame:03d}")
        
        # Build command
        command = [
            python_exe, script_path,
            "--npz", selected_path,
            "--frame", str(context.scene.frame_current),
            "--threshold", str(scene.megasam_mesh_threshold),
            "--upscale", str(scene.megasam_mesh_upscale),
            "--smooth", str(scene.megasam_mesh_smooth),
            "--format", "obj",
            "--output", output_base
        ]
        
        # Add filtering arguments if requested
        if scene.megasam_mesh_use_import_settings:
            command.extend([
                "--filter_mode", scene.megasam_import_filter_mode,
                "--motion_threshold", str(scene.megasam_import_motion_threshold)
            ])
        
        if scene.megasam_mesh_output_dir:
            command.extend(["--output_dir", bpy.path.abspath(scene.megasam_mesh_output_dir)])
            # If output_dir is used, the final obj path will be in that dir
            obj_path = os.path.join(bpy.path.abspath(scene.megasam_mesh_output_dir), os.path.basename(output_base) + ".obj")
        else:
            obj_path = output_base + ".obj"
        
        try:
            # Run the generation script
            subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            if not os.path.exists(obj_path):
                self.report({'ERROR'}, "Mesh generation failed: Output file not found")
                return {'CANCELLED'}
                
            # Import the OBJ into Blender
            if hasattr(bpy.ops.wm, "obj_import"):
                bpy.ops.wm.obj_import(filepath=obj_path)
            else:
                bpy.ops.import_scene.obj(filepath=obj_path)
                
            self.report({'INFO'}, f"Mesh generated for frame {context.scene.frame_current}")
            return {'FINISHED'}
            
        except subprocess.CalledProcessError as e:
            self.report({'ERROR'}, f"Mesh generation script failed. Check console.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}

def register():
    bpy.utils.register_class(MEGASAM_OT_run_pipeline)
    bpy.utils.register_class(MEGASAM_OT_import_result)
    bpy.utils.register_class(MEGASAM_OT_generate_mesh)

def unregister():
    bpy.utils.unregister_class(MEGASAM_OT_run_pipeline)
    bpy.utils.unregister_class(MEGASAM_OT_import_result)
    bpy.utils.unregister_class(MEGASAM_OT_generate_mesh)
