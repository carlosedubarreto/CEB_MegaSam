import os
import bpy

def get_python_path(context=None):
    """Resolve the Python executable path, using preferences or auto-detecting portable python."""
    py_path = ""
    addon_name = __package__.split('.')[0] if __package__ else "CEB_MegaSam"
    if context and hasattr(context, "preferences") and addon_name in context.preferences.addons:
        prefs = context.preferences.addons[addon_name].preferences
        py_path = prefs.python_path.strip()

    if py_path:
        py_path = bpy.path.abspath(py_path)
        if os.path.isdir(py_path):
            potential_exe = os.path.join(py_path, "python.exe")
            if os.path.exists(potential_exe):
                return potential_exe
        elif os.path.isfile(py_path):
            return py_path

    # Auto-detection search paths
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(addon_dir, "..", "Portable_torch210_cu128", "python-3.10.11-embed-amd64", "python.exe"),
        os.path.join(addon_dir, "..", "..", "Portable_torch210_cu128", "python-3.10.11-embed-amd64", "python.exe"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable_torch210_cu128\python-3.10.11-embed-amd64\python.exe",
        os.path.join(addon_dir, "..", "Portable", "python-3.10.11-embed-amd64", "python.exe"),
        os.path.join(addon_dir, "..", "..", "Portable", "python-3.10.11-embed-amd64", "python.exe"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable\python-3.10.11-embed-amd64\python.exe",
    ]

    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isfile(norm):
            return norm

    return py_path

def get_megasam_root(context=None):
    """Resolve the MegaSam project root folder, prioritizing the addon folder then portable fallbacks."""
    root_path = ""
    addon_name = __package__.split('.')[0] if __package__ else "CEB_MegaSam"
    if context and hasattr(context, "preferences") and addon_name in context.preferences.addons:
        prefs = context.preferences.addons[addon_name].preferences
        root_path = prefs.megasam_root_path.strip()

    if root_path:
        root_path = bpy.path.abspath(root_path)
        if os.path.isdir(root_path):
            return root_path

    addon_dir = os.path.dirname(os.path.abspath(__file__))
    # Prioritize mega-sam inside the addon folder
    candidates = [
        os.path.join(addon_dir, "mega-sam"),
        addon_dir,
    ]

    # Auto-detection relative to python portable path
    py_path = get_python_path(context)
    if py_path:
        py_dir = py_path if os.path.isdir(py_path) else os.path.dirname(py_path)
        py_parent = os.path.dirname(py_dir)
        candidates.extend([
            os.path.join(py_parent, "mega-sam"),
            os.path.join(py_dir, "mega-sam"),
            py_parent,
            py_dir
        ])

    candidates.extend([
        os.path.join(addon_dir, "..", "Portable_torch210_cu128", "mega-sam"),
        os.path.join(addon_dir, "..", "..", "Portable_torch210_cu128", "mega-sam"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable_torch210_cu128\mega-sam",
        os.path.join(addon_dir, "..", "Portable", "mega-sam"),
        os.path.join(addon_dir, "..", "mega-sam"),
        os.path.join(addon_dir, "..", "..", "Portable", "mega-sam"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable\mega-sam"
    ])

    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isdir(norm):
            # If inside norm there is a mega-sam subfolder containing run_pipeline.py
            if os.path.exists(os.path.join(norm, "run_pipeline.py")):
                return norm
            if os.path.exists(os.path.join(norm, "mega-sam", "run_pipeline.py")):
                return os.path.join(norm, "mega-sam")
            if os.path.exists(os.path.join(norm, "cvd_opt")) or os.path.exists(os.path.join(norm, "Depth-Anything")):
                return norm

    return root_path

def get_portable_root(context=None):
    """Resolve the portable Python root directory (e.g. Portable_torch210_cu128)."""
    py_path = get_python_path(context)
    if py_path:
        py_dir = py_path if os.path.isdir(py_path) else os.path.dirname(py_path)
        py_parent = os.path.dirname(py_dir)
        for cand in [py_parent, py_dir]:
            if (os.path.exists(os.path.join(cand, "DepthAnything_CKPT")) or
                os.path.exists(os.path.join(cand, "outputs"))):
                return cand

    addon_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(addon_dir, "..", "Portable_torch210_cu128"),
        os.path.join(addon_dir, "..", "..", "Portable_torch210_cu128"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable_torch210_cu128",
        os.path.join(addon_dir, "..", "Portable"),
        os.path.join(addon_dir, "..", "..", "Portable"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable",
    ]
    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isdir(norm):
            return norm

    if py_path:
        py_dir = py_path if os.path.isdir(py_path) else os.path.dirname(py_path)
        return os.path.dirname(py_dir) if os.path.basename(py_dir).lower().startswith("python") else py_dir

    return addon_dir

def get_depth_anything_ckpt(context=None, encoder="vits"):
    """
    Resolve the Depth-Anything checkpoint path.
    Searches inside the portable Python folder under DepthAnything_CKPT.
    """
    portable_root = get_portable_root(context)
    candidates = [
        os.path.join(portable_root, "DepthAnything_CKPT", "checkpoints", f"depth_anything_{encoder}14.pth"),
        os.path.join(portable_root, "DepthAnything_CKPT", f"depth_anything_{encoder}14.pth"),
        os.path.join(portable_root, "checkpoints", f"depth_anything_{encoder}14.pth"),
    ]
    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isfile(norm):
            return norm

    # Recursive search inside DepthAnything_CKPT
    ckpt_dir = os.path.join(portable_root, "DepthAnything_CKPT")
    if os.path.isdir(ckpt_dir):
        for root, _, files in os.walk(ckpt_dir):
            for f in files:
                if f.startswith(f"depth_anything_{encoder}") and f.endswith(".pth"):
                    return os.path.join(root, f)

    # Fallback to repo_root
    repo_root = get_megasam_root(context)
    if repo_root:
        legacy = os.path.join(repo_root, "Depth-Anything", "checkpoints", f"depth_anything_{encoder}14.pth")
        if os.path.isfile(legacy):
            return legacy

    return ""

def get_unique_scene_name(base_output_dir, base_name):
    """
    If base_output_dir / base_name exists or if base_output_dir / f'{base_name}_sgd_cvd_hr.npz' exists,
    increment with _001, _002, etc. until an unused name is found.
    Handles existing trailing numbers like 'chase_001' -> 'chase_002'.
    """
    import re
    if not base_output_dir or not base_name:
        return base_name or "scene"
        
    def is_taken(name):
        folder_path = os.path.join(base_output_dir, name)
        npz_cvd = os.path.join(base_output_dir, f"{name}_sgd_cvd_hr.npz")
        npz_droid = os.path.join(base_output_dir, f"{name}_droid.npz")
        inner_cvd = os.path.join(base_output_dir, name, "outputs_cvd", f"{name}_sgd_cvd_hr.npz")
        return os.path.exists(folder_path) or os.path.exists(npz_cvd) or os.path.exists(npz_droid) or os.path.exists(inner_cvd)

    if not is_taken(base_name):
        return base_name

    match = re.search(r'^(.*)_(\d+)$', base_name)
    if match:
        prefix = match.group(1)
        start_idx = int(match.group(2)) + 1
        pad_len = max(3, len(match.group(2)))
    else:
        prefix = base_name
        start_idx = 1
        pad_len = 3

    idx = start_idx
    while True:
        candidate = f"{prefix}_{idx:0{pad_len}d}"
        if not is_taken(candidate):
            return candidate
        idx += 1
