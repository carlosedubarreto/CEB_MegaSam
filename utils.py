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
    """Resolve the MegaSam project root folder, auto-detecting from the Python portable parent folder."""
    root_path = ""
    addon_name = __package__.split('.')[0] if __package__ else "CEB_MegaSam"
    if context and hasattr(context, "preferences") and addon_name in context.preferences.addons:
        prefs = context.preferences.addons[addon_name].preferences
        root_path = prefs.megasam_root_path.strip()

    if root_path:
        root_path = bpy.path.abspath(root_path)
        if os.path.isdir(root_path):
            return root_path

    # Auto-detection relative to python portable path
    py_path = get_python_path(context)
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

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
        os.path.join(addon_dir, "..", "Portable", "mega-sam"),
        os.path.join(addon_dir, "..", "mega-sam"),
        os.path.join(addon_dir, "..", "..", "Portable", "mega-sam"),
        r"D:\_Code\Meu\CEB_MegaSam_prj\Portable\mega-sam"
    ])

    for c in candidates:
        norm = os.path.normpath(c)
        if os.path.isdir(norm):
            # Check for characteristic MegaSam files / subfolders
            if (os.path.exists(os.path.join(norm, "run_pipeline.py")) or 
                os.path.exists(os.path.join(norm, "cvd_opt")) or 
                os.path.exists(os.path.join(norm, "Depth-Anything")) or
                os.path.exists(os.path.join(norm, "mega-sam", "run_pipeline.py"))):
                return norm

    return root_path
