"""
ComfyUI-SAM3: SAM3 Integration for ComfyUI
Version: 3.0.0
"""

import os
import sys
import traceback

__version__ = "3.0.0"

# CRITICAL: Initialize these FIRST at module level before any imports
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

INIT_SUCCESS = False
INIT_ERRORS = []

force_init = os.environ.get('SAM3_FORCE_INIT') == '1'
is_pytest = 'PYTEST_CURRENT_TEST' in os.environ
skip_init = is_pytest and not force_init

if not skip_init:
    print(f"[SAM3] ComfyUI-SAM3 v{__version__} initializing...")

    # Step 0: Apply H200 optimizations
    try:
        from .nodes import h200_optimizations
        print("[SAM3] [OK] H200 optimizations applied")
    except Exception as e:
        print(f"[SAM3] [WARNING] H200 optimizations not loaded: {e}")

    # Step 1: Register sam3 model folder
    try:
        import folder_paths
        sam3_model_dir = os.path.join(folder_paths.models_dir, "sam3")
        os.makedirs(sam3_model_dir, exist_ok=True)
        folder_paths.add_model_folder_path("sam3", sam3_model_dir)
        print(f"[SAM3] [OK] Registered model folder: {sam3_model_dir}")
    except Exception as e:
        INIT_ERRORS.append(f"Failed to register model folder: {e}")
        print(f"[SAM3] [WARNING] {INIT_ERRORS[-1]}")

    # Step 2: Import main node classes
    try:
        from .nodes import NODE_CLASS_MAPPINGS as MAIN_NODES
        from .nodes import NODE_DISPLAY_NAME_MAPPINGS as MAIN_DISPLAY
        NODE_CLASS_MAPPINGS.update(MAIN_NODES)
        NODE_DISPLAY_NAME_MAPPINGS.update(MAIN_DISPLAY)
        print("[SAM3] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        INIT_ERRORS.append(f"Failed to import node classes: {e}")
        print(f"[SAM3] [WARNING] {INIT_ERRORS[-1]}")
        print(f"[SAM3] Traceback:\n{traceback.format_exc()}")

    # Step 3: Import fast mask nodes
    try:
        from .nodes.fast_mask_composite import NODE_CLASS_MAPPINGS as FAST_NODES
        from .nodes.fast_mask_composite import NODE_DISPLAY_NAME_MAPPINGS as FAST_DISPLAY
        NODE_CLASS_MAPPINGS.update(FAST_NODES)
        NODE_DISPLAY_NAME_MAPPINGS.update(FAST_DISPLAY)
        print(f"[SAM3] [OK] Fast mask nodes added: {', '.join(FAST_NODES.keys())}")
    except Exception as e:
        INIT_ERRORS.append(f"Failed to import fast mask nodes: {e}")
        print(f"[SAM3] [WARNING] {INIT_ERRORS[-1]}")

    # Step 4: Import server
    try:
        from . import sam3_server
        print("[SAM3] [OK] API endpoints registered")
    except Exception as e:
        INIT_ERRORS.append(f"Failed to register API endpoints: {e}")
        print(f"[SAM3] [WARNING] {INIT_ERRORS[-1]}")

    # Report status
    if INIT_SUCCESS:
        print(f"[SAM3] [OK] Loaded successfully!")
        print(f"[SAM3] Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
        print(f"[SAM3] Interactive SAM3 Detector: Right-click any IMAGE/MASK node -> 'Open in SAM3 Detector'")
    else:
        print(f"[SAM3] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s))")

else:
    print(f"[SAM3] v{__version__} pytest mode - skipping init")

WEB_DIRECTORY = "./web"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
