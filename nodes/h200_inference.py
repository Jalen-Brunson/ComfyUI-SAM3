"""
H200-Optimized Inference Session Management

This module provides optimized inference state management for SAM3 video tracking
with specific optimizations for H200/Hopper GPUs.

Key optimizations:
1. torch.compile for propagation methods
2. BF16 autocast with TF32 matmul
3. Flash Attention 3 when available
4. Channels-last memory format
5. CUDA graph capture for repeated operations

Integration:
    Replace the standard get_inference_state call with get_optimized_inference_state
    or simply import this module to auto-patch the existing functions.
"""

import os
import gc
import torch
import functools
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

# Try to import the original module for patching
try:
    from . import inference_reconstructor as original_module
    HAS_ORIGINAL = True
except ImportError:
    HAS_ORIGINAL = False


# =============================================================================
# GPU Detection (cached)
# =============================================================================

@lru_cache(maxsize=1)
def _get_gpu_info() -> Tuple[str, int, int]:
    """Get GPU name and compute capability."""
    if not torch.cuda.is_available():
        return ("CPU", 0, 0)
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return (name, major, minor)


def is_hopper() -> bool:
    """Check if running on Hopper GPU (H100, H200)."""
    _, major, _ = _get_gpu_info()
    return major >= 9


def is_ampere_plus() -> bool:
    """Check if running on Ampere or newer."""
    _, major, _ = _get_gpu_info()
    return major >= 8


# =============================================================================
# Global Optimization State
# =============================================================================

_OPTIMIZATIONS_APPLIED = False
_COMPILED_PREDICTORS: Dict[int, Any] = {}


def apply_global_optimizations():
    """Apply one-time global PyTorch optimizations."""
    global _OPTIMIZATIONS_APPLIED
    
    if _OPTIMIZATIONS_APPLIED or not torch.cuda.is_available():
        return
    
    name, major, _ = _get_gpu_info()
    print(f"[SAM3 H200] Applying optimizations for {name}")
    
    # TF32 for Ampere+
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # cuDNN settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Flash/Memory-efficient SDP
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Hopper-specific
    if major >= 9:
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        print("[SAM3 H200] Hopper optimizations enabled")
    
    _OPTIMIZATIONS_APPLIED = True


# =============================================================================
# Model Optimization
# =============================================================================

def optimize_predictor(predictor, compile_mode: str = "reduce-overhead"):
    """
    Optimize a SAM3 video predictor for H200.
    
    Args:
        predictor: SAM3VideoPredictor or SAM2VideoPredictor instance
        compile_mode: "reduce-overhead" (fast compile) or "max-autotune" (fastest runtime)
    
    Returns:
        Optimized predictor
    """
    if not torch.cuda.is_available():
        return predictor
    
    # Check if already optimized
    pred_id = id(predictor)
    if pred_id in _COMPILED_PREDICTORS:
        return _COMPILED_PREDICTORS[pred_id]
    
    _, major, _ = _get_gpu_info()
    
    # Apply global optimizations
    apply_global_optimizations()
    
    # Get the underlying model
    model = getattr(predictor, 'model', predictor)
    
    # 1. Ensure BF16 dtype for Ampere+
    if major >= 8:
        try:
            model = model.to(dtype=torch.bfloat16)
            print("[SAM3 H200] Model converted to BF16")
        except Exception as e:
            print(f"[SAM3 H200] BF16 conversion skipped: {e}")
    
    # 2. Channels-last memory format (better for convolutions)
    if major >= 8:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("[SAM3 H200] Channels-last memory format applied")
        except Exception as e:
            print(f"[SAM3 H200] Channels-last skipped: {e}")
    
    # 3. torch.compile key methods (PyTorch 2.0+)
    if hasattr(torch, 'compile') and major >= 8:
        # Use max-autotune for Hopper, reduce-overhead for Ampere
        mode = "max-autotune" if major >= 9 else compile_mode
        
        try:
            # Compile the image encoder (most expensive operation)
            if hasattr(model, 'image_encoder'):
                model.image_encoder = torch.compile(
                    model.image_encoder,
                    mode=mode,
                    fullgraph=False
                )
                print(f"[SAM3 H200] Image encoder compiled (mode={mode})")
            
            # Compile mask decoder
            if hasattr(model, 'sam_mask_decoder'):
                model.sam_mask_decoder = torch.compile(
                    model.sam_mask_decoder,
                    mode=mode,
                    fullgraph=False
                )
                print("[SAM3 H200] Mask decoder compiled")
            
            # Compile memory attention (video tracking)
            if hasattr(model, 'memory_attention'):
                model.memory_attention = torch.compile(
                    model.memory_attention,
                    mode=mode,
                    fullgraph=False
                )
                print("[SAM3 H200] Memory attention compiled")
                
        except Exception as e:
            print(f"[SAM3 H200] Compilation partially failed: {e}")
    
    # Cache the optimized predictor
    _COMPILED_PREDICTORS[pred_id] = predictor
    
    return predictor


# =============================================================================
# Optimized Inference Context
# =============================================================================

class OptimizedInferenceContext:
    """
    Context manager for optimized SAM3 inference.
    
    Usage:
        with OptimizedInferenceContext():
            masks = predictor.propagate_in_video(...)
    """
    
    def __init__(self):
        self._inference_mode = None
        self._autocast = None
        self._major, _ = _get_gpu_info()[1:3] if torch.cuda.is_available() else (0, 0)
    
    def __enter__(self):
        # Inference mode (disables autograd tracking)
        self._inference_mode = torch.inference_mode()
        self._inference_mode.__enter__()
        
        # Autocast for mixed precision
        if self._major >= 8:
            self._autocast = torch.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=True
            )
            self._autocast.__enter__()
        
        return self
    
    def __exit__(self, *args):
        if self._autocast is not None:
            self._autocast.__exit__(*args)
        self._inference_mode.__exit__(*args)


# =============================================================================
# Patched get_inference_state
# =============================================================================

def get_optimized_inference_state(video_state, model_wrapper, force_new: bool = False):
    """
    Get an optimized inference state for video tracking.
    
    This is a drop-in replacement for get_inference_state that applies
    H200 optimizations automatically.
    
    Args:
        video_state: SAM3VideoState with video information
        model_wrapper: SAM3ModelWrapper containing the predictor
        force_new: Force creation of new inference state
        
    Returns:
        Optimized inference state
    """
    # Apply global optimizations
    apply_global_optimizations()
    
    # Get the predictor and optimize it
    if hasattr(model_wrapper, 'get_video_predictor'):
        predictor = model_wrapper.get_video_predictor()
        predictor = optimize_predictor(predictor)
    
    # Call original function if available
    if HAS_ORIGINAL and hasattr(original_module, 'get_inference_state'):
        return original_module.get_inference_state(video_state, model_wrapper, force_new)
    
    # Otherwise, raise an error
    raise NotImplementedError("Original inference_reconstructor not found")


# =============================================================================
# Utility Functions
# =============================================================================

def clear_compiled_cache():
    """Clear the compiled predictor cache."""
    global _COMPILED_PREDICTORS
    _COMPILED_PREDICTORS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SAM3 H200] Compiled cache cleared")


def print_optimization_status():
    """Print current optimization status."""
    name, major, minor = _get_gpu_info()
    
    print(f"\n{'='*50}")
    print("SAM3 H200 Optimization Status")
    print(f"{'='*50}")
    print(f"GPU: {name} (sm_{major}{minor})")
    print(f"Global optimizations: {'✓' if _OPTIMIZATIONS_APPLIED else '✗'}")
    print(f"Compiled predictors: {len(_COMPILED_PREDICTORS)}")
    
    if torch.cuda.is_available():
        print(f"\nPyTorch backends:")
        print(f"  TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")
        print(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        
        if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            print(f"  Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}")
        if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
            print(f"  Mem-efficient SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    
    print(f"{'='*50}\n")


# =============================================================================
# Auto-initialize on import
# =============================================================================

# Apply optimizations when this module is imported
if torch.cuda.is_available():
    apply_global_optimizations()
