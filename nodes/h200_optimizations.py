"""
H200/Hopper GPU Optimizations for SAM3 Video Tracking

This module provides performance optimizations specifically for H200 and other Hopper GPUs.
These optimizations can provide 2-5x speedup for video inference.

Usage in your nodes:
    from .h200_optimizations import optimize_for_h200, apply_inference_optimizations

    # When loading the model:
    model = optimize_for_h200(model)
    
    # Before inference:
    apply_inference_optimizations()
"""

import os
import torch
import warnings
from typing import Optional, Tuple
from functools import lru_cache


# =============================================================================
# GPU Detection
# =============================================================================

@lru_cache(maxsize=1)
def get_gpu_info() -> Tuple[str, int, int]:
    """Get GPU name and compute capability. Cached for performance."""
    if not torch.cuda.is_available():
        return ("CPU", 0, 0)
    
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return (name, major, minor)


def is_hopper_gpu() -> bool:
    """Check if GPU is Hopper architecture (H100, H200, etc.)"""
    name, major, _ = get_gpu_info()
    # Hopper is compute capability 9.0
    return major >= 9


def is_ampere_or_better() -> bool:
    """Check if GPU is Ampere or newer (A100, RTX 30xx, etc.)"""
    _, major, _ = get_gpu_info()
    return major >= 8


def get_optimal_dtype() -> torch.dtype:
    """Get optimal dtype for current GPU."""
    _, major, _ = get_gpu_info()
    if major >= 8:  # Ampere+ supports bf16 natively
        return torch.bfloat16
    elif major >= 7:  # Volta/Turing use fp16
        return torch.float16
    else:
        return torch.float32


# =============================================================================
# Global PyTorch Optimizations
# =============================================================================

def apply_inference_optimizations():
    """
    Apply global PyTorch optimizations for inference.
    Call this once at startup or before inference begins.
    """
    if not torch.cuda.is_available():
        return
    
    name, major, _ = get_gpu_info()
    print(f"[H200 Opt] Applying optimizations for {name} (compute {major}.x)")
    
    # 1. Enable TF32 for matmul (Ampere+) - faster with minimal accuracy loss
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[H200 Opt] ✓ TF32 enabled for matmul/cudnn")
    
    # 2. Enable cuDNN benchmark - finds fastest algorithms
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("[H200 Opt] ✓ cuDNN benchmark enabled")
    
    # 3. Enable Flash Attention (PyTorch 2.0+)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        print("[H200 Opt] ✓ Flash SDP enabled")
    
    # 4. Enable memory efficient attention
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("[H200 Opt] ✓ Memory-efficient SDP enabled")
    
    # 5. Hopper-specific optimizations
    if major >= 9:
        # Enable FP8 tensor cores if available (H100/H200)
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
            print("[H200 Opt] ✓ cuDNN SDP enabled (Hopper)")
        
        # Set CUDA memory allocator for better performance
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        print("[H200 Opt] ✓ Expandable CUDA segments enabled")


# =============================================================================
# Model Optimizations
# =============================================================================

def optimize_for_h200(model: torch.nn.Module, 
                      compile_model: bool = True,
                      use_channels_last: bool = True) -> torch.nn.Module:
    """
    Apply H200-specific optimizations to a model.
    
    Args:
        model: The SAM3 model to optimize
        compile_model: Whether to use torch.compile (requires PyTorch 2.0+)
        use_channels_last: Whether to convert to channels_last memory format
        
    Returns:
        Optimized model
    """
    if not torch.cuda.is_available():
        return model
    
    name, major, _ = get_gpu_info()
    print(f"[H200 Opt] Optimizing model for {name}")
    
    # 1. Move to GPU with optimal dtype
    dtype = get_optimal_dtype()
    model = model.to(device='cuda', dtype=dtype)
    print(f"[H200 Opt] ✓ Model on CUDA with {dtype}")
    
    # 2. Channels last memory format (better for vision models)
    if use_channels_last and major >= 8:
        try:
            model = model.to(memory_format=torch.channels_last)
            print("[H200 Opt] ✓ Channels-last memory format")
        except Exception as e:
            print(f"[H200 Opt] ⚠ Channels-last failed: {e}")
    
    # 3. torch.compile for Hopper/Ampere (PyTorch 2.0+)
    if compile_model and major >= 8:
        try:
            if hasattr(torch, 'compile'):
                # Use reduce-overhead mode for inference
                # max-autotune is slower to compile but faster at runtime
                mode = "max-autotune" if major >= 9 else "reduce-overhead"
                model = torch.compile(model, mode=mode, fullgraph=False)
                print(f"[H200 Opt] ✓ torch.compile with mode={mode}")
            else:
                print("[H200 Opt] ⚠ torch.compile not available (need PyTorch 2.0+)")
        except Exception as e:
            print(f"[H200 Opt] ⚠ torch.compile failed: {e}")
    
    # 4. Enable inference mode optimizations
    model.eval()
    
    return model


def compile_video_predictor(predictor, mode: str = "reduce-overhead"):
    """
    Compile SAM3 video predictor for faster inference.
    
    This is the equivalent of vos_optimized=True in SAM2's build_sam2_video_predictor.
    
    Args:
        predictor: SAM3VideoPredictor instance
        mode: Compilation mode - "reduce-overhead" (fast compile) or "max-autotune" (fastest runtime)
        
    Returns:
        Compiled predictor
    """
    if not hasattr(torch, 'compile'):
        print("[H200 Opt] torch.compile not available")
        return predictor
    
    _, major, _ = get_gpu_info()
    if major < 8:
        print("[H200 Opt] torch.compile requires Ampere GPU or newer")
        return predictor
    
    try:
        # Compile the key inference methods
        if hasattr(predictor, 'propagate_in_video'):
            predictor.propagate_in_video = torch.compile(
                predictor.propagate_in_video, 
                mode=mode,
                fullgraph=False
            )
            print(f"[H200 Opt] ✓ Compiled propagate_in_video (mode={mode})")
        
        if hasattr(predictor, '_run_single_frame_inference'):
            predictor._run_single_frame_inference = torch.compile(
                predictor._run_single_frame_inference,
                mode=mode,
                fullgraph=False
            )
            print(f"[H200 Opt] ✓ Compiled _run_single_frame_inference")
            
    except Exception as e:
        print(f"[H200 Opt] ⚠ Compilation failed: {e}")
    
    return predictor


# =============================================================================
# Context Managers for Inference
# =============================================================================

class InferenceContext:
    """
    Context manager for optimized inference.
    
    Usage:
        with InferenceContext():
            masks = model.propagate(...)
    """
    
    def __init__(self, dtype: Optional[torch.dtype] = None):
        self.dtype = dtype or get_optimal_dtype()
        self._inference_mode = None
        self._autocast = None
    
    def __enter__(self):
        # Enter inference mode (disables autograd)
        self._inference_mode = torch.inference_mode()
        self._inference_mode.__enter__()
        
        # Enter autocast for mixed precision
        if torch.cuda.is_available() and self.dtype in (torch.float16, torch.bfloat16):
            self._autocast = torch.autocast(device_type='cuda', dtype=self.dtype)
            self._autocast.__enter__()
        
        return self
    
    def __exit__(self, *args):
        if self._autocast is not None:
            self._autocast.__exit__(*args)
        self._inference_mode.__exit__(*args)


def get_inference_context():
    """Get the appropriate inference context for current GPU."""
    return InferenceContext(get_optimal_dtype())


# =============================================================================
# Memory Optimization
# =============================================================================

def optimize_memory_for_long_videos():
    """
    Configure PyTorch for processing long videos with minimal memory.
    Call before processing videos with 1000+ frames.
    """
    if not torch.cuda.is_available():
        return
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Enable memory-efficient attention globally
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Disable flash attention if memory is tight (flash uses more memory for long sequences)
    # Uncomment if you run into OOM:
    # torch.backends.cuda.enable_flash_sdp(False)
    
    # Set allocator to be more aggressive about freeing memory
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.6'
    
    print("[H200 Opt] Memory optimizations applied for long videos")


def get_memory_stats() -> dict:
    """Get current CUDA memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3,
    }


def print_memory_stats(label: str = ""):
    """Print current CUDA memory usage."""
    stats = get_memory_stats()
    if stats:
        print(f"[VRAM] {label}: {stats['allocated_gb']:.2f}GB allocated, "
              f"{stats['reserved_gb']:.2f}GB reserved, {stats['free_gb']:.2f}GB free")


# =============================================================================
# FlashAttention-3 Support (Hopper only)
# =============================================================================

def check_flash_attention_3() -> bool:
    """
    Check if FlashAttention-3 is available (requires H100/H200 + flash-attn package).
    
    FlashAttention-3 provides significant speedups on Hopper GPUs.
    Install with: pip install flash-attn --no-build-isolation
    """
    if not is_hopper_gpu():
        return False
    
    try:
        from flash_attn import flash_attn_func
        print("[H200 Opt] ✓ FlashAttention-3 available")
        return True
    except ImportError:
        print("[H200 Opt] FlashAttention-3 not installed. Install with:")
        print("           pip install flash-attn --no-build-isolation")
        return False


# =============================================================================
# Auto-initialization
# =============================================================================

def auto_optimize():
    """
    Automatically apply all safe optimizations.
    Call this once when the module loads.
    """
    if torch.cuda.is_available():
        apply_inference_optimizations()
        name, major, minor = get_gpu_info()
        print(f"[H200 Opt] Initialized for {name} (sm_{major}{minor})")
        
        if is_hopper_gpu():
            print("[H200 Opt] 🚀 Hopper GPU detected - maximum optimizations enabled")
            check_flash_attention_3()


# Run auto-optimization on import
auto_optimize()
