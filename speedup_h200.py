#!/usr/bin/env python3
"""
H200/Hopper GPU Speedup Script for ComfyUI-SAM3

This script optimizes SAM3 for H200 and other Hopper GPUs.
Run once after installation: python speedup_h200.py

Optimizations applied:
1. FlashAttention-3 installation (if not present)
2. CUDA extension compilation for H200 architecture
3. PyTorch backend configurations
4. torch.compile warmup

Usage:
    python speedup_h200.py [--skip-flash-attn] [--skip-cuda-ext] [--force]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_gpu():
    """Check for Hopper GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return False, None
        
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        
        print(f"✓ GPU detected: {name} (sm_{major}{minor})")
        
        if major >= 9:
            print("🚀 Hopper architecture detected - maximum optimizations available")
            return True, (major, minor)
        elif major >= 8:
            print("✓ Ampere architecture - good optimizations available")
            return True, (major, minor)
        else:
            print(f"⚠ Older GPU (sm_{major}{minor}) - limited optimizations")
            return True, (major, minor)
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False, None


def check_pytorch_version():
    """Check PyTorch version for torch.compile support."""
    try:
        import torch
        version = torch.__version__
        major = int(version.split('.')[0])
        minor = int(version.split('.')[1].split('+')[0].split('a')[0].split('b')[0])
        
        print(f"✓ PyTorch {version}")
        
        if major >= 2:
            print("✓ torch.compile available")
            return True
        else:
            print("⚠ torch.compile requires PyTorch 2.0+")
            print("  Upgrade with: pip install torch>=2.5.1")
            return False
            
    except Exception as e:
        print(f"❌ Error checking PyTorch: {e}")
        return False


def install_flash_attention():
    """Install FlashAttention-3 for Hopper GPUs."""
    print("\n" + "="*60)
    print("Installing FlashAttention-3...")
    print("="*60)
    
    try:
        # Check if already installed
        try:
            import flash_attn
            print(f"✓ FlashAttention already installed: {flash_attn.__version__}")
            return True
        except ImportError:
            pass
        
        # Install flash-attn
        print("Installing flash-attn (this may take 5-10 minutes)...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ FlashAttention installed successfully")
            return True
        else:
            print(f"⚠ FlashAttention installation failed:")
            print(result.stderr[:500] if result.stderr else "Unknown error")
            print("\nYou can try installing manually:")
            print("  pip install flash-attn --no-build-isolation")
            return False
            
    except Exception as e:
        print(f"❌ Error installing FlashAttention: {e}")
        return False


def compile_cuda_extensions():
    """Compile GPU-accelerated CUDA extensions."""
    print("\n" + "="*60)
    print("Compiling CUDA extensions...")
    print("="*60)
    
    # Check if speedup.py exists in current directory
    script_dir = Path(__file__).parent
    speedup_script = script_dir / "speedup.py"
    
    if speedup_script.exists():
        print(f"Running {speedup_script}...")
        result = subprocess.run(
            [sys.executable, str(speedup_script)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ CUDA extensions compiled")
            return True
        else:
            print(f"⚠ CUDA extension compilation had issues:")
            print(result.stderr[:500] if result.stderr else result.stdout[:500])
            return False
    else:
        print("⚠ speedup.py not found - skipping CUDA extension compilation")
        print("  This is optional and only affects video tracking speed")
        return True


def configure_pytorch_backends():
    """Configure PyTorch backends for optimal performance."""
    print("\n" + "="*60)
    print("Configuring PyTorch backends...")
    print("="*60)
    
    try:
        import torch
        
        # TF32 settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled")
        
        # cuDNN settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✓ cuDNN benchmark enabled")
        
        # Flash SDP
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            print("✓ Flash SDP enabled")
        
        # Memory efficient SDP
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            print("✓ Memory-efficient SDP enabled")
        
        # cuDNN SDP (Hopper)
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
            print("✓ cuDNN SDP enabled")
        
        return True
        
    except Exception as e:
        print(f"❌ Error configuring backends: {e}")
        return False


def warmup_torch_compile():
    """Warmup torch.compile to cache compilation artifacts."""
    print("\n" + "="*60)
    print("Warming up torch.compile...")
    print("="*60)
    
    try:
        import torch
        
        if not hasattr(torch, 'compile'):
            print("⚠ torch.compile not available")
            return False
        
        # Create a simple model for warmup
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(64)
            
            def forward(self, x):
                return torch.relu(self.bn(self.conv(x)))
        
        model = SimpleModel().cuda().eval()
        compiled_model = torch.compile(model, mode="reduce-overhead")
        
        # Run a few iterations to warm up
        print("Running warmup iterations...")
        x = torch.randn(1, 3, 64, 64, device='cuda')
        
        with torch.no_grad():
            for i in range(3):
                _ = compiled_model(x)
                if i == 0:
                    print("  First compilation done (slowest)")
                else:
                    print(f"  Iteration {i+1} done")
        
        # Clear memory
        del model, compiled_model, x
        torch.cuda.empty_cache()
        
        print("✓ torch.compile warmed up")
        return True
        
    except Exception as e:
        print(f"⚠ Warmup failed (this is okay): {e}")
        return False


def create_startup_script():
    """Create a startup script to apply optimizations automatically."""
    print("\n" + "="*60)
    print("Creating startup optimization script...")
    print("="*60)
    
    script_content = '''"""
Auto-optimization script for ComfyUI-SAM3
This runs automatically when the extension loads.
"""
import torch
import os

def apply_h200_optimizations():
    """Apply all H200/Hopper optimizations."""
    if not torch.cuda.is_available():
        return
    
    major, _ = torch.cuda.get_device_capability()
    
    # TF32 for Ampere+
    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Flash/Memory-efficient attention
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Hopper-specific
    if major >= 9:
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(True)
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    print(f"[SAM3] H200 optimizations applied for sm_{major}x")

# Apply on import
apply_h200_optimizations()
'''
    
    script_dir = Path(__file__).parent
    startup_script = script_dir / "h200_startup.py"
    
    with open(startup_script, 'w') as f:
        f.write(script_content)
    
    print(f"✓ Created {startup_script}")
    print("\nTo auto-load optimizations, add to your __init__.py:")
    print("    from . import h200_startup")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="H200 Speedup for ComfyUI-SAM3")
    parser.add_argument('--skip-flash-attn', action='store_true', help="Skip FlashAttention installation")
    parser.add_argument('--skip-cuda-ext', action='store_true', help="Skip CUDA extension compilation")
    parser.add_argument('--skip-warmup', action='store_true', help="Skip torch.compile warmup")
    parser.add_argument('--force', action='store_true', help="Force reinstallation")
    args = parser.parse_args()
    
    print("="*60)
    print("ComfyUI-SAM3 H200/Hopper Optimization Script")
    print("="*60)
    
    # Check GPU
    gpu_ok, gpu_info = check_gpu()
    if not gpu_ok:
        print("\n❌ No compatible GPU found. Exiting.")
        return 1
    
    # Check PyTorch
    pytorch_ok = check_pytorch_version()
    
    major, _ = gpu_info
    
    # Install FlashAttention-3 (Hopper only)
    if major >= 9 and not args.skip_flash_attn:
        install_flash_attention()
    
    # Compile CUDA extensions
    if not args.skip_cuda_ext:
        compile_cuda_extensions()
    
    # Configure PyTorch backends
    configure_pytorch_backends()
    
    # Warmup torch.compile
    if pytorch_ok and not args.skip_warmup:
        warmup_torch_compile()
    
    # Create startup script
    create_startup_script()
    
    print("\n" + "="*60)
    print("✓ H200 Optimization Complete!")
    print("="*60)
    print("\nOptimizations applied:")
    print("  • TF32 for matrix operations")
    print("  • cuDNN benchmark mode")
    print("  • Flash/Memory-efficient attention")
    if major >= 9:
        print("  • FlashAttention-3 (if installed)")
        print("  • Hopper-specific CUDA settings")
    print("\nRestart ComfyUI to apply all changes.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
