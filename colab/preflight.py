import platform
import sys
from typing import Any


class PreflightValidator:
    def check_python(self) -> dict[str, str]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }

    def check_torch(self) -> dict[str, Any]:
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is not installed.")

        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            raise RuntimeError("CUDA is not available. XTTS requires a GPU.")

        return {
            "torch": torch.__version__,
            "cuda": cuda_available,
        }

    def check_gpu(self) -> dict[str, Any]:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Cannot check GPU because CUDA is unavailable.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        
        # Convert bytes to GB
        total_memory_gb = gpu_props.total_memory / (1024**3)

        return {
            "gpu": gpu_name,
            "gpu_memory_gb": total_memory_gb,
        }

    def check_tts(self) -> dict[str, str]:
        try:
            import TTS
            return {"tts": getattr(TTS, "__version__", "unknown")}
        except ImportError:
            raise RuntimeError("TTS is not installed.")

    def run(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        
        results.update(self.check_python())
        results.update(self.check_torch())
        results.update(self.check_gpu())
        results.update(self.check_tts())
        
        return results
