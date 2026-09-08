import logging
from pathlib import Path

from text_embeddings_server.utils.device import is_rocm

logger = logging.getLogger(__name__)

_LOCKFILE = Path(__file__).resolve().parents[2] / "kernels.lock"

_triton_layer_norm = None
if is_rocm():
    try:
        # `load_kernel` loads a pre-downloaded, locked kernel from the local
        # cache with `local_files_only=True`, so the container never reaches
        # out to the Hub at runtime. The kernel is fetched at build time via
        # `kernels download` (see Dockerfile-rocm) into `KERNELS_CACHE`.
        from kernels import load_kernel

        _triton_layer_norm = load_kernel(
            "kernels-community/triton-layer-norm", lockfile=_LOCKFILE
        )
    except Exception as e:
        logger.warning(
            f"Could not load the triton-layer-norm kernel, falling back to "
            f"torch layer norm: {e}"
        )


def get_triton_layer_norm():
    """Return the pre-downloaded triton-layer-norm kernel, or None if unavailable."""
    return _triton_layer_norm
