import cupy as cp
import numpy as np
from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel


class ColorizeKernel:
    """Generic kernel to colorize scalar values using a color map"""

    def __init__(self, n_values: int):
        """
        Args:
            n_values: Number of entries in color map (e.g., n_iters + 1)
        """
        self.n_values = n_values

        # Use raw access to color map buffer
        self.kernel = ElementwiseKernel(
            in_params="i32 count, raw u32 color_map",
            out_params="u32 rgba",
            operation=f"""
// Clamp count to valid range
var idx: i32 = clamp(count, 0, {n_values - 1});

// Color map stores packed RGBA u32 values
rgba = color_map(idx);
            """,
            name=f"colorize_rgba_{n_values}",
        )

    def __call__(self, count_array, color_map_packed):
        """
        Args:
            count_array: cp.ndarray of int32 iteration counts (H, W)
            color_map_packed: cp.ndarray of uint32 packed RGBA (n_values,)

        Returns:
            cp.ndarray of uint32 packed RGBA values (H, W)
        """
        return self.kernel(count_array, color_map_packed)
