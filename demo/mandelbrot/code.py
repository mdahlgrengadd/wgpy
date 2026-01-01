from js import pythonIO
import numpy as np
import cupy as cp
import cupyx.scipy
import time

use_gpu = pythonIO.config.use_gpu
kernel_type = pythonIO.config.kernel_type
np.random.seed(0)

n_iters = 500


def mandelbrot(real, imag, n_iters):
    xp = cp.get_array_module(real)
    xs = xp.zeros((real.size, imag.size), dtype=np.float32)
    ys = xp.zeros((real.size, imag.size), dtype=np.float32)
    count = xp.zeros((real.size, imag.size), dtype=np.int32)
    for _ in range(n_iters):
        xs, ys = xs * xs - ys * ys + real, xs * ys * 2.0 + imag
        count += ((xs * xs + ys * ys) < 4.0).astype(np.int32)
    return count


### begin custom kernel

backend = cp.get_backend_name()
if backend == "webgpu":
    from wgpy_backends.webgpu import get_performance_metrics
    from wgpy_backends.webgpu.elementwise_kernel import ElementwiseKernel
elif backend == "webgl":
    from wgpy_backends.webgl import get_performance_metrics
    from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

_bs_kernels = {}


def customMandelbrot(R, I, n_iters):
    if backend == "webgpu":
        if (_bs_kernel := _bs_kernels.get(n_iters)) is None:
            _bs_kernel = ElementwiseKernel(
                in_params="f32 real, f32 imag",
                out_params="i32 c",
                operation="""
c = 0;
var x: f32 = 0.0;
var y: f32 = 0.0;
for(var k: u32 = 0u; k < """
                + str(n_iters)
                + """u; k = k + 1u) {
    var nx: f32 = x * x - y * y + real;
    var ny: f32 = x * y * 2.0 + imag;
    x = nx;
    y = ny;
    if (x * x + y * y < 4.0) {
        c = c + 1;
    }
}
                """,
                name=f"mandelbrot_{n_iters}",
            )
            _bs_kernels[n_iters] = _bs_kernel
        return _bs_kernel(R, I)
    elif backend == "webgl":
        if (_bs_kernel := _bs_kernels.get(n_iters)) is None:
            _bs_kernel = ElementwiseKernel(
                in_params="float real, float imag",
                out_params="int c",
                operation="""
c = 0;
float x = 0.0;
float y = 0.0;
for (int k = 0; k < """
                + str(n_iters)
                + """; k++) {
    float nx = x * x - y * y + real;
    float ny = x * y * 2.0 + imag;
    x = nx;
    y = ny;
    if (x * x + y * y < 4.0) {
        c = c + 1;
    }
}
                """,
                name=f"mandelbrot_{n_iters}",
            )
            _bs_kernels[n_iters] = _bs_kernel
        return _bs_kernel(R, I)
    raise ValueError


### end custom kernel


def run_once(real, imag, use_gpu, kernel_type):
    if use_gpu:
        real = cp.asarray(real)
        imag = cp.asarray(imag)
    if use_gpu:
        if kernel_type == "normal":
            ret = mandelbrot(real, imag, n_iters)
        elif kernel_type == "custom":
            ret = customMandelbrot(real, imag, n_iters)
        else:
            raise ValueError
    else:
        ret = mandelbrot(real, imag, n_iters)
    if use_gpu:
        ret = cp.asnumpy(ret)
    return ret


def generate_input(grid: int, real_min=-2.0, real_max=0.5, imag_min=-1.2, imag_max=1.2):
    real = np.linspace(real_min, real_max, grid, dtype=np.float32)[np.newaxis, :]
    imag = np.linspace(imag_min, imag_max, grid, dtype=np.float32)[:, np.newaxis]
    return real, imag


def visualize(grid: int, real_min=-2.0, real_max=0.5, imag_min=-1.2, imag_max=1.2):
    print(f"visualizeing grid={grid}")
    real, imag = generate_input(grid, real_min, real_max, imag_min, imag_max)
    
    start_time = time.time()
    ret_gpu = run_once(real, imag, use_gpu, kernel_type)
    gpu_time = time.time()
    
    rgba = count_to_rgba(ret_gpu)
    color_time = time.time()
    
    display_image(rgba)
    display_time = time.time()
    
    print(f"GPU compute: {(gpu_time - start_time)*1000:.1f}ms, Color map: {(color_time - gpu_time)*1000:.1f}ms, Display: {(display_time - color_time)*1000:.1f}ms, Total: {(display_time - start_time)*1000:.1f}ms")


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=1)


def generate_color_map():
    hsv = np.array(
        [
            np.linspace(0.66, 0.0, n_iters + 1),
            np.full(n_iters + 1, 0.9),
            np.full(n_iters + 1, 0.95),
        ]
    ).T
    rgb = hsv_to_rgb(hsv)
    rgb[n_iters] = [1, 1, 1]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


color_map = generate_color_map()


def count_to_rgba(count):
    # Direct color mapping - optimized vectorized version
    height, width = count.shape
    
    # Normalize count to 0-1 range
    t = count.astype(np.float32) / n_iters
    
    # HSV to RGB conversion on the full array
    h = 0.66 * (1.0 - t)  # Hue from blue to red
    s = 0.9
    v = 0.95
    
    # Set points at max iterations to white
    mask = (count == n_iters)
    h[mask] = 0
    s_arr = np.where(mask, 0, s)
    v_arr = np.where(mask, 1.0, v)
    
    # HSV to RGB vectorized (optimized)
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - i
    p = v_arr * (1.0 - s_arr)
    q = v_arr * (1.0 - f * s_arr)
    t_hsv = v_arr * (1.0 - (1.0 - f) * s_arr)
    
    # Direct assignment based on i value (faster than np.select)
    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)
    
    m0 = (i == 0); r[m0] = v_arr[m0]; g[m0] = t_hsv[m0]; b[m0] = p[m0]
    m1 = (i == 1); r[m1] = q[m1]; g[m1] = v_arr[m1]; b[m1] = p[m1]
    m2 = (i == 2); r[m2] = p[m2]; g[m2] = v_arr[m2]; b[m2] = t_hsv[m2]
    m3 = (i == 3); r[m3] = p[m3]; g[m3] = q[m3]; b[m3] = v_arr[m3]
    m4 = (i == 4); r[m4] = t_hsv[m4]; g[m4] = p[m4]; b[m4] = v_arr[m4]
    m5 = (i == 5); r[m5] = v_arr[m5]; g[m5] = p[m5]; b[m5] = q[m5]
    
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = (r * 255.0).astype(np.uint8)
    rgba[:, :, 1] = (g * 255.0).astype(np.uint8)
    rgba[:, :, 2] = (b * 255.0).astype(np.uint8)
    rgba[:, :, 3] = 255
    
    return rgba


def display_image(rgba_array):
    # Zero-copy approach: send raw RGBA pixel data directly
    # Pass the numpy array itself, JavaScript will extract the buffer
    print(f"Array shape: {rgba_array.shape}, size: {rgba_array.size}, dtype: {rgba_array.dtype}")
    pythonIO.displayImageRaw(rgba_array, rgba_array.shape[1], rgba_array.shape[0])
