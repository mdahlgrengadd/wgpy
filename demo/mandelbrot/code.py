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
    from wgpy_backends.webgpu.webgpu_texture import WebGPUTexture
    from wgpy_backends.webgpu.colorize_kernel import ColorizeKernel
elif backend == "webgl":
    from wgpy_backends.webgl import get_performance_metrics
    from wgpy_backends.webgl.elementwise_kernel import ElementwiseKernel

_bs_kernels = {}
_rgba_kernels = {}


def customMandelbrotRGBA(R, I, n_iters):
    """Compute Mandelbrot iteration count on GPU"""
    if backend == "webgpu":
        # Single GPU kernel: compute iteration count only
        if (_rgba_kernel := _rgba_kernels.get(n_iters)) is None:
            _rgba_kernel = ElementwiseKernel(
                in_params="f32 real, f32 imag",
                out_params="i32 count",
                operation="""
// Compute Mandelbrot iteration count
var c: i32 = 0;
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
count = c;
                """,
                name=f"mandelbrot_iter_{n_iters}",
            )
            _rgba_kernels[n_iters] = _rgba_kernel
        
        # Return iteration counts - let count_to_rgba handle colorization
        return _rgba_kernel(R, I)
    elif backend == "webgl":
        # WebGL version TBD - use CPU fallback for now
        return None
    raise ValueError


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
        elif kernel_type == "custom_rgba":
            # GPU computes RGBA directly - single GPU->CPU transfer!
            ret = customMandelbrotRGBA(real, imag, n_iters)
            if ret is not None:
                # Already RGBA on GPU, just transfer to CPU
                return cp.asnumpy(ret)
            else:
                # Fallback to CPU path
                ret = mandelbrot(real, imag, n_iters)
                if use_gpu:
                    ret = cp.asnumpy(ret)
                return ret
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
    # Use GPU direct rendering if WebGPU backend and kernel_type is gpu_direct
    if use_gpu and backend == "webgpu" and kernel_type == "gpu_direct":
        visualize_gpu_direct(grid, real_min, real_max, imag_min, imag_max)
    else:
        # Existing CPU path
        real, imag = generate_input(grid, real_min, real_max, imag_min, imag_max)

        start_time = time.time()
        ret_gpu = run_once(real, imag, use_gpu, kernel_type)
        gpu_time = time.time()

        # If kernel_type is custom_rgba, ret_gpu is already RGBA
        if kernel_type == "custom_rgba" and ret_gpu.ndim == 3:
            rgba = ret_gpu
        else:
            rgba = count_to_rgba(ret_gpu)
        color_time = time.time()

        display_image(rgba)


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


def create_gpu_color_map():
    """Upload color map to GPU as packed RGBA u32 buffer"""
    rgba_packed = np.empty(n_iters + 1, dtype=np.uint32)
    for i in range(n_iters + 1):
        r, g, b = color_map[i]
        # Pack as little-endian ABGR (0xAABBGGRR)
        rgba_packed[i] = np.uint32((255 << 24) | (b << 16) | (g << 8) | r)
    return cp.asarray(rgba_packed)


# Initialize GPU rendering components if using WebGPU
gpu_color_map = None
colorize_kernel_instance = None
render_texture = None

if use_gpu and backend == "webgpu":
    gpu_color_map = create_gpu_color_map()
    colorize_kernel_instance = ColorizeKernel(n_iters + 1)


def count_to_rgba(count):
    # Use pre-computed color map with optimized indexing
    height, width = count.shape
    
    # Clamp count values to valid range [0, n_iters]
    count_clamped = np.minimum(count, n_iters)
    
    # Create RGBA array directly
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    
    # Use take to index into color map (faster than fancy indexing)
    colors = np.take(color_map, count_clamped, axis=0)
    rgba[:, :, :3] = colors.reshape(height, width, 3)
    rgba[:, :, 3] = 255
    
    return rgba


def visualize_gpu_direct(grid: int, real_min=-2.0, real_max=0.5, imag_min=-1.2, imag_max=1.2):
    """GPU-to-canvas rendering path (zero GPU→CPU transfer)"""
    global render_texture

    real, imag = generate_input(grid, real_min, real_max, imag_min, imag_max)

    # Step 1: Compute iteration counts on GPU
    real_gpu = cp.asarray(real)
    imag_gpu = cp.asarray(imag)
    count_gpu = customMandelbrot(real_gpu, imag_gpu, n_iters)  # (grid, grid) int32

    # Step 2: Colorize on GPU (iteration counts → RGBA)
    rgba_gpu = colorize_kernel_instance(count_gpu, gpu_color_map)  # (grid, grid) uint32

    # Step 3: Create or reuse texture
    if render_texture is None or render_texture.width != grid or render_texture.height != grid:
        if render_texture is not None:
            del render_texture
        render_texture = WebGPUTexture(grid, grid, "rgba8unorm")

    # Step 4: Copy GPU buffer to texture
    render_texture.copy_from_buffer(rgba_gpu.buffer.buffer_id)

    # Step 5: Present texture to canvas
    render_texture.present()

    # Signal completion to main thread (enables continuous rendering during drag)
    pythonIO.displayImageRaw(np.empty((1, 1, 4), dtype=np.uint8), 1, 1)


def display_image(rgba_array):
    # Zero-copy approach: send raw RGBA pixel data directly
    # Pass the numpy array itself, JavaScript will extract the buffer
    pythonIO.displayImageRaw(rgba_array, rgba_array.shape[1], rgba_array.shape[0])
