function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

const gridWidth = 1280;
const gridHeight = 720;
const grid = 1280; // Grid size for worker rendering

function getConfig() {
  let backend = document.querySelector('input[name="backend"]:checked').value;
  const use_gpu = backend !== 'cpu';
  if (!use_gpu) {
    backend = 'webgl';
  }
  // Use custom_rgba for direct GPU RGBA output (eliminates CPU color conversion)
  const kernel_type = use_gpu ? 'custom_rgba' : 'custom';

  return { backend, use_gpu, kernel_type, gridWidth, gridHeight };
}

let worker;
async function run() {
  const config = getConfig();
  worker = new Worker('worker.js');

  log('Initializing wgpy main-thread-side javascript interface');
  let initializedBackend = 'cpu';
  if (typeof wgpy !== 'undefined') {
    try {
      const initResult = await wgpy.initMain(worker, { backendOrder: [config.backend] });
      initializedBackend = initResult.backend;
    } catch (e) {
      log(`wgpy initialization failed: ${e.message}`);
    }
  } else {
    log('wgpy not loaded, GPU features not available');
  }
  config.backend = initializedBackend; // actually initialized backend
  log(`Initialized backend: ${initializedBackend}`);

  // Canvas for rendering
  let canvas = null;
  let ctx = null;

  // Setup WebGPU canvas if using GPU direct rendering
  if (initializedBackend === 'webgpu' && config.use_gpu) {
    canvas = document.createElement('canvas');
    canvas.width = gridWidth;
    canvas.height = gridHeight;
    canvas.style.width = '1280px';
    canvas.style.height = '720px';

    // Replace img with canvas (keep same ID for event listeners)
    let displayElement = document.getElementById('mandelbrotImage');
    if (displayElement) {
      displayElement.parentNode.replaceChild(canvas, displayElement);
      canvas.id = 'mandelbrotImage';  // Use same ID so setupInteractionListeners works
    }

    // Initialize WebGPU canvas context
    if (typeof wgpy.initCanvasContext === 'function') {
      try {
        await wgpy.initCanvasContext(canvas);
        config.kernel_type = 'gpu_direct';
        log('GPU direct rendering enabled');
      } catch (e) {
        log(`Failed to initialize canvas context: ${e.message}`);
        // Fall back to CPU colorization
      }
    }

    setupInteractionListeners();
  }

  worker.addEventListener('message', (e) => {
    if (e.data.namespace !== 'app') {
      // message for library
      return;
    }

    switch (e.data.method) {
      case 'log':
        log(e.data.message);
        break;
      case 'displayImage':
        document.getElementById('mandelbrotImage').src = e.data.url;
        break;
      case 'displayImageRaw':
        // Skip ImageData rendering if using GPU direct mode
        if (config.kernel_type === 'gpu_direct') {
          isRendering = false;
          break;
        }

        const t0 = performance.now();
        // True zero-copy rendering using SharedArrayBuffer
        let displayElement = document.getElementById('mandelbrotImage');

        if (!canvas || canvas.width !== e.data.width || canvas.height !== e.data.height) {
          canvas = document.createElement('canvas');
          canvas.width = e.data.width;
          canvas.height = e.data.height;
          canvas.style.width = '1280px';
          canvas.style.height = '720px';
          ctx = canvas.getContext('2d', { willReadFrequently: false });
          // Replace img/old canvas with new canvas
          if (displayElement) {
            displayElement.parentNode.replaceChild(canvas, displayElement);
          }
          canvas.id = 'mandelbrotImage';

          // Reattach event listeners to the new canvas
          setupInteractionListeners();
        }

        // Create ImageData from transferred buffer
        const imageData = new ImageData(new Uint8ClampedArray(e.data.buffer), e.data.width, e.data.height);

        // Render to canvas
        ctx.putImageData(imageData, 0, 0);

        isRendering = false; // Mark render complete
        break;
    }
  });

  // after wgpy.initMain completed, wgpy.initWorker can be called in worker thread
  worker.postMessage({ namespace: 'app', method: 'start', config });
}

function render(x_min, x_max, y_min, y_max) {
  if (worker) {
    worker.postMessage({ namespace: 'app', method: 'render', grid, x_min, x_max, y_min, y_max });
  }
}

function debounce(func, delay) {
  let timerId;

  return function(...args) {
    clearTimeout(timerId);
    timerId = setTimeout(() => {
      func.apply(this, args);
    }, delay);
  };
}

const debouncedRender = debounce(render, 100); // Reduced from 300ms to 100ms

let x_min = -2.0, x_max = 0.5, y_min = -1.2, y_max = 1.2;
let isDragging = false;
let isRendering = false; // Track if render is in progress
let startX, startY;
let startXMin, startXMax, startYMin, startYMax;

function setupInteractionListeners() {
  const displayElement = document.getElementById("mandelbrotImage");
  
  // Attach listeners directly (they'll be removed when element is replaced)
  displayElement.addEventListener("mousedown", (e) => {
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    startXMin = x_min;
    startXMax = x_max;
    startYMin = y_min;
    startYMax = y_max;
  });

  displayElement.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    const dx = (e.clientX - startX) / displayElement.width * (x_max - x_min);
    const dy = (e.clientY - startY) / displayElement.height * (y_max - y_min);

    x_min = startXMin - dx;
    x_max = startXMax - dx;
    y_min = startYMin - dy;
    y_max = startYMax - dy;
    
    // Render continuously during drag, but only if previous render is complete
    if (!isRendering) {
      isRendering = true;
      render(x_min, x_max, y_min, y_max);
    }
  });

  displayElement.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      render(x_min, x_max, y_min, y_max); // Immediate render on mouse up
    }
  });

  displayElement.addEventListener("mouseleave", () => {
    if (isDragging) {
      isDragging = false;
      render(x_min, x_max, y_min, y_max); // Immediate render on mouse leave
    }
  });

  displayElement.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = 1.1;
    const scale = e.deltaY > 0 ? zoomFactor : 1 / zoomFactor;

    const mouseX = e.offsetX / displayElement.width;
    const mouseY = e.offsetY / displayElement.height;

    const x_center = x_min + mouseX * (x_max - x_min);
    const y_center = y_min + (1 - mouseY) * (y_max - y_min);

    const new_width = (x_max - x_min) * scale;
    const new_height = (y_max - y_min) * scale;

    x_min = x_center - mouseX * new_width;
    x_max = x_min + new_width;
    y_min = y_center - (1 - mouseY) * new_height;
    y_max = y_min + new_height;

    debouncedRender(x_min, x_max, y_min, y_max);
  });
}

window.addEventListener('load', () => {
  document.getElementById('run').onclick = () => {
    document.getElementById('run').disabled = true;
    run().catch((error) => {
      log(`Main thread error: ${error.message}`);
    });
  };

  if (navigator.gpu) {
    document.querySelector('input[name="backend"][value="webgpu"]').checked = true;
  } else {
    document.querySelector('input[name="backend"][value="webgl"]').checked = true;
  }

  // Set up initial interaction listeners
  setupInteractionListeners();

  document.getElementById('resetZoom').addEventListener("click", (e) => {
    x_min = -2.0;
    x_max = 0.5;
    y_min = -1.2;
    y_max = 1.2;
    render(x_min, x_max, y_min, y_max);
  });
});
