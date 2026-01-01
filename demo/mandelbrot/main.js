function log(message) {
  const parent = document.getElementById('log');
  const item = document.createElement('pre');
  item.innerText = message;
  parent.appendChild(item);
}

const grid = 1024;

function getConfig() {
  let backend = document.querySelector('input[name="backend"]:checked').value;
  const use_gpu = backend !== 'cpu';
  if (!use_gpu) {
    backend = 'webgl';
  }
  const kernel_type = 'custom';

  return { backend, use_gpu, kernel_type, grid };
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

  // Canvas for zero-copy rendering
  let canvas = null;
  let ctx = null;

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
        const t0 = performance.now();
        // Zero-copy rendering directly to canvas
        let displayElement = document.getElementById('mandelbrotImage');
        
        // Debug: check buffer size
        const expectedSize = e.data.width * e.data.height * 4;
        const actualSize = e.data.buffer.byteLength;
        
        const t1 = performance.now();
        if (!canvas || canvas.width !== e.data.width || canvas.height !== e.data.height) {
          canvas = document.createElement('canvas');
          canvas.width = e.data.width;
          canvas.height = e.data.height;
          canvas.style.width = '512px';
          canvas.style.height = '512px';
          ctx = canvas.getContext('2d', { willReadFrequently: false });
          // Replace img/old canvas with new canvas
          displayElement.parentNode.replaceChild(canvas, displayElement);
          canvas.id = 'mandelbrotImage';
          
          // Reattach event listeners to the new canvas
          setupInteractionListeners();
        }
        
        const t2 = performance.now();
        const imageData = new ImageData(
          new Uint8ClampedArray(e.data.buffer),
          e.data.width,
          e.data.height
        );
        const t3 = performance.now();
        ctx.putImageData(imageData, 0, 0);
        const t4 = performance.now();
        
        isRendering = false; // Mark render complete
        console.log(`JS: setup ${(t1-t0).toFixed(1)}ms, canvas ${(t2-t1).toFixed(1)}ms, ImageData ${(t3-t2).toFixed(1)}ms, putImageData ${(t4-t3).toFixed(1)}ms, total ${(t4-t0).toFixed(1)}ms`);
        break;
    }
  });

  // after wgpy.initMain completed, wgpy.initWorker can be called in worker thread
  worker.postMessage({ namespace: 'app', method: 'start', config });
}

function render(x_min, x_max, y_min, y_max) {
  if (worker) {
    console.log({ namespace: 'app', method: 'render', grid, x_min, x_max, y_min, y_max })
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
