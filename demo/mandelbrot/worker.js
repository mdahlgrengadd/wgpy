// Load Pyodide from CDN
const PYODIDE_VERSION = 'v0.26.4';
importScripts(`https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/pyodide.js`);

// Try to load wgpy-worker.js if available (for GPU support)
try {
  importScripts('../../dist/wgpy-worker.js');
} catch (e) {
  console.log('wgpy-worker.js not found, GPU features will not be available');
}

let pyodide;
let initialized = false;

function log(message) {
  postMessage({ namespace: 'app', method: 'log', message: message });
}

function displayImage(url) {
  postMessage({ namespace: 'app', method: 'displayImage', url });
}

function displayImageRaw(numpyArray, width, height) {
  // Extract the underlying buffer from NumPy array via Pyodide's buffer protocol
  // getBuffer returns an object with { data: TypedArray, ... }
  const bufferInfo = numpyArray.getBuffer('u8');
  
  // Get the TypedArray's underlying buffer, only the data portion
  // Calculate the correct size: width * height * 4 (RGBA)
  const expectedSize = width * height * 4;
  const buffer = bufferInfo.data.buffer.slice(bufferInfo.data.byteOffset, bufferInfo.data.byteOffset + expectedSize);
  
  console.log(`Worker: sending buffer of size ${buffer.byteLength}, expected ${expectedSize}`);
  
  // Transfer ArrayBuffer for zero-copy
  postMessage({ 
    namespace: 'app', 
    method: 'displayImageRaw', 
    buffer: buffer, 
    width, 
    height 
  }, [buffer]);
}

function stdout(line) {
  // remove escape seqeunce
  log(line.replace(/\x1b\[[0-9;]*[A-Za-z]/, ''));
}

async function loadPythonCode(name) {
  const f = await fetch(name);
  if (!f.ok) {
    throw new Error(f.statusText);
  }
  return f.text();
}

async function start(config) {
  log('Initializing wgpy worker-side javascript interface');
  let initWorkerResult = null;
  try {
    initWorkerResult = await wgpy.initWorker();
  } catch (e) {
    // if no backend is available, wgpy.initWorker throws an error.
    log(`initWorker failed: ${e.message}`);
  }

  log('Loading pyodide');
  pyodide = await loadPyodide({
    indexURL: `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`,
    stdout: stdout,
    stderr: stdout,
  });
  await pyodide.loadPackage('micropip');
  await pyodide.loadPackage('numpy');
  await pyodide.loadPackage('scipy');
  await pyodide.loadPackage("pillow"); // for visualizing the result
  if (initWorkerResult) {
    // load wgpy python package corresponding to the backend.
    // if wgpy is not initialized, wgpy (and cupy) is not available.
    await pyodide.loadPackage(`../../dist/wgpy_${initWorkerResult.backend}-1.0.0-py3-none-any.whl`);
  }

  log('Loading pyodide succeeded');
  const pythonCode = await loadPythonCode('code.py');

  self.pythonIO = {
    config,
    displayImage,
    displayImageRaw,
  };
  log('Running python code');
  await pyodide.runPythonAsync(pythonCode);
  await pyodide.runPythonAsync(`visualize(${config.grid})`);
  initialized = true;
}

addEventListener('message', (ev) => {
  if (ev.data.namespace !== 'app') {
    // message for library
    return;
  }

  switch (ev.data.method) {
    case 'start':
      start(ev.data.config).catch((reason) => log(`Worker error: ${reason}`));
      break;
    case 'render':
      if (initialized) {
        pyodide.runPythonAsync(`visualize(${ev.data.grid}, real_min=${ev.data.x_min}, real_max=${ev.data.x_max}, imag_min=${ev.data.y_min}, imag_max=${ev.data.y_max})`);
      }
      break;
  }
});
