// Load Pyodide from CDN
const PYODIDE_VERSION = 'v0.26.4';
importScripts(`https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/pyodide.js`);
importScripts('../../dist/wgpy-worker.js');

let pyodide;

function log(message) {
  postMessage({ namespace: 'app', method: 'log', message: message });
}

async function loadPythonCode() {
  const f = await fetch('code.py');
  if (!f.ok) {
    throw new Error(f.statusText);
  }
  return f.text();
}

async function start(backend, data) {
  log('Initializing wgpy worker-side javascript interface');
  const initWorkerResult = await wgpy.initWorker();

  log('Loading pyodide');
  pyodide = await loadPyodide({
    indexURL: `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`,
  });
  await pyodide.loadPackage('micropip');
  await pyodide.loadPackage('numpy');
  const wheelUrl = new URL(`../../dist/wgpy_${initWorkerResult.backend}-1.0.0-py3-none-any.whl`, self.location.href).href;
  await pyodide.loadPackage(wheelUrl);

  log('Loading pyodide succeeded');
  const pythonCode = await loadPythonCode();

  self.pythonIO = {
    getInputData: () => JSON.stringify(data),
    setOutputData: (data) =>
      postMessage({
        namespace: 'app',
        method: 'output',
        data: JSON.parse(data),
      }),
  };
  log('Running python code');
  await pyodide.runPythonAsync(pythonCode);
}

addEventListener('message', (ev) => {
  if (ev.data.namespace !== 'app') {
    // message for library
    return;
  }

  switch (ev.data.method) {
    case 'start':
      start(ev.data.backend, ev.data.data).catch((reason) => log(`Worker error: ${reason}`));
      break;
  }
});
