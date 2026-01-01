import { getNNWebGPUContext } from './webgpuContext';

export class WebGPUTexture {
  gpuTexture: GPUTexture;
  width: number;
  height: number;
  format: GPUTextureFormat;

  constructor(width: number, height: number, format: GPUTextureFormat = 'rgba8unorm') {
    const ctx = getNNWebGPUContext();
    this.width = width;
    this.height = height;
    this.format = format;

    this.gpuTexture = ctx.device.createTexture({
      size: { width, height, depthOrArrayLayers: 1 },
      format,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  dispose() {
    this.gpuTexture.destroy();
    (this as { gpuTexture: GPUTexture | null }).gpuTexture = null;
  }
}
