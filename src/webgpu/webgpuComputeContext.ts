import { nonNull } from '../util';
import { getNNWebGPUContext, initializeNNWebGPUContext } from './webgpuContext';
import {
  WebGPUTensorBuffer,
} from './webgpuTensorBuffer';
import { WebGPUTexture } from './webgpuTexture';

export type WorkGroupDim = 'x' | 'y' | 'z';

export interface GPUKernelRunDescriptor {
  name: string;
  tensors: number[];
  workGroups: { [key in WorkGroupDim]: number };
}

export interface ComputeContextGPUMessageCreateBuffer {
  method: 'gpu.createBuffer';
  id: number;
  byteLength: number;
}

export interface ComputeContextGPUMessageCreateMetaBuffer {
  method: 'gpu.createMetaBuffer';
  id: number;
  byteLength: number;
  data: Uint8Array;
}

export interface ComputeContextGPUMessageDisposeBuffer {
  method: 'gpu.disposeBuffer';
  id: number;
}

export interface ComputeContextGPUMessageSetData {
  method: 'gpu.setData';
  id: number;
  data: Uint8Array;
}

export interface ComputeContextGPUMessageGetData {
  method: 'gpu.getData';
  id: number;
  data: SharedArrayBuffer; // TypedArray of SharedArrayBuffer
  notify: SharedArrayBuffer; // Int32Array(1) of SharedArrayBuffer
}

export interface ComputeContextGPUMessageAddKernel {
  method: 'gpu.addKernel';
  name: string;
  descriptor: { source: string; bindingTypes: GPUBufferBindingType[] };
}

export interface ComputeContextGPUMessageRunKernel {
  method: 'gpu.runKernel';
  descriptor: GPUKernelRunDescriptor;
}

export interface ComputeContextGPUMessageCreateTexture {
  method: 'gpu.createTexture';
  id: number;
  width: number;
  height: number;
  format: GPUTextureFormat;
}

export interface ComputeContextGPUMessageDisposeTexture {
  method: 'gpu.disposeTexture';
  id: number;
}

export interface ComputeContextGPUMessageCopyBufferToTexture {
  method: 'gpu.copyBufferToTexture';
  bufferId: number;
  textureId: number;
  width: number;
  height: number;
}

export interface ComputeContextGPUMessagePresentTexture {
  method: 'gpu.presentTexture';
  textureId: number;
}

export type ComputeContextGPUMessage =
  | ComputeContextGPUMessageAddKernel
  | ComputeContextGPUMessageCreateBuffer
  | ComputeContextGPUMessageCreateMetaBuffer
  | ComputeContextGPUMessageDisposeBuffer
  | ComputeContextGPUMessageGetData
  | ComputeContextGPUMessageRunKernel
  | ComputeContextGPUMessageSetData
  | ComputeContextGPUMessageCreateTexture
  | ComputeContextGPUMessageDisposeTexture
  | ComputeContextGPUMessageCopyBufferToTexture
  | ComputeContextGPUMessagePresentTexture;

export class ComputeContextGPU {
  tensorBuffers: Map<number, WebGPUTensorBuffer> = new Map();
  textures: Map<number, WebGPUTexture> = new Map();
  canvasContext: GPUCanvasContext | null = null;
  blitPipeline: GPURenderPipeline | null = null;
  blitBindGroupLayout: GPUBindGroupLayout | null = null;

  async init() {
    await initializeNNWebGPUContext();
  }

  createBuffer(
    id: number,
    byteLength: number,
  ) {
    const tensorBuffer = new WebGPUTensorBuffer({
      byteLength,
    }, false);
    this.tensorBuffers.set(id, tensorBuffer);
  }

  createMetaBuffer(
    id: number,
    byteLength: number,
    data: Uint8Array,
  ) {
    const tensorBuffer = new WebGPUTensorBuffer({
      byteLength,
    }, true);
    tensorBuffer.setMetaBufferContent(data);
    this.tensorBuffers.set(id, tensorBuffer);
  }

  disposeBuffer(id: number) {
    const tb = this.tensorBuffers.get(id);
    if (tb) {
      tb.dispose();
      this.tensorBuffers.delete(id);
    }
  }

  setData(id: number, data: Uint8Array): void {
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return;
    }
    tb.setDataRaw(data);
  }

  getData(id: number): Promise<Uint8Array> {
    const tb = this.tensorBuffers.get(id);
    if (!tb) {
      return Promise.reject();
    }
    return tb.getDataRaw() as Promise<Uint8Array>;
  }

  addKernel(
    name: string,
    descriptor: { source: string; bindingTypes: GPUBufferBindingType[] }
  ) {
    const ctx = getNNWebGPUContext();
    ctx.createPipeline(name, descriptor.source, descriptor.bindingTypes);
  }

  runKernel(descriptor: GPUKernelRunDescriptor) {
    const ctx = getNNWebGPUContext();
    const tensor = descriptor.tensors.map((id) =>
      nonNull(this.tensorBuffers.get(id))
    );
    ctx.runKernel({
      pipelineName: descriptor.name,
      tensorBuffers: tensor,
      workGroups: descriptor.workGroups,
    });
  }

  createTexture(id: number, width: number, height: number, format: GPUTextureFormat) {
    const texture = new WebGPUTexture(width, height, format);
    this.textures.set(id, texture);
  }

  disposeTexture(id: number) {
    const texture = this.textures.get(id);
    if (texture) {
      texture.dispose();
      this.textures.delete(id);
    }
  }

  copyBufferToTexture(bufferId: number, textureId: number, width: number, height: number) {
    const buffer = this.tensorBuffers.get(bufferId);
    const texture = this.textures.get(textureId);
    if (!buffer || !texture) {
      throw new Error(`Buffer ${bufferId} or texture ${textureId} not found`);
    }

    const ctx = getNNWebGPUContext();
    const commandEncoder = ctx.device.createCommandEncoder();

    commandEncoder.copyBufferToTexture(
      {
        buffer: buffer.gpuBuffer,
        offset: 0,
        bytesPerRow: width * 4, // 4 bytes per RGBA pixel
        rowsPerImage: height,
      },
      {
        texture: texture.gpuTexture,
        mipLevel: 0,
        origin: { x: 0, y: 0, z: 0 },
      },
      {
        width,
        height,
        depthOrArrayLayers: 1,
      }
    );

    ctx.device.queue.submit([commandEncoder.finish()]);
  }

  presentTexture(textureId: number) {
    const texture = this.textures.get(textureId);
    if (!texture) {
      throw new Error(`Texture ${textureId} not found`);
    }
    if (!this.canvasContext) {
      throw new Error('Canvas context not initialized');
    }
    if (!this.blitPipeline || !this.blitBindGroupLayout) {
      throw new Error('Blit pipeline not initialized');
    }

    const ctx = getNNWebGPUContext();
    const commandEncoder = ctx.device.createCommandEncoder();

    // Get current canvas texture
    const canvasTexture = this.canvasContext.getCurrentTexture();

    // Create sampler for texture sampling
    const sampler = ctx.device.createSampler({
      magFilter: 'nearest',
      minFilter: 'nearest',
    });

    // Create bind group for the source texture
    const bindGroup = ctx.device.createBindGroup({
      layout: this.blitBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: texture.gpuTexture.createView(),
        },
        {
          binding: 1,
          resource: sampler,
        },
      ],
    });

    // Render pass to blit texture to canvas
    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: canvasTexture.createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });

    renderPass.setPipeline(this.blitPipeline);
    renderPass.setBindGroup(0, bindGroup);
    renderPass.draw(3, 1, 0, 0); // Draw full-screen triangle
    renderPass.end();

    ctx.device.queue.submit([commandEncoder.finish()]);
  }

  async initCanvasContext(canvas: HTMLCanvasElement) {
    const ctx = getNNWebGPUContext();
    const preferredFormat = navigator.gpu.getPreferredCanvasFormat();

    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error('Failed to get WebGPU canvas context');
    }

    context.configure({
      device: ctx.device,
      format: preferredFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
      alphaMode: 'opaque',
    });

    this.canvasContext = context;

    // Create render pipeline for blitting textures to canvas
    const blitShaderModule = ctx.device.createShaderModule({
      code: `
        @vertex
        fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
          // Full-screen triangle
          let x = f32((vertexIndex << 1u) & 2u);
          let y = f32(vertexIndex & 2u);
          return vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
        }

        @group(0) @binding(0) var srcTexture: texture_2d<f32>;
        @group(0) @binding(1) var srcSampler: sampler;

        @fragment
        fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
          let texSize = textureDimensions(srcTexture);
          let texCoord = pos.xy / vec2f(f32(texSize.x), f32(texSize.y));
          return textureSample(srcTexture, srcSampler, texCoord);
        }
      `,
    });

    this.blitBindGroupLayout = ctx.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: {},
        },
      ],
    });

    this.blitPipeline = ctx.device.createRenderPipeline({
      layout: ctx.device.createPipelineLayout({
        bindGroupLayouts: [this.blitBindGroupLayout],
      }),
      vertex: {
        module: blitShaderModule,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: blitShaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: preferredFormat }],
      },
      primitive: {
        topology: 'triangle-list',
      },
    });
  }

  mdata: SharedArrayBuffer | null = null;
  mnotify: Int32Array | null = null;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  handleMessage(message: ComputeContextGPUMessage, worker: Worker) {
    switch (message.method) {
      case 'gpu.addKernel':
        this.addKernel(message.name, message.descriptor);
        break;
      case 'gpu.createBuffer':
        this.createBuffer(
          message.id,
          message.byteLength,
        );
        break;
      case 'gpu.createMetaBuffer':
        this.createMetaBuffer(message.id, message.byteLength, message.data);
        break;
      case 'gpu.disposeBuffer':
        this.disposeBuffer(message.id);
        break;
      case 'gpu.getData':
        if (message.data) {
          this.mdata = message.data;
        }
        if (message.notify) {
          this.mnotify = new Int32Array(message.notify);
        }
        this.getData(message.id)
          .then((data) => {
            (new Uint8Array(this.mdata!)).set(data);
            this.mnotify![0] = 1;
            Atomics.notify(this.mnotify!, 0);
          })
          .catch((reason) => {
            console.error(reason);
          });
        break;
      case 'gpu.runKernel':
        this.runKernel(message.descriptor);
        break;
      case 'gpu.setData':
        this.setData(message.id, message.data);
        break;
      case 'gpu.createTexture':
        this.createTexture(message.id, message.width, message.height, message.format);
        break;
      case 'gpu.disposeTexture':
        this.disposeTexture(message.id);
        break;
      case 'gpu.copyBufferToTexture':
        this.copyBufferToTexture(message.bufferId, message.textureId, message.width, message.height);
        break;
      case 'gpu.presentTexture':
        this.presentTexture(message.textureId);
        break;
    }
  }
}
