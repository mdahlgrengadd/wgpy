from typing import Optional
from wgpy_backends.webgpu.platform import get_platform


class WebGPUTexture:
    """Manages a WebGPU texture for rendering to canvas"""

    next_id = 1

    def __init__(self, width: int, height: int, format: str = "rgba8unorm"):
        self.width = width
        self.height = height
        self.format = format
        self.texture_id = WebGPUTexture.next_id
        WebGPUTexture.next_id += 1

        get_platform().createTexture(self.texture_id, width, height, format)

    def copy_from_buffer(self, buffer_id: int):
        """Copy RGBA data from buffer to this texture"""
        get_platform().copyBufferToTexture(buffer_id, self.texture_id, self.width, self.height)

    def present(self):
        """Present this texture to the canvas"""
        get_platform().presentTexture(self.texture_id)

    def __del__(self):
        get_platform().disposeTexture(self.texture_id)
