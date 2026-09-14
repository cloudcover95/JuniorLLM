# Memory barriers vs Junior wire

trit.comp writes an SSBO. Home never `vkQueueSubmit`s, so a barrier graph cannot beat Python/C Winsor.

If an Asahi box later submits one dispatch:
- after compute: `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` → `VK_PIPELINE_STAGE_HOST_BIT`
- access: `SHADER_WRITE` → `HOST_READ`
- or `vkMapMemory` with `HOST_COHERENT`

No intra-workgroup shared memory in trit.comp, so `memoryBarrierBuffer()` inside the shader is not a win.
Do not add a second packer that waits on GPU just to clip 256 floats.
