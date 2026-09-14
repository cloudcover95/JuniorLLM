# Barriers for trit.comp (when someone actually submits)

trit.comp only **writes** `t[i]`. γ is a push constant computed on CPU. No shared memory, no workgroup barrier inside the shader.

## Host → SSBO in
If the input buffer is non-coherent: `vkFlushMappedMemoryRanges` then

```
srcStage = HOST
srcAccess = HOST_WRITE
dstStage = COMPUTE_SHADER
dstAccess = SHADER_READ
```

HOST_WRITE in srcAccess is the host→device domain op.

## SSBO out → host
After `vkCmdDispatch`:

```
srcStage = COMPUTE_SHADER
srcAccess = SHADER_WRITE
dstStage = HOST
dstAccess = HOST_READ
```

Then wait a fence / timeline. A pipeline barrier alone does not wake the CPU. Non-coherent memory also needs `vkInvalidateMappedMemoryRanges`.

## Not needed here
Workgroup `barrier()` / `memoryBarrierShared` — we have no `shared` vars.
Draw-indirect barriers — we do not chain dispatch sizes.
Queue-family release — single compute queue.

Home still does not submit. This file is the contract for an Asahi operator box.
