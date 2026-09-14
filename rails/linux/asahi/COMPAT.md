# Cross-compat (no fetch)

| Host | Vulkan stack | Compute for trit.comp |
|---|---|---|
| Fedora Asahi Remix | Honeykrisp (Mesa, AGX) | possible; Home does not submit |
| Darwin | KosmicKrisp / MoltenVK (Vulkan-on-Metal) | possible; Home does not submit |
| x86 Linux | RADV/ANV/NVK | possible; Home does not submit |
| this VM | none | CPU exec_shader |

Do not `dnf upgrade` or curl firmware from Home. Operator box owns Mesa.
GPU step not warranted for T0/T1 notes.
