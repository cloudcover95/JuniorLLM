---
local: true
date: 2026-11-07
wave: "2026-11-07 02:05 (bench 02:55, ran ~03:00–03:10 MT)"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Deck sheet: package ports (JuniorOSai) — `ports/terraform_lean.py` via `scripts/terraform_lean_prod.py code|llm` (blender off)"
verdict: not-viable
tags: [osai, vault, findings, ternary, terraform-lean, juniorosai, ports, trit-energy, absmean, lut, numpy, pure-python, bench, not-viable]
---

# terraform_lean trit_energy: ternary count path is 2.27x faster on the math but only ~1.07x end-to-end

## Question

The last eight vault notes all stayed on the `juniorctl ask` pair-code-vs-int8 Dev-emulation line (latest [[2026-11-06_1455_ask-b16-b32-b64-emulated-random-mixed-rw-thermal-pair-vs-int8]]). This run rotates to a surface the bench lane had never measured: the JuniorLLM Terraform lean port. `ports/terraform_lean.py` `run(note, surface)` computes `trit_energy` by absmean-ternarizing the first 32 chars of the note as a Python float list, then writes one JSON file per surface. Does a ternary-native path (count the three |trit| levels directly, no float list) make a real efficiency difference for `scripts/terraform_lean_prod.py code|llm`, and does numpy help at this size?

## Method

`python3 scripts/terraform_lean_prod.py code` and `... llm` were run locally in JuniorLLM first (both `ok: true`, `trit_energy: 0.8`, `bind: 127.0.0.1`, `launch/bpy/model_pull: false`; blender not run). Bench compares four implementations of the same `trit_energy` math:

- **ref**: exact copy of `run()`'s float-list path.
- **lutpy**: pure-Python ternary count. Each char maps to |2x| in {0,1,2}; gamma = (c1 + 2·c2)/(2n); zero-trit count is c0 plus c1 or c2 only when round(0.5/γ) or round(1/γ) is 0. No floats per element.
- **npf**: numpy float32 vector path (`frombuffer`, `% 5`, `rint`).
- **lut**: numpy 256-entry int8 byte→level LUT plus `bincount`.

10,000 notes (the literal `"JuniorOSai"` plus 9,999 random printable-ASCII strings of length 1–160, seed `202611070255`). All three alternatives were checked against **ref** on every note. 7 timed repeats × 20 passes × 10,000 calls per implementation. Then full `run("JuniorOSai", surface)` (its one write to `~/.juniorhome/os/terraform_<surface>.json`) for code and llm, 7 repeats × 2,000 calls each. numpy 2.5.4, 8 vCPU shared box, python3 and numpy only, no network, nothing bound. Inputs (`ports/terraform_lean.py`, previous note) byte-identical to JuniorLLM tip 343661f. Command (script `bench.py`, scratch `/workspace/bench0255_1107/`):

```
python3 bench.py 7 202611070255
```

## Results

Error: **0 mismatches** vs ref out of 10,000 notes for lutpy, npf and lut (exact `trit_energy`).

| path | median µs/call | min–max | vs ref |
|---|---|---|---|
| ref (float list, current) | 5.172 | 4.843–5.443 | 1.00x |
| lutpy (pure-Python ternary count) | 2.274 | 2.232–2.350 | **2.27x** |
| npf (numpy float32) | 8.467 | 8.252–10.610 | 0.61x |
| lut (numpy int8 LUT + bincount) | 6.126 | 6.001–6.539 | 0.84x |

| full run() | median µs/call | min–max | ref math share |
|---|---|---|---|
| run(code) | 40.62 | 39.78–50.98 | 12.7% |
| run(llm) | 43.38 | 41.12–48.76 | 11.9% |

Bytes for the 32-char ternarized window: float64 list 256 B, float32 128 B, int8 trits 32 B, 2-bit packed 8 B. The written body is 213 B of JSON either way, because `trit_energy` is a single scalar.

## Verdict

**Not viable.** The ternary count path is 2.27x faster on the isolated math and exact on all 10,000 notes. But the math is only about 12–13% of `run()`, and the single JSON write and `mkdir` take the rest. Swapping it in saves about 2.9 µs of about 41 µs, roughly **1.07x end-to-end**, which is far below the 1.5x bar. Memory is irrelevant here: the working set is at most 32 values and the output is one scalar. numpy makes things *slower* at this size (0.61x–0.84x) because per-call overhead dominates. The ternary-native count is still a reasonable zero-risk cleanup if the port is touched for other reasons, but it is not worth a change on efficiency grounds. Bounded to this one-scalar surface on a shared box.

Unpublished carry-over: the 2026-11-06 18:55 bench finished its run but never pushed a note (routine failed). Its local output (`/workspace/bench1855_1106/full.txt`) answered the 14:55 next question: under emulated random+thermal it gave K=1/2/4/8 at about 2.11x/2.79x/4.58x/4.86x. Asymmetric R/W at 25/50/75 MB/s gave K=8 at about 4.88x–4.92x and K=1 at about 3.39x–4.23x. All runs were B=16/32/64, N=2M, with exact top-10. This is recorded here for the ledger only and is not this run's finding.

## Pair files this wave

- No pair file expected or visible for this wave: Chief has reported both pairs no gap for many idle waves, and the roadmap blobs are still JuniorHome ROADMAP 4efb666 and JuniorLLM BETA_TO_OS bac6923. JuniorLLM tip before this push was [343661f](https://github.com/cloudcover95/JuniorLLM/commit/343661f) (`rails/onerepo/one_repo_check.py`, unattributed, no roadmap item). The pair A base files are still Winsor [JuniorQuant 946f352](https://github.com/cloudcover95/JuniorQuant/commit/946f352) and Trit Pack [JuniorLLM 184a480](https://github.com/cloudcover95/JuniorLLM/commit/184a480).

## Next question

Terraform lean is not compute-bound, so ternary cannot matter there. Next, rotate to a surface that is not `ask`: the bitnetd/i2sd in-process path. Specifically, compare I2_S 2-bit block decode into the 127.0.0.1:8765 handler buffer against int8 at a realistic per-request token batch, or measure Winsor-clipped activation error vs speed at the bitnetd boundary, if neither has been covered.

close: 57517ba3f166975b (sha256 of previous note text || this note text above this line, first 16 hex; prev 2026-11-06_1455)
