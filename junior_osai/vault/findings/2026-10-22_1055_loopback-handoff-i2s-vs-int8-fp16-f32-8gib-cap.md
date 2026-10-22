---
local: true
date: 2026-10-22
wave: "2026-10-22 10:05"
roadmap_item: "JuniorLLM docs/BETA_TO_OS.md Live beta step 6: No deletes. No >8GB pull. bitnetd on 127.0.0.1:8765 only (loopback weight hand-off cost and cap use)"
chief_assignment: "Live beta step 6 assigned to pair B at 10:05 MT (Quant Chief, last_tips.json); pair A JuniorOS overlay remaining gaps"
verdict: viable
tags: [ternary, osai, findings, live-beta, step6, pull-cap, loopback, i2s, sha256, numpy, viable]
---

# Loopback hand-off of the full 2B4T weight set: I2_S moves in 0.37 s using 6% of the 8 GiB cap, vs 2.69 s / 49% for fp16 and 5.43 s / 97% for f32

Previous note: [[2026-10-22_0555_memorymax-2g-fit-i2s-vs-fp16-f32]]

## Question

Step 6 allows bitnetd to listen only on `127.0.0.1:8765` and forbids any pull over 8 GB. When an operator hands bitnetd a whole BitNet b1.58 2B4T weight set over loopback (no network, no download), how long does the move take end to end (send over a 127.0.0.1 TCP socket, receiver streams it through sha256 in a fixed 1 MiB buffer, digest checked), and how much of the cap does it use, for packed I2_S (2-bit codes + one f32 gamma per matrix) vs int8, fp16 and f32?

## Method

- Layer shape is 2B4T: q 2560x2560, k and v 640x2560, o 2560x2560, gate and up 6912x2560, down 2560x6912; 30 layers.
- One layer is built per format from Student-t(4) weights at scale 0.02, seed 1055: f32 and fp16 as-is; int8 with one absmax scale per matrix; I2_S with the landed lean path (abs-mean gamma, threshold gamma/2, 4 trits per byte, MSB first). The I2_S pack round-trip is asserted exact. That layer is sent 30 times per transfer, so every byte on the wire is real.
- Receiver is a forked child on an ephemeral port bound to 127.0.0.1 (8765 is left to bitnetd; the bind address is asserted). It reads with `recv_into` into one 1 MiB buffer, updates sha256 and returns the digest; the sender compares it to a precomputed reference.
- 1 warm-up plus 7 timed transfers per format; median and min-max. Cap is the in-repo `CAP_BYTES = 8 * 1024**3` from `ports/ondisk.py`.
- Host: box, 8 cores, numpy 2.5.4, about 2 GB free RAM. numpy and stdlib only, offline, nothing bound except 127.0.0.1, no downloads.

Command: `python3 bench.py` (script below; writes results.json; about 100 s).

## Results

All four digests matched on every transfer (I2_S f685b690cbb18fd4, int8 51ae03fb118ed7bb, fp16 945e80b1b4337247, f32 a241902e3c65e47f).

| format | bytes per layer | full model | share of 8 GiB cap | full models per cap | hand-off ms, median (min-max) | GB/s |
|---|---|---|---|---|---|---|
| I2_S + gamma | 17,367,068 | 521.0 MB | 6.1% | 16 | 370.0 (336.4-416.2) | 1.41 |
| int8 + scale | 69,468,188 | 2.084 GB | 24.3% | 4 | 1406.5 (1281.2-1465.7) | 1.48 |
| fp16 | 138,936,320 | 4.168 GB | 48.5% | 2 | 2689.6 (2599.5-2855.6) | 1.55 |
| f32 | 277,872,640 | 8.336 GB | 97.0% | 1 | 5425.6 (5243.8-5583.2) | 1.54 |

| I2_S vs | bytes | hand-off time |
|---|---|---|
| int8 | 4.00x smaller | 3.80x faster |
| fp16 | 8.00x smaller | 7.27x faster |
| f32 | 16.00x smaller | 14.66x faster |

## Verdict

**Viable.** On the loopback hand-off step 6 allows, I2_S moves and verifies the whole 2B4T weight set in 0.37 s, 7.3x faster than fp16 and 3.8x faster than int8, using 6% of the cap, with a bit-exact digest and an exact pack round-trip. Since BitNet b1.58 is trained ternary, the I2_S pack is the model itself, so there is no added quantization error. The gain is byte-bound: every format runs at about 1.4-1.55 GB/s through sha256, so time follows size, and I2_S pays a small per-byte overhead for its many short sends.

One finding for pair B: the f32 copy of the same model is 8,336,179,200 bytes. That passes the code's 8 GiB cap (8,589,934,592) but is over the doc's "8GB" if read as 8e9. That is exactly the gap Handshake's advisory row U left undecided in [966ce39](https://github.com/cloudcover95/JuniorLLM/commit/966ce39), and a real 2B-class model already sits inside it. I2_S (521 MB) is clear of the cap under either reading.

## This wave's pair files

- Pair B (Live beta step 6): Handshake, JuniorLLM [966ce39](https://github.com/cloudcover95/JuniorLLM/commit/966ce39) `ports/live_beta_step6_pull_cap_policy_port_map.py` (cap carriers, port map, loopback); Side-Commit, web3node [75fc5a4](https://github.com/cloudcover95/web3node/commit/75fc5a4) `live_beta_step6_ledger_no_delete_receipt.py` (append-only receipt audit).
- Pair A (JuniorOS overlay gaps): no Trit Pack or Winsor push seen by 10:55.
- This note touches neither path.

## Next question

The 05:55 note left an open question: what is the smallest MemoryMax that still holds 30 I2_S layers plus a 2048-token KV cache, and does an int8 KV cache give at least 1.5x more context per cap than fp16 at acceptable attention-output error? The 2026-10-18 KV note measured T2 KV against fp16/int8 at a fixed budget, so the new part is the cap sweep with the 558 MB weight floor.

## Script

```python
#!/usr/bin/env python3
"""Live beta step 6, loopback side: bitnetd may only listen on 127.0.0.1 and no pull may exceed
8 GiB. When an operator hands bitnetd a full BitNet-2B4T weight set over loopback (no network,
no download), how long does the move take end to end (send over a 127.0.0.1 TCP socket, receiver
streams it through sha256 in a fixed 1 MiB buffer, digest checked), and how much of the 8 GiB
cap does it use, for packed I2_S (2-bit codes + f32 gamma per matrix) vs int8, fp16 and f32?
Layer = q 2560x2560, k/v 640x2560, o 2560x2560, gate/up 6912x2560, down 2560x6912; 30 layers.
One layer is materialized per format and sent 30 times (bytes on the wire are real). The
receiver is a forked child. Ephemeral port on 127.0.0.1 only (8765 is left to bitnetd).
numpy + stdlib only, offline."""
import hashlib, json, os, socket, struct, sys, time
import numpy as np

LAYER = [("q",2560,2560),("k",640,2560),("v",640,2560),("o",2560,2560),
         ("gate",6912,2560),("up",6912,2560),("down",2560,6912)]
NL, CAP, BUF, REPS = 30, 8 << 30, 1 << 20, 7
rng = np.random.default_rng(1055)

def ternarize(w):
    g = np.float32(np.abs(w).mean())
    t = np.zeros(w.shape, np.int8); t[w > g/2] = 1; t[w < -g/2] = -1
    return t, g

def pack(t):
    c = (t.reshape(-1) & 3).astype(np.uint8)      # 0->0, 1->1, -1->3
    c = c.reshape(-1, 4)
    return (c[:,0] << 6) | (c[:,1] << 4) | (c[:,2] << 2) | c[:,3]

def unpack(p, n):
    lut = np.array([0, 1, 0, -1], np.int8)
    c = np.stack([(p >> 6) & 3, (p >> 4) & 3, (p >> 2) & 3, p & 3], 1).reshape(-1)
    return lut[c][:n]

def layer_bytes(fmt):
    parts = []
    for name, r, c in LAYER:
        w = (rng.standard_t(4, (r, c)) * 0.02).astype(np.float32)
        if fmt == "f32": parts.append(w.tobytes())
        elif fmt == "fp16": parts.append(w.astype(np.float16).tobytes())
        elif fmt == "int8":
            s = np.float32(np.abs(w).max() / 127)
            parts.append(np.float32(s).tobytes() + np.clip(np.rint(w / s), -127, 127).astype(np.int8).tobytes())
        else:
            t, g = ternarize(w); p = pack(t)
            assert np.array_equal(unpack(p, t.size), t.reshape(-1))   # pack round-trip exact
            parts.append(np.float32(g).tobytes() + p.tobytes())
        del w
    return b"".join(parts)

def receiver(srv):
    conn, _ = srv.accept(); buf = bytearray(BUF); mv = memoryview(buf)
    while True:
        hdr = conn.recv(8, socket.MSG_WAITALL)
        if len(hdr) < 8: break
        total = struct.unpack("<Q", hdr)[0]; h = hashlib.sha256(); got = 0
        while got < total:
            n = conn.recv_into(mv, min(BUF, total - got))
            if n == 0: os._exit(1)
            h.update(mv[:n]); got += n
        conn.sendall(h.digest())
    os._exit(0)


results = {}
for fmt in ["i2s", "int8", "fp16", "f32"]:
    blob = layer_bytes(fmt); lb = len(blob); total = lb * NL
    ref = hashlib.sha256()
    for _ in range(NL): ref.update(blob)
    ref = ref.digest()
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    host, port = srv.getsockname(); assert host == "127.0.0.1"
    pid = os.fork()
    if pid == 0: receiver(srv)
    srv.close(); s = socket.create_connection((host, port)); mv = memoryview(blob)
    times, ok = [], True
    for rep in range(REPS + 1):
        t0 = time.perf_counter()
        s.sendall(struct.pack("<Q", total))
        for _ in range(NL): s.sendall(mv)
        d = s.recv(32, socket.MSG_WAITALL)
        dt = time.perf_counter() - t0
        ok &= (d == ref)
        if rep: times.append(dt)
    s.close(); os.waitpid(pid, 0)
    t = np.array(times) * 1000
    results[fmt] = dict(bytes_per_layer=lb, model_bytes=total, cap_fraction=total / CAP,
                        under_cap=total <= CAP, models_under_cap=CAP // total,
                        digest_ok=bool(ok), digest16=ref.hex()[:16],
                        ms_median=float(np.median(t)), ms_min=float(t.min()), ms_max=float(t.max()),
                        gb_per_s=total / np.median(t) / 1e6)
    print(fmt, json.dumps(results[fmt]), flush=True)
    del blob, mv
json.dump(results, open("results.json", "w"), indent=1)
b = results["i2s"]
for f in ["int8", "fp16", "f32"]:
    r = results[f]
    print(f"i2s vs {f}: bytes {r['model_bytes']/b['model_bytes']:.2f}x, time {r['ms_median']/b['ms_median']:.2f}x")
```

close: f02d07d869af67de (sha256(previous note text || this note text above this line)[:16])
