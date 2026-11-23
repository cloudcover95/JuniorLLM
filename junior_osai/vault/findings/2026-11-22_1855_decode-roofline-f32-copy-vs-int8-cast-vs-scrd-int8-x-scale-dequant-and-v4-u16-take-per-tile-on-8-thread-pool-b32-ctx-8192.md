---
local: true
date: 2026-11-22
wave: "2026-11-22 18:55 (fired 18:43 MT, ran ~18:44–19:00 MT; no pair reports reached chat this wave, quant derivation lane paused)"
roadmap_item: "JuniorHome docs/ROADMAP.md i2sd userspace 127.0.0.1:8767 / JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl ask (decode)"
verdict: "inconclusive (diagnostic, no new kernel: scrD decode is NOT at the bandwidth roofline; K int8×scale dequant is 1.55/1.61x a full f32 copy and 2.80x a bare int8→f32 cast, V4 take+scale is 2.87/2.74x; decode is 75/79% of the tile; copy-speed decode would bound the tile at 1.70/1.74x)"
tags: [osai, vault, findings, ternary, i2sd, bitnetd, decode, attention, roofline, bandwidth, dequant, kv-cache, k8v4, b32, numpy, inconclusive]
---

# Decode roofline for the scratch grouped K8V4 tile at B = 32 / ctx 8192: the scrD decode is not bandwidth-bound. The K dequant costs 1.55/1.61x a plain f32 copy and 2.80x a bare int8→f32 cast; the broadcast scale multiply, not memory, is the cost. The V4 nibble path is 2.87/2.74x a copy and gather-bound

The 14:55 note ([[2026-11-22_1455_lut-gather-decode-f32-pair-lut-v4-take-and-256-entry-f32-k8-lut-vs-cast-dequant-in-scratch-grouped-k8v4-attention-tile-b32-ctx-8192]]) asked: is the scrD decode already at the memory-bandwidth roofline? Time per pass of the old-token tiles at B = 32 / ctx 8192 for a plain f32→f32 np.copyto of the same elements, a bare int8→f32 cast with no scale, and the scrD int8×scale out= dequant, all on the 8-thread pool. If the dequant is within ~1.2x of the f32 copy, numpy has no decode lever left and the lane moves back to the 2-bit weight side. Otherwise the gap is the headroom for a fused K8V4 dot product.

Answers:
- **Not at the roofline.** K dequant (scrD) is 15.4 / 13.5 ms pooled vs 9.9 / 8.4 ms for an f32 copy that moves ~1.6x more bytes (1.55/1.61x; serial 2.07/2.08x). That is above the 1.2x bar, so there is decode headroom.
- **The scale broadcast is the cost, not the int8 read.** A bare int8→f32 cast is 5.5 / 4.8 ms, 0.56–0.58x of the f32 copy. Adding the (L, 2, 1) g64 scale broadcast makes it 2.80x slower. An f32 source × the same scale costs the same as int8 × scale (15.1 / 13.6 ms), so numpy's broadcast multiply loop is the limit (48–55 GB/s effective vs 119–152 GB/s for copy and cast).
- **V4 is gather-bound.** The u16 LUT take alone is 16.4 / 13.6 ms (22–27 GB/s), and take + scale is 28.4 / 23.0 ms, 2.87/2.74x a copy. 14:55 already showed that an f32 pair-LUT take is slower, so numpy has no cheaper way to unpack nibbles.
- **Decode share:** K dequant + V decode are 75% / 79% of the decode+matmul tile (58.3 / 46.4 ms; no softmax here), in line with 10:55's ~70%. The two matmuls are only 14.8 / 12.0 ms.
- **Headroom:** if both decodes ran at f32-copy speed, the tile would be 1.70/1.74x faster. That is the most a fused K8V4 dot product (no f32 materialization) could buy on this tile, and it needs native code, not numpy. Within numpy, the only lever left is the one scrF already uses: cast K without scale and fold the g64 scales into q·K. Swapping kdeq for the bare cast bounds the tile at 1.20/1.23x, a gain scrF (05:55, folded K8 scales; step 1.60–1.62x vs pool) has already taken. So there is no new numpy lever.
- **Error:** dequant max abs vs float64 reference: K 1.2e-07, V 7.5e-09. Ternary weights and bytes are untouched (KV + weights 295.40 MiB at ctx 8192, unchanged from 10:55).
- **Verdict: inconclusive.** This is a diagnostic with no new speed or memory win to rate against the 1.5x bar. It answers the lane question: numpy decode tuning is exhausted (scrF fold already harvested the cast-vs-scale gap, and V4 is gather-bound), while ~1.7x of headroom remains for native code only.

## Method

- Standalone roof.py (below), python3 + numpy only, no network, OPENBLAS_NUM_THREADS = 1, 8-thread ThreadPoolExecutor on a shared 8-core box. Same shapes as the grouped tile: B = 32 × 5 KV heads = 160 head rows, 32 tasks of 5 heads, ctx 8192 with 7168 old tokens = 7 old tiles of TK 1024, HD 128, G = 4. K8 int8 with g64 f32 scales; V4 nibble pairs with g32 f32 scales; LUT16 u16 nibble-pair table, same as the 05:55 sk.py.
- Every op writes into per-thread preallocated scratch, the same way scrD does. Sources stream from 140 MiB (K8) / 70 MiB (V4) / 560 MiB (f32 reference) histories. GB/s counts bytes read + written per element: copy_f32 8, copy_i8 2, cast 5, kdeq 5.06, f32×scale 8.06, take 2.5, vdeq 8.6, matmuls 4 (nominal).
- 5 rounds × 3 reps pooled per op (round order alternates), plus 15 serial single-task reps. Two seeds (11221855, 11221455).
- Command: `cd /workspace/bench1855_1122 && OPENBLAS_NUM_THREADS=1 SEED=11221855 python3 roof.py run1.json > run1.txt && OPENBLAS_NUM_THREADS=1 SEED=11221455 python3 roof.py run2.json > run2.txt` (~1 min per seed).

## Results

Pooled wall per full pass over the old tiles (all 160 heads × 7 tiles), median (min–max) of 15, ms; serial = one 5-head task, median of 15, ms.

| op | run1 pooled ms | run2 pooled ms | GB/s run1 / run2 | serial ms run1 / run2 |
|---|---|---|---|---|
| copy_f32 | 9.9 (8.5–12.5) | 8.4 (7.5–10.3) | 119 / 140 | 1.30 / 1.09 |
| copy_i8 | 3.0 (2.5–3.6) | 2.5 (2.1–4.4) | 99 / 115 | 0.18 / 0.16 |
| cast_i8_f32 | 5.5 (5.2–6.0) | 4.8 (4.3–5.5) | 134 / 152 | 0.82 / 0.68 |
| kdeq_scrD | 15.4 (14.1–19.0) | 13.5 (11.9–16.6) | 48 / 55 | 2.70 / 2.26 |
| f32_x_scale | 15.1 (13.1–18.5) | 13.6 (12.0–15.0) | 79 / 87 | 2.56 / 2.11 |
| v4_take_u16 | 16.4 (14.2–21.1) | 13.6 (11.9–15.1) | 22 / 27 | 2.79 / 2.32 |
| v4_deq_scrD | 28.4 (25.1–31.9) | 23.0 (21.2–27.5) | 45 / 55 | 5.01 / 4.25 |
| qk_matmul | 9.9 (7.7–13.2) | 8.1 (7.4–9.8) | 59 / 72 | 1.81 / 1.54 |
| pv_matmul | 4.9 (4.3–5.7) | 3.9 (3.5–4.9) | 121 / 151 | 0.57 / 0.50 |
| tile_scrD | 58.3 (52.1–66.1) | 46.4 (43.3–53.5) | — | 10.42 / 8.71 |

Ratios (pooled medians unless prefixed ser_):

| ratio | run1 | run2 |
|---|---|---|
| kdeq_over_copy_f32 | 1.55 | 1.61 |
| kdeq_over_cast | 2.80 | 2.80 |
| cast_over_copy_f32 | 0.56 | 0.58 |
| kdeq_over_copy_i8 | 5.16 | 5.32 |
| vdeq_over_copy_f32 | 2.87 | 2.74 |
| decode_share_tile | 0.75 | 0.79 |
| ser_kdeq_over_copy_f32 | 2.07 | 2.08 |
| ser_vdeq_over_copy_f32 | 3.85 | 3.89 |
| ser_decode_share | 0.74 | 0.75 |

## Pair files this wave

No pair reports reached chat this wave (quant derivation lane paused, builders idle). Previous findings: [[2026-11-22_1455_lut-gather-decode-f32-pair-lut-v4-take-and-256-entry-f32-k8-lut-vs-cast-dequant-in-scratch-grouped-k8v4-attention-tile-b32-ctx-8192]].

## Inline script (roof.py)

    import os, sys, time, json, threading, numpy as np
    from concurrent.futures import ThreadPoolExecutor
    SEED=int(os.environ.get("SEED","11221855")); rng=np.random.default_rng(SEED)
    B,KVH,G,HD,N,TK,WIN,W,NT=32,5,4,128,8192,1024,1024,32,8
    BH=B*KVH; HPT=BH//W; NOLD=(N-WIN)//TK; H2=HD//2
    POOL=ThreadPoolExecutor(NT)
    # history stores (old tokens only; the A8 window tile is common to all variants)
    NO=N-WIN
    k8=rng.integers(-127,128,(BH,NO,HD),dtype=np.int8)
    ks=(rng.random((BH,NO,2),dtype=np.float32)*0.02+0.005).astype(np.float32)
    v4=rng.integers(0,256,(BH,NO,H2),dtype=np.uint8)            # packed nibble pairs, 64 B/token
    v4s=(rng.random((BH,NO,4),dtype=np.float32)*0.02+0.005).astype(np.float32)
    kf=k8.astype(np.float32)                                     # f32 copy of the same elements (reference source)
    lo=np.arange(256)&15; hi=np.arange(256)>>4
    LUT16=((np.where(hi>7,hi-16,hi).astype(np.int16)<<8)|(np.where(lo>7,lo-16,lo).astype(np.int16)&0xff)).astype(np.uint16)
    Q=(rng.standard_normal((BH,G,HD))/np.sqrt(HD)).astype(np.float32)
    tl=threading.local()
    def scr():
        s=getattr(tl,"s",None)
        if s is None:
            s=dict(K=np.empty((HPT,TK,HD),np.float32),V=np.empty((HPT,TK,HD),np.float32),U=np.empty((HPT,TK,H2),np.uint16),
                   C=np.empty((HPT,TK,HD),np.int8),S=np.empty((HPT,G,TK),np.float32),acc=np.zeros((HPT,G,HD),np.float32)); tl.s=s
        return s
    def op_copy_f32(s,h0,h1,a,b): np.copyto(s["K"],kf[h0:h1,a:b])
    def op_copy_i8(s,h0,h1,a,b): np.copyto(s["C"],k8[h0:h1,a:b])
    def op_cast(s,h0,h1,a,b): np.copyto(s["K"],k8[h0:h1,a:b],casting="unsafe")
    def op_kdeq(s,h0,h1,a,b): np.multiply(k8[h0:h1,a:b].reshape(HPT,TK,2,64),ks[h0:h1,a:b][...,None],out=s["K"].reshape(HPT,TK,2,64))
    def op_f32scale(s,h0,h1,a,b): np.multiply(kf[h0:h1,a:b].reshape(HPT,TK,2,64),ks[h0:h1,a:b][...,None],out=s["K"].reshape(HPT,TK,2,64))
    def op_vtake(s,h0,h1,a,b): np.take(LUT16,v4[h0:h1,a:b],out=s["U"])
    def op_vdeq(s,h0,h1,a,b):
        np.take(LUT16,v4[h0:h1,a:b],out=s["U"]); np.multiply(s["U"].view(np.int8).reshape(HPT,TK,4,32),v4s[h0:h1,a:b][...,None],out=s["V"].reshape(HPT,TK,4,32))
    def op_qk(s,h0,h1,a,b): np.matmul(Q[h0:h1],s["K"].transpose(0,2,1),out=s["S"])
    def op_pv(s,h0,h1,a,b): s["acc"]+=np.matmul(s["S"],s["V"])
    def op_tile(s,h0,h1,a,b):  # full scrD decode+matmul of an old tile (no softmax)
        op_kdeq(s,h0,h1,a,b); op_qk(s,h0,h1,a,b); op_vdeq(s,h0,h1,a,b); op_pv(s,h0,h1,a,b)
    OPS=dict(copy_f32=(op_copy_f32,8),copy_i8=(op_copy_i8,2),cast_i8_f32=(op_cast,5),kdeq_scrD=(op_kdeq,5.0625),f32_x_scale=(op_f32scale,8.0625),
             v4_take_u16=(op_vtake,2.5),v4_deq_scrD=(op_vdeq,4+0.5+0.125+4),qk_matmul=(op_qk,4),pv_matmul=(op_pv,4),tile_scrD=(op_tile,None))
    # bytes/element moved (read+write, per K/V element of HD): copy_f32 4+4; copy_i8 1+1; cast 1+4; kdeq 1+4+scales(8B/128el=.0625); v take: 0.5 read+2 write per elem
    def task(fn,h0):
        s=scr(); h1=h0+HPT
        for t in range(NOLD): fn(s,h0,h1,t*TK,(t+1)*TK)
    def pooled(fn):
        fs=[POOL.submit(task,fn,h0) for h0 in range(0,BH,HPT)]; [f.result() for f in fs]
    for nm,(fn,_) in OPS.items(): pooled(fn)   # warm, allocate per-thread scratch
    R,REP=5,3; res={}
    order=list(OPS)
    for r in range(R):
        for nm in (order if r%2==0 else order[::-1]):
            fn=OPS[nm][0]
            for _ in range(REP):
                t0=time.perf_counter(); pooled(fn); res.setdefault(nm,[]).append(time.perf_counter()-t0)
    ser={}
    for nm in order:
        fn=OPS[nm][0]; task(fn,0)
        for _ in range(15):
            t0=time.perf_counter(); task(fn,0); ser.setdefault(nm,[]).append(time.perf_counter()-t0)
    EL=BH*NO*HD
    out={}
    print(f"seed {SEED}  B={B} ctx={N} old tokens {NO} BH={BH} tasks={W}x{HPT} heads  {NT} threads  elements/pass {EL/1e6:.1f}M")
    print(f"{'op':14s} {'pooled med ms':>13s} {'spread':>15s} {'GB/s':>7s} {'serial task ms':>14s} {'spread':>13s}")
    for nm in order:
        p=np.array(res[nm])*1e3; q=np.array(ser[nm])*1e3; bpe=OPS[nm][1]
        gbs=(EL*bpe/1e9)/(np.median(p)/1e3) if bpe else float('nan')
        out[nm]=dict(pool_med=float(np.median(p)),pool_min=float(p.min()),pool_max=float(p.max()),ser_med=float(np.median(q)),ser_min=float(q.min()),ser_max=float(q.max()),gbs=gbs)
        print(f"{nm:14s} {np.median(p):13.1f} {p.min():6.1f}-{p.max():6.1f}  {gbs:7.1f} {np.median(q):14.2f} {q.min():6.2f}-{q.max():6.2f}")
    m=lambda k:out[k]["pool_med"]; sm=lambda k:out[k]["ser_med"]
    rat=dict(kdeq_over_copy_f32=m("kdeq_scrD")/m("copy_f32"),kdeq_over_cast=m("kdeq_scrD")/m("cast_i8_f32"),cast_over_copy_f32=m("cast_i8_f32")/m("copy_f32"),
     kdeq_over_copy_i8=m("kdeq_scrD")/m("copy_i8"),vdeq_over_copy_f32=m("v4_deq_scrD")/m("copy_f32"),
     decode_share_tile=(m("kdeq_scrD")+m("v4_deq_scrD"))/m("tile_scrD"),
     ser_kdeq_over_copy_f32=sm("kdeq_scrD")/sm("copy_f32"),ser_vdeq_over_copy_f32=sm("v4_deq_scrD")/sm("copy_f32"),ser_decode_share=(sm("kdeq_scrD")+sm("v4_deq_scrD"))/sm("tile_scrD"))
    for k,v in rat.items(): print(f"{k:24s} {v:.2f}")
    # correctness: dequant matches reference math
    s=scr(); op_kdeq(s,0,HPT,0,TK); ref=(k8[:HPT,:TK].astype(np.float64).reshape(HPT,TK,2,64)*ks[:HPT,:TK,:,None]).reshape(HPT,TK,HD)
    op_vdeq(s,0,HPT,0,TK); u=v4[:HPT,:TK].astype(np.int16); l4=u&15; h4=u>>4; l4=np.where(l4>7,l4-16,l4); h4=np.where(h4>7,h4-16,h4)
    vr=(np.stack([l4,h4],-1).reshape(HPT,TK,HD).astype(np.float64).reshape(HPT,TK,4,32)*v4s[:HPT,:TK,:,None]).reshape(HPT,TK,HD)
    err=dict(kdeq_maxabs=float(np.abs(s["K"]-ref).max()),vdeq_maxabs=float(np.abs(s["V"]-vr).max()))
    print("error", err)
    json.dump(dict(seed=SEED,ops=out,ratios=rat,err=err),open(sys.argv[1],"w"),indent=1)

## Next question

The numpy attention tile is done (decode headroom is native-only, ~1.7x). Move back to the 2-bit weight side: at B = 32 / ctx 2048, qkv/o/gate-up/down are ~50% of the step. Does a K-blocked u32-LUT weight GEMM with per-worker preallocated f32 unpack scratch (the trick that cleared 1.5x on attention at 05:55) speed up the 2-bit weight pass by ≥ 1.5x vs the current u32-LUT path, at unchanged error and the same 4.00x weight bytes vs int8?

close: 80851b16f0cf4b9d
