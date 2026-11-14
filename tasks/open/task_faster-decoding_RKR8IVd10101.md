# Task: Faster decoding

  id: RKR8IVd10101  
  status: open  

## Goal
More tokens/s from the API server on the PS4 Pro. Measured 2026-11-14 (server-only app, 400-token reply, streamed):
37–38 tok/s at T=0 and T=0.7 (MTP head on, K=2, Pro mode). Rough ceiling: 4.6 GB of weights per verify pass at
the Pro's ~218 GB/s ≈ 47 passes/s, × ~1.6 tokens kept per pass ≈ 75 tok/s (estimate, not measured).

Already done: frame throttle while answering (26 → 29), Pro mode via param.sfo ATTRIBUTE 0x00800000 (29 → 33), no
frames while answering, one GPU submission less per cycle, spinning instead of sleeping on the GPU (→ 37–38).
Measured (the log's spec line per request): GPU 41.6 ms per verify + 4.2 ms per draft, 1.75–1.8 tokens per cycle, GPU
busy 99 % of the decode.
Tried, no gain: K=3, theta 0.3 / 0.7 / 0.9, sleeping 100 or 1000 µs between GPU checks (`/data/garlic_tune.txt`
"<K> <theta> <wait_us>", read before every request: no restart needed).

## Plan
1. **3-bit gate/up weights (q3s, fmt 8).** The kernels exist (`k_swar3`). ~9 % fewer bytes per pass, some quality
   cost. Needs the bf16 Qwen/Qwen3.5-9B tensors and a converter run in `pc/`.
2. **Kernels for 36 CUs.** A verify pass moves ~110 GB/s of the ~218 the Pro has: the product kernels are the limit.

## Gate
`/usr/bin/python3` speed script (streamed, 400 tokens, T=0 and T=0.7, three runs after a warm-up) beats the numbers
above; the contract tests in `wrk/openflow/.../project/tests` still pass.
