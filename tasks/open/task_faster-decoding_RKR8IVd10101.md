# Task: Faster decoding

  id: RKR8IVd10101  
  status: open  

## Goal
More tokens/s from the API server on the PS4 Pro. Measured 2026-11-14 (server-only app, 400-token reply, streamed):
34–35 tok/s at T=0.7, 39–40 at T=0 (MTP head on, K=2, Pro mode). Rough ceiling: 4.6 GB of weights per verify pass at
the Pro's ~218 GB/s ≈ 47 passes/s, × ~1.6 tokens kept per pass ≈ 75 tok/s (estimate, not measured).

Already done: frame throttle while answering (26 → 29), Pro mode via param.sfo ATTRIBUTE 0x00800000 (29 → 33).
Tried: K=3 (`/data/garlic_spec.txt` "3 0.5"): +3 % at T=0.7, −11 % at T=0; K=2 stays.

## Plan
1. **Measure where the time goes.** Log `spec_stats` (GPU verify / draft time vs wall per cycle) after each server
   request, as the chat's `finish_turn` did. If the GPU idles between passes (CPU sampling, embedding rows read from
   the weights file, submit/wait), cut that overhead first.
2. **Theta.** Try a few values of the MTP draft threshold through `/data/garlic_spec.txt` ("2 <theta>"), one app
   restart each; bake the winner into `engine_spec.inc` if it beats 0.5.
3. **3-bit gate/up weights (q3s, fmt 8).** The kernels exist (`k_swar3`). ~9 % fewer bytes per pass, some quality
   cost. Needs the bf16 Qwen/Qwen3.5-9B tensors and a converter run in `pc/`.
4. **Kernels for 36 CUs.** Only if step 1 shows the GPU itself is the limit.

## Gate
`/usr/bin/python3` speed script (streamed, 400 tokens, T=0 and T=0.7, three runs after a warm-up) beats the numbers
above; the contract tests in `wrk/openflow/.../project/tests` still pass.
