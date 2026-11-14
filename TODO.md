# TODO

## Faster decoding (later)
Details: `tasks/open/task_faster-decoding_RKR8IVd10101.md`.

Now (2026-11-14, PS4 Pro, server-only app): 37–38 tok/s at T=0 and T=0.7 (400-token reply, streamed). The engine is
GPU-bound: 41.6 ms per verify pass + 4.2 ms per draft, the GPU busy 99 % of the decode. Rough ceiling ~75 tok/s
(estimate).

1. 3-bit gate/up weights (q3s): ~9 % fewer bytes per pass; needs the bf16 model and a converter run.
2. Faster kernels for the Pro's 36 CUs: a verify pass reads 4.6 GB in 41.6 ms (~110 GB/s of ~218).

Tried, no gain: MTP theta 0.3–0.9, K=3, sleeping instead of spinning while the GPU runs (`/data/garlic_tune.txt`
"<K> <theta> <wait_us>", read before every request).

## Phi-4-mini (dropped, 2026-11-14)
`pc/convert_gguf.py` converts `~/models/Phi-4-mini-instruct-Q4_K_M.gguf` and the PS4 loads it (3569 MiB, 8K
context), but the GPU answers "!!!!": NaN in every value already after layer 0's attention-add. A numpy forward
on the same weights.bin (scratch `phi_ref.py`) answers right ("The capital of France is Paris."); the 15 prompt
tokens match; K max 20, V max 6.4 (no fp16 overflow). Kernels read without finding it: matvec_q4, rmsnorm,
embed_q4, silu, attn_part (gq=3), qk_rope_cache with ATTN_NO_QK_NORM. Next step if picked up: copy the last token's
layer-0 values to a debug buffer (the debug code was never committed) and compare with the reference: x in 0.1207,
x normed 2.943, qkv 16.24, after rope ?, attention 1.384, o 1.659 (max abs).
