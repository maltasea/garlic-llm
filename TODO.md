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
