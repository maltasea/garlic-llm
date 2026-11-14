# TODO

## Faster decoding (later)
Details: `tasks/open/task_faster-decoding_RKR8IVd10101.md`.

Now (2026-11-14, PS4 Pro, server-only app): 34–35 tok/s at T=0.7, 39–40 at T=0. Rough ceiling ~75 tok/s (estimate).

1. Measure where the time goes: log `spec_stats` (GPU vs wall) per server request; cut GPU idle time first.
2. Try MTP theta values via `/data/garlic_spec.txt` ("2 <theta>"), one app restart each.
3. 3-bit gate/up weights (q3s): ~9 % fewer bytes per pass; needs the bf16 model and a converter run.
4. Kernels for the Pro's 36 CUs, only if step 1 shows the GPU is the limit.
