# TODO

## Faster decoding (later)
Details: `tasks/open/task_faster-decoding_RKR8IVd10101.md`.

Now (2026-11-14, PS4 Pro, server-only app): 34–35 tok/s at T=0.7, 39–40 at T=0. Rough ceiling ~75 tok/s (estimate).

1. Measure where the time goes: log `spec_stats` (GPU vs wall) per server request; cut GPU idle time first.
2. Try MTP theta values via `/data/garlic_spec.txt` ("2 <theta>"), one app restart each.
3. 3-bit gate/up weights (q3s): ~9 % fewer bytes per pass; needs the bf16 model and a converter run.
4. Kernels for the Pro's 36 CUs, only if step 1 shows the GPU is the limit.

## Longer context (later)
Now 16K tokens. Memory allows much more: in Pro mode 670 MiB of GPU memory are free (4354 of 5024 MiB used), and the
TurboQuant 4-bit cache takes 66 MiB per 8K positions, so 32K (+132 MiB) or 64K (+400 MiB) fit.

1. `shaders/attn_tq.h:27` (`TQ_AT`): block 0 or 1 -> block `p / TQ_SEQ` at `(p / TQ_SEQ) * split` bytes.
2. `chat/engine.c` engine_init: allocate N blocks with that stride, RoPE table and attention partitions for the new
   max_seq; drop the kv_hi / lend_ctx special case (voice is gone).
3. `chat/engine_spec.inc` spec_layout: the MTP head's cache is sized for two blocks (`4 * kb`).
4. Rebuild `kernels.bin` from `shaders/` with LLVM's amdgcn target (`pc/export_kernels_9b.py`).

Cost: at 64K each token also reads ~0.5 GB of cache (vs 4.6 GB of weights), so long conversations get slower.
