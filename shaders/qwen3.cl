// Qwen3 decode-step kernels for the PS4 GPU (GCN gfx701), compiled by LLVM from OpenCL C 1.2 without its
// standard headers. One token per command buffer step: embed -> layers x (norm, qkv, q/k norm + RoPE + cache,
// attention, o, add, norm, gate/up, SiLU x up, down, add) -> final norm -> logits -> sample. The next token never
// leaves the GPU.

#ifndef HEAD_DIM
#define HEAD_DIM 128            // (LFM2: -DHEAD_DIM=64, pc/convert_gguf.py)
#endif
#define ROPE_DIM HEAD_DIM
#ifndef MAX_SEQ
#define MAX_SEQ 2048            // (-DMAX_SEQ=1024: the embedder's kernels, pc/convert_gguf.py)
#endif

#include "common.h"
#include "attn.h"

// LFM2's short convolution, in place: in = B | C | x (n channels each) -> in[0, n) = C * (w0 s0 + w1 s1 + w2 B x);
// the state keeps the last 2 B x per channel. w: [n][3]; state: [n][2] (zero at the start of a conversation).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void short_conv(global float* in, global const float* w, global float* state, uint n) {
    uint c = GID * 256 + LID;
    if (c >= n) return;
    global float* s = state + 2 * (ulong)c;
    global const float* wc = w + 3 * (ulong)c;
    float bx = in[c] * in[2 * n + c], s0 = s[0], s1 = s[1];
    in[c] = in[n + c] * (wc[0] * s0 + wc[1] * s1 + wc[2] * bx);
    s[0] = s1;
    s[1] = bx;
}

#ifdef PREFILL   // LFM2: the prompt in batches (chat/engine.c prefill; the DeltaNet kernels there go unused)
#define DK 128
#define DV 128
#include "prefill.h"

// short_conv for `rows` consecutive tokens: in rows of 3n (B | C | x), out rows of n; the state stays in registers.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void short_conv_rows(global const float* in, global const float* w, global float* state, global float* out, uint n,
                     uint rows) {
    uint c = GID * 256 + LID;
    if (c >= n) return;
    global float* s = state + 2 * (ulong)c;
    global const float* wc = w + 3 * (ulong)c;
    float w0 = wc[0], w1 = wc[1], w2 = wc[2], s0 = s[0], s1 = s[1];
    for (uint t = 0; t < rows; t++) {
        global const float* r = in + (ulong)t * 3 * n;
        float bx = r[c] * r[2 * n + c];
        out[(ulong)t * n + c] = r[n + c] * (w0 * s0 + w1 * s1 + w2 * bx);
        s0 = s1;
        s1 = bx;
    }
    s[0] = s0;
    s[1] = s1;
}
#endif
