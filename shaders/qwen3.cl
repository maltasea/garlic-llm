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

// ---- 8-bit weights ("q8", fmt 9: the embedder, pc/convert_gguf.py) -----------------------------------------------
// GGUF's Q8_0 values as they are: per row cols int8, then (separately) one fp16 scale per block of 32
// [rows][cols / 32]; w = q * d.

// y = W x for 4 rows per 64-thread group; a lane takes 16 weights (half a block) at a time, so a row of 1024 keeps
// all 64 lanes busy.
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void matvec_q8(global const uchar* w, global const ushort* scales, global const float* x, global float* y,
               uint cols) {
    local float red[4][64];
    uint lane = LID, r0 = GID * 4, nb = cols / 32;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint h = lane; h < cols / 16; h += 64) {
        float xh[16];
        global const float4v* xv = (global const float4v*)(x + h * 16);
        for (uint k = 0; k < 4; k++) {
            float4v v = xv[k];
            xh[4 * k] = v.x; xh[4 * k + 1] = v.y; xh[4 * k + 2] = v.z; xh[4 * k + 3] = v.w;
        }
        for (uint r = 0; r < 4; r++) {
            ulong row = r0 + r;
            uint4v q = *(global const uint4v*)(w + row * cols + h * 16);
            uint words[4] = {q.x, q.y, q.z, q.w};
            float s = 0.0f;
            for (uint k = 0; k < 4; k++)
                for (uint b = 0; b < 4; b++) s += (float)((int)(words[k] << (24 - 8 * b)) >> 24) * xh[4 * k + b];
            acc[r] += h2f(scales[row * nb + h / 2]) * s;
        }
    }
    for (uint r = 0; r < 4; r++) red[r][lane] = acc[r];
    BARRIER();
    for (uint s = 32; s > 0; s >>= 1) {
        if (lane < s)
            for (uint r = 0; r < 4; r++) red[r][lane] += red[r][lane + s];
        BARRIER();
    }
    if (lane < 4) y[r0 + lane] = red[lane][0];
}

// x = dequantized row `token` of a q8 embedding (token from `forced` or the last pick).
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void embed_q8(global const schar* w, global const ushort* scales, global const uint* last, uint forced,
              global float* x, uint cols) {
    uint tok = forced != NO_TOKEN ? forced : *last;
    ulong row = (ulong)tok * cols;
    for (uint i = LID; i < cols; i += 256) x[i] = (float)w[row + i] * h2f(scales[(row + i) >> 5]);
}

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

#ifdef PREFILL   // LFM2, Qwen3: the prompt in batches (chat/engine.c prefill; the DeltaNet kernels there go unused)
#define DK 128
#define DV 128
#include "prefill.h"

// mm_q4i_rows (prefill.h) for q8 weights (fmt 9, rows as they are): lane = output row of the tile, its 32 int8 of
// block b and their scale; the same group shape (64 rows x MM_ROWS tokens), sums and stores.
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void mm_q8_rows(global const uchar* restrict w, global const ushort* restrict sc, global const float* restrict X,
                global float* restrict Y, uint cols, uint rows, uint n, uint accumulate) {
    local float part[4][MM_ROWS][64];
    uint l = LID & 63, nb = cols / 32, nch = (n + MM_ROWS - 1) / MM_ROWS;
    uint wv = __builtin_amdgcn_readfirstlane(LID >> 6);
    uint tile = GID / nch, r0 = (GID % nch) * MM_ROWS;
    uint nv = n - r0 < MM_ROWS ? n - r0 : MM_ROWS;
    ulong row = (ulong)tile * 64 + l;
    global const uint4v* wp = (global const uint4v*)(w + row * cols);
    global const ushort* sp = sc + row * nb;
    global const float* x0 = X + (ulong)r0 * cols;
    float acc[MM_ROWS];
    for (uint v = 0; v < MM_ROWS; v++) acc[v] = 0.0f;
    for (uint b = wv; b < nb; b += 4) {
        uint4v qa = wp[2 * b], qb = wp[2 * b + 1];
        float d = h2f(sp[b]);
        uint ws[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
        float wf[32];
        for (uint k = 0; k < 8; k++)
            for (uint j = 0; j < 4; j++) wf[4 * k + j] = (float)((int)(ws[k] << (24 - 8 * j)) >> 24) * d;
#pragma unroll
        for (uint v = 0; v < MM_ROWS; v++) {
            if (v < nv) {
                global const float* xb = x0 + (ulong)v * cols + b * 32;
                float s = 0.0f;
                for (uint j = 0; j < 32; j++) s += wf[j] * xb[j];
                acc[v] += s;
            }
        }
    }
    for (uint v = 0; v < MM_ROWS; v++) part[wv][v][l] = acc[v];
    BARRIER();
    for (uint v = wv; v < nv; v += 4) {
        float r = part[0][v][l] + part[1][v][l] + part[2][v][l] + part[3][v][l];
        global float* y = Y + (ulong)(r0 + v) * rows + tile * 64 + l;
        *y = accumulate ? *y + r : r;
    }
}

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

// Attention for `n` consecutive prompt tokens (rows) at positions pos0.., after their k and v are in the cache
// (qk_rope_rows). Group = (kv head, AR_ROWS rows, slice of the positions): its rows x gq q heads (up to 32 queries)
// go through its slice of the cache together, 64 positions at a time, with an online softmax: each k and v is read
// once per group instead of once per token. The slices (`split` of them) spread a long context over the GPU; each
// writes its unnormalised output, max and sum to part, and attn_rows_combine merges them. q rows of ldq floats (the
// q heads first). Positions past a row's own get weight 0 (causal); past the last one written, the position is
// clamped (the cache there may hold NaNs).
#define AR_ROWS 8
#define AR_Q (AR_ROWS * GQ_MAX)
#define AR_PARTS (256 / HEAD_DIM)
#define AR_QPT (AR_Q / AR_PARTS)
kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void attn_rows(global const float* q, global const half_t* kcache, global const half_t* vcache, global float* part,
               uint pos0, uint n, uint n_heads, uint n_kv, uint ldq, float scale, uint split) {
    local float qs[AR_Q][HEAD_DIM];
    local float s[AR_Q][ATT_BLOCK];
    local float mq[AR_Q], lq[AR_Q], aq[AR_Q];
    uint t = LID, lane = t & 63, qg = t >> 6, d = t % HEAD_DIM, pt = t / HEAD_DIM;
    uint nch = (n + AR_ROWS - 1) / AR_ROWS, kv = GID % n_kv, ch = GID / n_kv % nch, sl = GID / n_kv / nch;
    uint r0 = ch * AR_ROWS, gq = n_heads / n_kv;
    uint nr = n - r0 < AR_ROWS ? n - r0 : AR_ROWS, nq = nr * gq, pmax = pos0 + r0 + nr - 1;
    uint nblk = pmax / ATT_BLOCK + 1, per = (nblk + split - 1) / split;
    uint b0 = sl * per, b1 = b0 + per < nblk ? b0 + per : nblk;
    for (uint i = t; i < AR_Q * HEAD_DIM; i += 256) {   // query i = row * gq + head
        uint qi = i / HEAD_DIM, dd = i % HEAD_DIM;
        qs[qi][dd] = qi < nq ? q[(ulong)(r0 + qi / gq) * ldq + (kv * gq + qi % gq) * HEAD_DIM + dd] * scale : 0.0f;
    }
    if (t < AR_Q) mq[t] = -3.0e38f, lq[t] = 0.0f;
    float o[AR_QPT];
    for (uint i = 0; i < AR_QPT; i++) o[i] = 0.0f;
    BARRIER();
    for (uint bk = b0; bk < b1; bk++) {
        uint p0 = bk * ATT_BLOCK, p = p0 + lane;
        float a[AR_Q / 4];   // scores: lane = position, wave qg = queries qg * AR_Q / 4 ..
        for (uint i = 0; i < AR_Q / 4; i++) a[i] = 0.0f;
        if (p <= pmax) {
            global const half_t* k = kcache + (ulong)kv * HEAD_DIM * MAX_SEQ + p;
            for (uint dd = 0; dd < HEAD_DIM; dd++) {
                float kf = (float)k[(ulong)dd * MAX_SEQ];
                for (uint i = 0; i < AR_Q / 4; i++) a[i] += qs[qg * (AR_Q / 4) + i][dd] * kf;
            }
        }
        for (uint i = 0; i < AR_Q / 4; i++) {
            uint qi = qg * (AR_Q / 4) + i;
            s[qi][lane] = qi < nq && p <= pos0 + r0 + qi / gq ? a[i] : -3.0e38f;
        }
        BARRIER();
        if (t < nq) {   // online softmax: the new max, the weights in place, the old sum rescaled
            float m0 = mq[t], mx = m0;
            for (uint j = 0; j < ATT_BLOCK; j++) mx = __builtin_fmaxf(mx, s[t][j]);
            float sum = 0.0f;
            for (uint j = 0; j < ATT_BLOCK; j++) {
                float e = s[t][j] > -1.0e38f ? exp_(s[t][j] - mx) : 0.0f;
                s[t][j] = e;
                sum += e;
            }
            float al = m0 > -1.0e38f ? exp_(m0 - mx) : 0.0f;
            aq[t] = al, lq[t] = lq[t] * al + sum, mq[t] = mx;
        }
        BARRIER();
        for (uint i = 0; i < AR_QPT; i++) o[i] *= aq[pt * AR_QPT + i];
        for (uint j = 0; j < ATT_BLOCK; j++) {
            uint pc = p0 + j <= pmax ? p0 + j : pmax;
            float v = (float)vcache[((ulong)pc * n_kv + kv) * HEAD_DIM + d];
            for (uint i = 0; i < AR_QPT; i++) o[i] += s[pt * AR_QPT + i][j] * v;
        }
        BARRIER();
    }
    // part: [row][q head][slice] x (HEAD_DIM outputs, then max, sum)
    for (uint i = 0; i < AR_QPT; i++) {
        uint qi = pt * AR_QPT + i;
        if (qi < nq) {
            global float* pp = part + (((ulong)(r0 + qi / gq) * n_heads + kv * gq + qi % gq) * split + sl) * (HEAD_DIM + 2);
            pp[d] = o[i];
            if (d == 0) pp[HEAD_DIM] = mq[qi], pp[HEAD_DIM + 1] = lq[qi];
        }
    }
}

// Merges attn_rows' slices: group = (row, q head), HEAD_DIM threads; out rows of n_heads * HEAD_DIM.
kernel __attribute__((reqd_work_group_size(HEAD_DIM, 1, 1)))
void attn_rows_combine(global const float* part, global float* out, uint split) {
    global const float* pp = part + (ulong)GID * split * (HEAD_DIM + 2);
    float m = -3.0e38f;
    for (uint k = 0; k < split; k++) m = __builtin_fmaxf(m, pp[k * (HEAD_DIM + 2) + HEAD_DIM]);
    float o = 0.0f, sum = 0.0f;
    for (uint k = 0; k < split; k++) {
        global const float* x = pp + k * (HEAD_DIM + 2);
        float w = x[HEAD_DIM] > -1.0e38f ? exp_(x[HEAD_DIM] - m) : 0.0f;
        o += w * x[LID];
        sum += w * x[HEAD_DIM + 1];
    }
    out[(ulong)GID * HEAD_DIM + LID] = o / sum;
}

// qk_rope_cache (attn.h) for `n` consecutive rows in one dispatch: group = row * (n_heads + 2 n_kv) + g, row r at
// position pos0 + r, its q | k | v at qkv + r * ldq. (Plain cache only: no ATTN_SPLIT, no V_INT8.)
kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void qk_rope_rows(global float* qkv, global const float* q_norm, global const float* k_norm, global const float* rope,
                  global half_t* kcache, global half_t* vcache, uint pos0, uint n_heads, uint n_kv, float eps,
                  uint ldq) {
    local float red[64];
    local float hv[HEAD_DIM];
    uint G = n_heads + 2 * n_kv, r = GID / G, g = GID % G, j = LID, pos = pos0 + r;
    qkv += (ulong)r * ldq;
    if (g < n_heads + n_kv) {
        global float* h = qkv + g * HEAD_DIM;
        float x[EPT];
        float ss = 0.0f;
        for (uint m = 0; m < EPT; m++) {
            x[m] = h[j + 64 * m];
            ss += x[m] * x[m];
        }
        red[j] = ss;
        BARRIER();
        for (uint s = 32; s > 0; s >>= 1) {
            if (j < s) red[j] += red[j + s];
            BARRIER();
        }
        float rr = rsq(red[0] / (float)HEAD_DIM + eps);
        global const float* w = g < n_heads ? q_norm : k_norm;
        for (uint m = 0; m < EPT; m++) {
            x[m] *= rr * w[j + 64 * m];
            hv[j + 64 * m] = x[m];
        }
        BARRIER();
        const uint rh = ROPE_DIM / 2;
        global const float* rp = rope + (ulong)pos * ROPE_DIM;
        for (uint m = 0; m < EPT; m++) {
            uint i = j + 64 * m;
            float y = x[m];
            if (i < ROPE_DIM) {
                float c = rp[i % rh], sn = rp[rh + i % rh];
                y = i < rh ? y * c - hv[i + rh] * sn : y * c + hv[i - rh] * sn;
            }
            if (g < n_heads) h[i] = y;
            else kcache[((ulong)(g - n_heads) * HEAD_DIM + i) * MAX_SEQ + pos] = (half_t)y;
        }
    } else {
        uint kv = g - n_heads - n_kv;
        global const float* v = qkv + (n_heads + n_kv + kv) * HEAD_DIM;
        global half_t* dst = vcache + ((ulong)pos * n_kv + kv) * HEAD_DIM;
        for (uint m = 0; m < EPT; m++) dst[j + 64 * m] = (half_t)v[j + 64 * m];
    }
}
#endif
