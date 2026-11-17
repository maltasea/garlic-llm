// Qwen3 / Qwen3.5 inference on the PS4 GPU (the same command packets as pc/gpu.py).
#ifndef CHAT_ENGINE_H
#define CHAT_ENGINE_H

#include <stdint.h>

// Speculative decoding with the MTP head (engine_spec.inc): the 9B app only. Not the 4B (its own MTP head is not
// converted). Without mtp.idx or the q4s model it stays off (spec_init fails).
#if !defined(GARLIC_4B)
#define GARLIC_SPEC 1
#endif

typedef struct {
    char name[32];
    uint64_t addr;
    uint32_t rsrc1, rsrc2, user_sgprs, kernarg_sgpr;
} GpuKernel;

typedef struct {
    uint64_t off, size;
    uint32_t rows, cols;
    uint64_t scales;
    uint32_t fmt;
} Tensor;

typedef void (*ProgressFn)(void* user, float fraction, const char* what);

typedef struct {
    // model shape (assets/model.idx)
    uint32_t n_layers, hidden, inter, n_heads, n_kv, head_dim, vocab, max_seq, eos, eos2;
    float eps, rope_theta;
    // Qwen3.5 (model.idx "PMQ5"): Gated DeltaNet shape, partial RoPE, which layers are full attention
    int qwen35;
    uint32_t n_kh, n_vh, dk, dv, rope_dim, full_mask, im_start, im_end;
    // LFM2 (model.idx "PML2"): the layers not in full_mask are short convolutions (shaders/qwen3.cl short_conv);
    // bos starts every prompt, think / think_end: the model's (empty) think block, 0 = the server's own
    int lfm;
    uint32_t bos, think, think_end;
    uint32_t* layer_base;   // index in t[] of each layer's first tensor (attn_norm)
    int has_head;           // untied output head: the last tensor (Qwen3.5-9B), or the one before "ids"
    uint32_t head_idx;      // index of lm_head in t[] (0 = tied to the embedding)
    // The 9B's q4s model (pc/convert_9b_q4s.py): lm_head trimmed to `vocab` rows ordered by frequency; ids[row] is
    // the token id of each row (NULL: rows are ids). The sampled index is a row; engine_run returns the id.
    uint32_t* ids;
    uint32_t last_row;      // the row engine_run drew last (its token: engine_token_of)
    int cpu_embed;          // input embedding rows stay in the file (fmt 4); the CPU reads each token's row
    int fd;                 // the weights file, kept open for those rows
    uint8_t* row_cache;     // direct-mapped cache of embedding rows
    int32_t* row_tag;
    uint64_t staging;       // one dequantized row per step of a submission (GPU memory)
    char weights_path[129];
    uint64_t weights_size;
    Tensor* t;
    uint32_t n_tensors;

    // GPU-visible memory (Garlic): command buffer, kernel arguments, code, model, caches, activations
    uint8_t* mem;
    uint64_t mem_size, next;
    uint32_t* dcb;
    uint32_t ndw, dcb_cap;
    uint8_t* kargs;
    uint32_t kargs_used, kargs_cap;
    volatile uint32_t* label;
    volatile uint64_t* stamps;
    uint64_t model, rope, layer_cache, attn_part;
    uint64_t *kc, *vc;            // per layer: KV cache (full attention) or conv state / delta-rule state (DeltaNet)
    uint64_t conv_n, state_bytes;
    // Qwen3.5: each full layer's cache is two halves (ATTN_SPLIT in shaders/attn.h); the second halves of all the
    // layers form one block [kv_hi, kv_hi + kv_split)
    uint64_t kv_hi, kv_split;
    uint32_t ctx_limit;           // positions the conversation may use (max_seq)
    uint32_t ctx_cap;             // engine_ctx: at most this many positions (0 = as many as fit)
    int wait_us;                  // waiting for the GPU: sleep this long between checks (0 = spin)
    // TurboQuant 4-bit KV cache (shaders/attn_tq.h), allocated by engine_ctx
    int tq;
    uint64_t tq_tab;              // 40 words for the kernels: levels, boundaries, sign bits (pc/turboquant.py)
    const GpuKernel *k_qk_tq, *k_attn_tq, *k_comb_tq;
    // With TurboQuant, positions 8192 b .. 8192 b + 8191 are in cache block b, b * tq_split bytes further
    uint64_t tq_split;
    int tq16;                     // kernels.bin's TurboQuant kernels take tq_split (blocks of 8192 positions)
    // The prompt in batches of up to PF_ROWS tokens (shaders/prefill.h): row buffers for the residual (pf_x),
    // its norm (pf_xn), the projections (pf_a: DeltaNet input, q|k|v|gate or gate|up) and their products (pf_h:
    // attention or delta-rule output, SiLU(gate) * up)
    int pf, pf_off;               // batched prompt available; pf_off = 1: token by token (tests)
    uint64_t pf_x, pf_xn, pf_a, pf_h;
    const GpuKernel *k_mm4r, *k_mm3r, *k_rmsr, *k_silur, *k_sigr, *k_convr, *k_deltar;
    const GpuKernel* k_shortr;   // LFM2: short_conv_rows
    const GpuKernel *k_attnr, *k_attnrc, *k_qkr;   // PMDL / PML2: the prompt's attention, all rows at once
    double pf_seconds;            // GPU time of the last batched prompt (diagnostics)
    uint64_t submit_tsc;          // when the last submission went to the GPU (submit_go)
    // the fastest product kernel for each matrix shape, timed at start-up (engine.c mm_tune): [0] for batches of
    // more than 16 tokens, [1] for up to 16; r = tokens per group of that kernel
    struct { uint32_t fmt, rows, cols, r[2]; const GpuKernel* k[2]; } pf_mm[8];
    int pf_nmm;
    uint64_t x, xn, attn, hmlp, qkv, lin, o, gu, down, logits, last, history;
    GpuKernel k[200];             // the kernels of kernels.bin
    int nk;
    const GpuKernel *k_embed, *k_rmsnorm, *k_qk, *k_attn, *k_attn_comb, *k_silu, *k_add, *k_mv, *k_sample;
    const GpuKernel *k_sig, *k_conv, *k_delta;   // Qwen3.5 only
    const GpuKernel* k_short;                    // LFM2 only
    const GpuKernel *k_bsum, *k_mvi, *k_mvi8;    // interleaved Q4 ("q4i", fmt 3) matrices
    const GpuKernel *k_mv3, *k_mv38, *k_copy;    // interleaved Q3 ("q3i", fmt 5); staging row -> x
    int q8;                                      // q8 matrices (fmt 9, the embedder): matvec_q8
    const GpuKernel* k_mv8;
    uint64_t bsum;                               // block sums of the current matrix-vector input
    // SWAR int4 x int8 products ("q4s", fmt 6; shaders/spec.h): the activations in int8 per block of 32 (x8, dx,
    // init) for up to PF_ROWS rows, the split-K partial sums, and per matrix shape and number of tokens (1..4) the
    // fastest kernel variant (swar_tune at start-up)
    int swar;
    const GpuKernel *k_qx8, *k_sumsplit, *k_swar[4][5], *k_swar3[4][5];   // q4s (fmt 6), q3s (fmt 8)
    uint64_t x8, dx8, init8, yp;
    uint64_t yp_rows;                            // the largest rows x 4 tokens x 4 parts the partial sums take
    struct { uint32_t rows, cols, fmt; uint8_t v[4]; } swar_tab[24];
    int swar_n;
    uint32_t init_state[256];
    uint64_t tsc_freq;
    double last_gpu_seconds;
    char error[160];
} Engine;

// Direct memory engine_init leaves to the system (MiB; main.c sets it before engine_init).
extern int eng_margin_mb;
// Reserves GPU memory, reads kernels.bin from assets_dir and model.idx from model_dir and loads the weights file.
int engine_init(Engine* e, const char* assets_dir, const char* model_dir, ProgressFn progress, void* user);
// Another model in the same GPU memory (everything set up again; after it engine_ctx and the MTP head as after
// engine_init). On failure the engine holds no model: load one again before using it.
int engine_load(Engine* e, const char* assets_dir, const char* model_dir, ProgressFn progress, void* user);
// A second engine (the embedding model, Qwen3) in [mem, mem + size) of the first one's block (engine_alloc);
// engine_need: the size it takes, from model_dir/model.idx (0 = cannot tell, or a Qwen3.5 model).
int engine_init_at(Engine* e, uint64_t mem, uint64_t size, const char* assets_dir, const char* model_dir,
                   ProgressFn progress, void* user);
uint64_t engine_need(const char* model_dir);
// Runs n tokens through the model starting at position pos0 and samples the token after the last one.
// temperature 0 = greedy. Returns 0 and the sampled token in *next, or -1 with e->error set.
// (n >= 4 goes through the layers in batches of up to 32 tokens, shaders/prefill.h, unless e->pf_off.)
int engine_run(Engine* e, const uint32_t* tokens, int n, int pos0, float temperature, uint32_t seed, uint32_t* next);
// The token id of an lm_head row (rows are ids unless lm_head is trimmed: Engine.ids), and the row of a token id
// (-1 if that token is not in lm_head: it can be read but not written).
uint32_t engine_token_of(const Engine* e, uint32_t row);
int32_t engine_row_of(const Engine* e, uint32_t token);
// Forget the conversation: clears the Gated DeltaNet states (the KV cache is simply overwritten from position 0).
void engine_reset(Engine* e);
// The Gated DeltaNet states saved to GPU memory at buf (engine_state_size bytes; save 1) or put back from it
// (save 0), so that the conversation can go back to the position where they were saved. 0 = done.
uint64_t engine_state_size(const Engine* e);
int engine_state_copy(Engine* e, uint64_t buf, int save);
// With the TurboQuant cache: the cache, after everything else (call it once, after engine_init, before the first
// run). As many blocks of 8192 positions as fit (up to 64K), leaving `reserve` bytes free and counting per_block
// more bytes for each block (the MTP head's cache). Sets max_seq and ctx_limit. 0 = done (or nothing to do).
int engine_ctx(Engine* e, uint64_t per_block, uint64_t reserve);
// GPU memory of the engine's block (before engine_ctx, which takes what is left); 0 = no room.
uint64_t engine_alloc(Engine* e, uint64_t size);
// Speculative decoding with Qwen3.5's MTP head (GARLIC_SPEC; engine_spec.inc). spec_init reads mtp.idx from the
// engine's assets; spec_start gives the head, its draft rows and its cache GPU memory of their own and loads them
// (spec_load: into [base, base + size); start = the conversation position it follows from, 0 = empty).
// spec_need: what spec_start will take, before engine_ctx (*per_block more for each cache block of 8192 positions).
int spec_init(Engine* e, const char* dir);
uint64_t spec_need(Engine* e, uint64_t* per_block);
int spec_start(Engine* e);
int spec_load(Engine* e, uint64_t base, uint64_t size, int start);
void spec_unload(Engine* e);
int spec_active(const Engine* e);
// One cycle: feeds tok (drawn last, at pos) and the draft's tokens, keeps what the exact acceptance keeps (out[0..*n_out),
// at most max_new - 1 of them) and draws the next token. Same distribution as engine_run one token at a time.
int engine_spec_step(Engine* e, uint32_t tok, int pos, float temperature, uint32_t seed, int max_new, uint32_t* out,
                     int* n_out, uint32_t* next);
void spec_stats(const Engine* e, char* buf, int n);
void spec_stats_reset(void);
// The draft length cap (0 = the head off; k < 0 keeps it); returns the old one.
int spec_kmax(int k);
float spec_theta(float theta);   // the draft's stop threshold (theta < 0 keeps it); returns the old one

#endif
