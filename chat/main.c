// Garlic LLM: Qwen3.5 on the PS4's own GPU, served over the LAN as an OpenAI-compatible API (server.c). This file
// is the console side: log, screen, start-up; the screens are drawn by ui.c.
// The model weights live in /data (see model.idx); GPU code, tokenizer and fonts are in the package's assets.

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <dirent.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>
#include <orbis/SystemService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include "engine.h"
#include "server.h"
#include "tokenizer.h"
#include "ui.h"

#define ASSETS "/app0/assets"
#define THINK 248068u           // Qwen3.5 <think> and </think>: an empty block = answer without thinking
#define THINK_END 248069u
#define MAX_REPLY 3000
// One log per app (so that opening the other app does not lose this one's log), and the one before it
#if defined(GARLIC_4B)
#define LOG_NAME "/data/ps4ai_log_4b"
#else
#define LOG_NAME "/data/ps4ai_log"
#endif
#define LOG_FILE LOG_NAME ".txt"
#define LOG_PREV LOG_NAME "_prev.txt"

uint32_t* back;

// Log to the kernel log and to LOG_FILE (synced every line, so it survives a crash; read it over FTP).
void say(const char* fmt, ...) {   // a line in the log, with the seconds since the app started
    char buf[420];
    int h = snprintf(buf, 32, "[garlic %8.3f] ", sceKernelGetProcessTime() / 1e6);
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + h, sizeof(buf) - h - 2, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof(buf) - h - 2) n = sizeof(buf) - h - 2;
    buf[h + n] = '\n';
    buf[h + n + 1] = 0;
    sceKernelDebugOutText(0, buf);
    int f = open(LOG_FILE, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (f >= 0) {
        write(f, buf, h + n + 1);
        fsync(f);
        close(f);
    }
}

uint8_t* read_all(const char* path, size_t* size) {
    int f = open(path, O_RDONLY);
    if (f < 0) return NULL;
    off_t n = lseek(f, 0, SEEK_END);
    lseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc(n);
    size_t done = 0;
    while (buf && done < (size_t)n) {
        ssize_t r = read(f, buf + done, n - done);
        if (r <= 0) break;
        done += r;
    }
    close(f);
    *size = done;
    return buf;
}

static double now(void) { return (double)sceKernelReadTsc() / sceKernelGetTscFrequency(); }

// ---- screen ------------------------------------------------------------------------------------------------------

static int video;
static OrbisKernelEqueue flip_queue;
static uint32_t* fb[2];
static int cur, frame = 1;

static int video_init(void) {
    video = sceVideoOutOpen(ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, 0);
    if (video < 0) return -1;
    sceKernelCreateEqueue(&flip_queue, "garlic flip");
    sceVideoOutAddFlipEvent(flip_queue, video, 0);
    size_t size = (2ull * SCREEN_W * SCREEN_H * 4 + 0x1FFFFF) & ~0x1FFFFFull;
    off_t off;
    void* mem = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, 0x200000, 3, &off)) return -2;
    if (sceKernelMapDirectMemory(&mem, size, 0x33, 0, off, 0x200000)) return -3;
    fb[0] = mem;
    fb[1] = (uint32_t*)mem + SCREEN_W * SCREEN_H;
    OrbisVideoOutBufferAttribute attr;
    sceVideoOutSetBufferAttribute(&attr, 0x80000000, 1, 0, SCREEN_W, SCREEN_H, SCREEN_W);
    if (sceVideoOutRegisterBuffers(video, 0, (void**)fb, 2, &attr)) return -4;
    sceVideoOutSetFlipRate(video, 0);
    back = ui_screen_buf();   // drawn in cached memory, then copied to the write-combined frame buffer
    return back ? 0 : -5;
}

static void present(void) {
    static int stuck;
    static double tried;
    if (stuck && now() - tried < 5) return;   // (no screen: try again every 5 s, not at every frame)
    tried = now();
    memcpy(fb[cur], back, SCREEN_W * SCREEN_H * 4);
    sceVideoOutSubmitFlip(video, cur, ORBIS_VIDEO_OUT_FLIP_VSYNC, frame);
    // (at most half a second: after rest mode, or with the TV off, the flip may never come, and the server would stop)
    for (int i = 0;; i++) {
        OrbisVideoOutFlipStatus st;
        sceVideoOutGetFlipStatus(video, &st);
        if (st.flipArg >= frame) {
            if (stuck) say("screen: flips again");
            stuck = 0;
            break;
        }
        if (i == 5) {
            if (!stuck) say("screen: no flip for 0.5 s");
            stuck = 1;
            break;
        }
        OrbisKernelEvent ev;
        int count;
        OrbisKernelUseconds wait = 100000;
        sceKernelWaitEqueue(flip_queue, &ev, 1, &count, &wait);
    }
    cur ^= 1;
    frame++;
}

static void loading(void* user, float frac, const char* what) {
    static double last;
    double t = now();
    if (frac < 1.0f && t - last < 0.05) return;
    last = t;
    draw_loading(frac, user ? (const char*)user : what);
    present();
}

static void show_error(const char* what) {
    say("fatal: %s", what);
    for (;;) {
        draw_error(what);
        present();
    }
}

static void boot_report(void) {
    OrbisKernelSwVersion v;
    memset(&v, 0, sizeof(v));
    v.Size = sizeof(v);
    int r = sceKernelGetSystemSwVersion(&v);
    v.VersionString[sizeof(v.VersionString) - 1] = 0;
    const char* fw = v.VersionString;
    while (*fw == ' ' || *fw == '0') fw++;   // " 11.520.001" -> "11.520.001"
    size_t flex = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
    say("console: firmware %s (0x%08x%s); PS4 Pro hardware %d, Neo mode %d; direct memory %u MiB, system RAM free "
        "%u MiB; CPU TSC %.1f MHz, cpumode %d", r ? "?" : fw, (unsigned)v.Version, r ? ", query failed" : "",
        sceKernelHasNeoMode(), sceKernelIsNeoMode(), (unsigned)(sceKernelGetDirectMemorySize() >> 20),
        (unsigned)(flex >> 20), sceKernelGetTscFrequency() / 1e6, sceKernelGetCpumode());
}

// ---- the server --------------------------------------------------------------------------------------------------

static Engine eng;
static Tokenizer tok;

static int encode_into(uint32_t* ids, int n, int max, const char* text) { return n + tok_encode(&tok, text, ids + n, max - n); }

static int shown = -1;   // what the server screen shows (-1: draw it again)

static int server_frame(void* user, const ServerStatus* st) {
    static char ip[16];
    const ServerConfig* cfg = user;
    // A frame only when what it shows changes (while answering, present would wait for the TV's refresh, up to 16 ms
    // per token); idle, the server checks for a connection 50 times a second and sleeps in between
    int now_shown = st->busy ? -2 - st->served : st->served;
    if (now_shown != shown || strcmp(ip, st->ip)) {
        shown = now_shown;
        memcpy(ip, st->ip, sizeof(ip));
        char models[200];   // what is loaded: the chat model, its context, the embedding model
        int n = snprintf(models, sizeof(models), "%s · %uK context", cfg->model, eng.ctx_limit / 1024);
        if (cfg->embed_model[0]) snprintf(models + n, sizeof(models) - n, " · embeddings: %s", cfg->embed_model);
        draw_server(st->ip, st->port, st->busy, st->busy ? st->served + 1 : st->served, models);
        present();
    } else if (!st->busy) {
        sceKernelUsleep(20000);
    }
    return 0;   // the server runs until the app is closed
}

// The tokenizer: the model folder's tokenizer.bin, else the package's (every Qwen3.5 size has the same)
static int load_tokenizer(const char* dir) {
    static uint8_t* data;
    char p[160];
    size_t size;
    snprintf(p, sizeof(p), "%s/tokenizer.bin", dir);
    uint8_t* d = read_all(p, &size);
    if (!d) d = read_all(ASSETS "/tokenizer.bin", &size);
    Tokenizer t;
    if (!d || tok_load(&t, d, size)) {
        free(d);
        return -1;
    }
    tok_free(&tok);
    free(data);
    data = d, tok = t;
    return 0;
}

// ---- the embedding model: a folder of MODELS_DIR with embedding.txt (pc/convert_gguf.py), in the chat model's block

static Engine emb;
static Tokenizer etok;

static int embed_text(ServerConfig* cfg, const char* text, float* out, int* tokens) {
    (void)cfg;
    int max = (int)emb.max_seq;
    uint32_t* ids = malloc(max * sizeof(uint32_t));
    if (!ids) return -1;
    int n = tok_encode(&etok, text ? text : "", ids, max - 1);
    if (n < 0) n = 0;
    ids[n++] = emb.eos2;   // (Qwen3-Embedding: the text, then <|endoftext|>, whose state is the embedding)
    uint32_t next;
    engine_reset(&emb);
    int r = engine_run(&emb, ids, n, 0, 0.0f, 0, &next);
    free(ids);
    if (r) return say("embedding: %s", emb.error), -1;
    const volatile float* x = (const volatile float*)(uintptr_t)emb.xn;   // the final norm at the last token
    double s = 0;
    for (uint32_t k = 0; k < emb.hidden; k++) out[k] = x[k], s += (double)out[k] * out[k];
    float f = s > 0 ? (float)(1.0 / sqrt(s)) : 0.0f;
    for (uint32_t k = 0; k < emb.hidden; k++) out[k] *= f;
    *tokens = n;
    return 0;
}

// The first folder with embedding.txt, loaded into the chat model's GPU block (before engine_ctx takes the rest)
static void load_embedder(ServerConfig* cfg) {
    static uint8_t* tdata;
    cfg->embed_model[0] = 0;
    DIR* d = opendir(MODELS_DIR);
    char dir[160] = "";
    for (struct dirent* x; d && (x = readdir(d));) {
        if (x->d_name[0] == '.') continue;
        char p[192];
        snprintf(p, sizeof(p), MODELS_DIR "/%s/embedding.txt", x->d_name);
        int f = open(p, O_RDONLY);
        if (f < 0) continue;
        close(f);
        snprintf(dir, sizeof(dir), MODELS_DIR "/%s", x->d_name);
        snprintf(cfg->embed_model, sizeof(cfg->embed_model), "%s", x->d_name);
        break;
    }
    if (d) closedir(d);
    if (!dir[0]) return;
    char p[192];
    size_t size;
    snprintf(p, sizeof(p), "%s/tokenizer.bin", dir);
    tok_free(&etok);
    free(tdata);
    tdata = read_all(p, &size);
    uint64_t need = engine_need(dir), base = need ? engine_alloc(&eng, need) : 0;
    if (!tdata || tok_load(&etok, tdata, size) || !base ||
        engine_init_at(&emb, base, need, dir, dir, loading, (void*)tr(S_LOADING_MODEL))) {
        say("embedding model %s: %s", cfg->embed_model, !tdata ? "no tokenizer.bin" : !base ? "no memory" : emb.error);
        cfg->embed_model[0] = 0;
        return;
    }
    cfg->embed_dim = (int)emb.hidden, cfg->embed = embed_text;
    say("embedding model %s: %u MiB, dimension %u, up to %u tokens", cfg->embed_model, (unsigned)(need >> 20), emb.hidden,
        emb.max_seq - 1);
}

// Model `id` (MODEL_ID: the package's idx files; else the folder MODELS_DIR/id) with its context and MTP head, into
// the engine's GPU memory; the last one loaded is the next start's (MODELS_DIR/default.txt).
static int load_model(ServerConfig* cfg, const char* id) {
    char dir[160], p[200];
    size_t n;
    if (strcmp(id, MODEL_ID)) snprintf(dir, sizeof(dir), MODELS_DIR "/%s", id);
    else snprintf(dir, sizeof(dir), "%s", ASSETS);
    say("model: %s (%s)", id, dir);
    shown = -1;
    if (load_tokenizer(dir)) return snprintf(eng.error, sizeof(eng.error), "no tokenizer for %s", id), -1;
    say("tokenizer: %u tokens", tok.n_vocab);
    int r = eng.mem ? engine_load(&eng, ASSETS, dir, loading, (void*)tr(S_LOADING_MODEL))
                    : engine_init(&eng, ASSETS, dir, loading, (void*)tr(S_LOADING_MODEL));
    if (r) return -1;
    uint64_t per_block = 0, reserve = 0;
#ifdef GARLIC_SPEC
    int spec = !spec_init(&eng, dir);
    if (spec) reserve = spec_need(&eng, &per_block);
#endif
    load_embedder(cfg);
    // the state snapshot of the server (+ the MTP head's carry, which engine_state_size counts once the head is in)
    uint64_t snap = engine_alloc(&eng, engine_state_size(&eng) + eng.hidden * 4);
    if (engine_ctx(&eng, per_block, reserve)) return -1;
    say("language model ready; GPU memory used %u of %u MiB; context cache %s, %u tokens",
        (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20), eng.tq ? "TurboQuant 4-bit" : "fp16 k + int8 v",
        eng.ctx_limit);
#ifdef GARLIC_SPEC
    if (!spec || spec_start(&eng)) say("spec: no MTP head (mtp.idx, the q4s model or memory missing)");
#endif
    say("GPU memory used: %u of %u MiB", (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20));
    cfg->im_start = eng.im_start, cfg->im_end = eng.im_end, cfg->snap = snap;
    snprintf(cfg->model, sizeof(cfg->model), "%s", id);
    mkdir(MODELS_DIR, 0777);
    int f = open(MODELS_DIR "/default.txt", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (f >= 0) write(f, id, strlen(id)), close(f);
    return 0;
}

// A switch that failed: back to the model before it (or to the built-in one)
static int load_or_back(ServerConfig* cfg, const char* id) {
    char before[64];
    snprintf(before, sizeof(before), "%s", cfg->model);
    if (!load_model(cfg, id)) return 0;
    char why[160];
    snprintf(why, sizeof(why), "%s", eng.error);
    say("model %s: %s", id, why);
    if (load_model(cfg, before) && load_model(cfg, MODEL_ID)) show_error(eng.error);
    snprintf(eng.error, sizeof(eng.error), "%s", why);
    return -1;
}

int main(void) {
    if (rename(LOG_FILE, LOG_PREV)) unlink(LOG_FILE);   // the last log of this app is kept, as LOG_PREV
    say("starting");
    boot_report();
    int lang = 1;
    sceSystemServiceParamGetInt(ORBIS_SYSTEM_SERVICE_PARAM_ID_LANG, &lang);
    ui_lang = lang_from_system(lang);
    int vr = video_init();
    if (vr) {
        say("video init failed: %d", vr);
        for (;;) sceKernelUsleep(1000000);
    }
    if (ui_init(ASSETS)) {
        say("ui assets missing: %s", ui_error);
        for (;;) sceKernelUsleep(1000000);
    }
    sceSystemServiceHideSplashScreen();   // the system shows pic1.png until the app says it is drawing
    loading((void*)tr(S_PREPARING), 0.0f, NULL);
    eng_margin_mb = 64 - ui_screen_pool_mb();   // (the screen block came out of the system's 64 MiB)
    ServerConfig cfg = {.eng = &eng, .tok = &tok, .think = THINK, .think_end = THINK_END, .max_reply = MAX_REPLY,
                        .encode = encode_into, .frame = server_frame, .load = load_or_back};
    cfg.user = &cfg;
    char id[64] = MODEL_ID;
    size_t n;
    char* d = (char*)read_all(MODELS_DIR "/default.txt", &n);   // the model of the last run
    while (d && n && (d[n - 1] == '\n' || d[n - 1] == '\r' || d[n - 1] == ' ')) n--;
    if (d && n && n < sizeof(id)) memcpy(id, d, n), id[n] = 0;
    free(d);
    if (load_model(&cfg, id)) {
        say("model %s: %s", id, eng.error);
        if (!strcmp(id, MODEL_ID) || load_model(&cfg, MODEL_ID)) show_error(eng.error);
    }
    if (server_run(&cfg)) show_error(eng.error);
    for (;;) sceKernelUsleep(1000000);
}
