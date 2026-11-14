// Garlic LLM: Qwen3.5 on the PS4's own GPU, served over the LAN as an OpenAI-compatible API (server.c). This file
// is the console side: log, screen, start-up; the screens are drawn by ui.c.
// The model weights live in /data (see model.idx); GPU code, tokenizer and fonts are in the package's assets.

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
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

static int server_frame(void* user, const ServerStatus* st) {
    static int shown = -1;
    static char ip[16];
    (void)user;
    // A frame only when what it shows changes (while answering, present would wait for the TV's refresh, up to 16 ms
    // per token); idle, the server checks for a connection 50 times a second and sleeps in between
    int now_shown = st->busy ? -2 - st->served : st->served;
    if (now_shown != shown || strcmp(ip, st->ip)) {
        shown = now_shown;
        memcpy(ip, st->ip, sizeof(ip));
        draw_server(st->ip, st->port, st->busy, st->busy ? st->served + 1 : st->served);
        present();
    } else if (!st->busy) {
        sceKernelUsleep(20000);
    }
    return 0;   // the server runs until the app is closed
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
    size_t tsize;
    uint8_t* tdata = read_all(ASSETS "/tokenizer.bin", &tsize);
    if (!tdata || tok_load(&tok, tdata, tsize)) show_error("tokenizer.bin");
    say("tokenizer: %u tokens", tok.n_vocab);
    eng_margin_mb = 64 - ui_screen_pool_mb();   // (the screen block came out of the system's 64 MiB)
    if (engine_init(&eng, ASSETS, loading, (void*)tr(S_LOADING_MODEL))) show_error(eng.error);
    uint64_t per_block = 0, reserve = 0;
#ifdef GARLIC_SPEC
    int spec = !spec_init(&eng, ASSETS);
    if (spec) reserve = spec_need(&eng, &per_block);
#endif
    // the state snapshot of the server (+ the MTP head's carry, which engine_state_size counts once the head is in)
    uint64_t snap = engine_alloc(&eng, engine_state_size(&eng) + eng.hidden * 4);
    if (engine_ctx(&eng, per_block, reserve)) show_error(eng.error);
    say("language model ready; GPU memory used %u of %u MiB; context cache %s, %u tokens",
        (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20), eng.tq ? "TurboQuant 4-bit" : "fp16 k + int8 v",
        eng.ctx_limit);
#ifdef GARLIC_SPEC
    if (!spec || spec_start(&eng)) say("spec: no MTP head (mtp.idx, the q4s model or memory missing)");
#endif
    say("GPU memory used: %u of %u MiB", (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20));
    ServerConfig cfg = {&eng, &tok, eng.im_start, eng.im_end, THINK, THINK_END, MAX_REPLY, encode_into, server_frame, NULL, snap};
    if (server_run(&cfg)) show_error(eng.error);
    for (;;) sceKernelUsleep(1000000);
}
