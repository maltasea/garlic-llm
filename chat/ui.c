// Garlic LLM's screens (see ui.h). Software rendering into `back` (1920 x 1080, 0x80RRGGBB): a dark blue theme,
// anti-aliased shapes from signed distances, bitmap fonts from pc/export_chat.py.
#include "ui.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32   // (this file also builds for Windows, which uses malloc instead)
#include <orbis/libkernel.h>
#endif

// ---- theme ---------------------------------------------------------------------------------------------------
#define C_BG_TOP 0x0B1C3A
#define C_BG_BOT 0x040A18
#define C_ACCENT 0x0070D1
#define C_ACCENT_HI 0x3FA9F5
#define C_TEXT_DIM 0xA3B5D2
#define C_TEXT_MUTED 0x6A80A3
#define C_RED 0xFF6B76

int ui_lang = LANG_ES;
const char* ui_error = "";

// ---- strings ---------------------------------------------------------------------------------------------------

static const char* strings[N_STRINGS][N_LANGS] = {

    [S_PREPARING] = {"Preparando", "Getting ready", "Preparando", "Подготовка"},
    [S_LOADING_MODEL] = {"Cargando el modelo", "Loading the model", "Carregando o modelo", "Загрузка модели"},
    [S_ERROR_TITLE] = {"No se pudo iniciar", "Couldn't start", "Não foi possível iniciar", "Не удалось запустить"},
    [S_ERROR_HINT] = {"Pulsa el botón PS para salir.", "Press the PS button to exit.", "Pressione o botão PS para sair.",
                      "Нажмите кнопку PS, чтобы выйти."},
    [S_SERVER_TITLE] = {"Servidor API", "API server", "Servidor API", "API-сервер"},
    [S_SERVER_IDLE] = {"Esperando peticiones…", "Waiting for requests…", "Aguardando pedidos…", "Жду запросов…"},
    [S_SERVER_BUSY] = {"Respondiendo la petición %d…", "Answering request %d…", "Respondendo ao pedido %d…",
                       "Отвечаю на запрос %d…"},
    [S_SERVER_HINT] = {"En tu cliente OpenAI, usa la URL base http://%s:%d/v1",
                       "In your OpenAI client, set the base URL to http://%s:%d/v1",
                       "No seu cliente OpenAI, use a URL base http://%s:%d/v1",
                       "В клиенте OpenAI укажите базовый URL http://%s:%d/v1"},
};

const char* tr(StrId id) { return strings[id][ui_lang] ? strings[id][ui_lang] : strings[id][LANG_EN]; }

int lang_from_system(int l) {
    if (l == 3 || l == 20) return LANG_ES;   // Spanish, Spanish (Latin America)
    if (l == 7 || l == 17) return LANG_PT;   // Portuguese (Portugal, Brazil)
    if (l == 8) return LANG_RU;
    return LANG_EN;
}

// ---- pixels and shapes -------------------------------------------------------------------------------------------

static int clip_x0, clip_y0, clip_x1 = SCREEN_W, clip_y1 = SCREEN_H;
static int text_alpha = 255;   // multiplies text and shapes drawn by the fading toast

static inline void blend(uint32_t* d, uint32_t rgb, uint32_t a) {
    if (a >= 255) {
        *d = 0x80000000 | rgb;
        return;
    }
    uint32_t s = *d, ia = 255 - a;
    uint32_t r = ((s >> 16 & 255) * ia + (rgb >> 16 & 255) * a) * 257 >> 16;
    uint32_t g = ((s >> 8 & 255) * ia + (rgb >> 8 & 255) * a) * 257 >> 16;
    uint32_t b = ((s & 255) * ia + (rgb & 255) * a) * 257 >> 16;
    *d = 0x80000000 | r << 16 | g << 8 | b;
}

static inline float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

// Signed distance from (px, py) to a rounded box (centre, half size, corner radius r); negative inside.
static inline float rbox_sd(float px, float py, float cx, float cy, float hw, float hh, float r) {
    float qx = fabsf(px - cx) - (hw - r), qy = fabsf(py - cy) - (hh - r);
    float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
    float in = qx > qy ? qx : qy;
    if (in > 0) in = 0;
    float out = ox == 0 ? oy : oy == 0 ? ox : sqrtf(ox * ox + oy * oy);
    return out + in - r;
}

// Rounded rectangle, filled (thick 0) or outlined; radii top-left, top-right, bottom-right, bottom-left.
static void rrect4(int x, int y, int w, int h, const int rad[4], int thick, uint32_t rgb, int alpha) {
    if (w <= 0 || h <= 0) return;
    alpha = alpha * text_alpha / 255;
    float cx = x + w * 0.5f, cy = y + h * 0.5f, hw = w * 0.5f, hh = h * 0.5f;
    int rmax = rad[0];
    for (int k = 1; k < 4; k++)
        if (rad[k] > rmax) rmax = rad[k];
    int band = (thick > rmax ? thick : rmax) + 1;
    int y0 = y < clip_y0 ? clip_y0 : y, y1 = y + h > clip_y1 ? clip_y1 : y + h;
    int xa = x < clip_x0 ? clip_x0 : x, xb = x + w > clip_x1 ? clip_x1 : x + w;
    for (int py = y0; py < y1; py++) {
        uint32_t* row = back + py * SCREEN_W;
        float fy = py + 0.5f;
        int top = fy < cy;
        int edge_row = py < y + band || py >= y + h - band;
        for (int px = xa; px < xb; px++) {
            if (!edge_row && px >= x + band && px < x + w - band) {   // the plain middle of the row
                if (thick) {
                    px = x + w - band - 1;
                    continue;
                }
                int stop = x + w - band < xb ? x + w - band : xb;
                for (; px < stop; px++) blend(row + px, rgb, alpha);
                px--;
                continue;
            }
            float fx = px + 0.5f;
            float r = (float)(fx < cx ? (top ? rad[0] : rad[3]) : (top ? rad[1] : rad[2]));
            if (r > hw) r = hw;
            if (r > hh) r = hh;
            float d = rbox_sd(fx, fy, cx, cy, hw, hh, r);
            float cov = clamp01(0.5f - d);
            if (thick) cov -= clamp01(0.5f - (d + thick));
            if (cov > 0.004f) blend(row + px, rgb, (uint32_t)(cov * alpha));
        }
    }
}
static void rrect(int x, int y, int w, int h, int r, uint32_t rgb, int alpha) {
    int rad[4] = {r, r, r, r};
    rrect4(x, y, w, h, rad, 0, rgb, alpha);
}

// Disc (thick 0) or ring of radius r around (cx, cy). Per row, only the pixels near an edge need their distance:
// further out they are not covered, and the inner span is either fully covered (a disc: cov 1) or the ring's hole
// (cov 0). The margins of 0.01 px keep those spans strictly where the per-pixel formula gives exactly 1 or 0, so the
// result is the same as computing every pixel (this keeps the voice screen's big halo discs cheap to draw).
static void circle(float cx, float cy, float r, float thick, uint32_t rgb, int alpha) {
    alpha = alpha * text_alpha / 255;
    int x0 = (int)floorf(cx - r - 1), x1 = (int)ceilf(cx + r + 1), y0 = (int)floorf(cy - r - 1), y1 = (int)ceilf(cy + r + 1);
    if (x0 < clip_x0) x0 = clip_x0;
    if (y0 < clip_y0) y0 = clip_y0;
    if (x1 > clip_x1) x1 = clip_x1;
    if (y1 > clip_y1) y1 = clip_y1;
    const float ro = r + 0.51f, ri = thick > 0 ? r - thick - 0.51f : r - 0.51f;
    for (int py = y0; py < y1; py++) {
        float dy = py + 0.5f - cy, ady = fabsf(dy);
        if (ady >= ro) continue;   // the whole row is outside (d > 0.5: cov 0)
        float ho = sqrtf(ro * ro - dy * dy);
        int o0 = (int)floorf(cx - ho - 0.5f), o1 = (int)ceilf(cx + ho - 0.5f) + 1;   // outside [o0, o1): cov 0
        if (o0 < x0) o0 = x0;
        if (o1 > x1) o1 = x1;
        int a0 = o1, a1 = o1;   // [a0, a1): the inner span
        if (ri > ady) {
            float hi = sqrtf(ri * ri - dy * dy);
            a0 = (int)ceilf(cx - hi - 0.5f), a1 = (int)floorf(cx + hi - 0.5f) + 1;
            if (a0 < o0) a0 = o0;
            if (a1 > o1) a1 = o1;
            if (a1 < a0) a1 = a0;
        }
        uint32_t* row = back + py * SCREEN_W;
        for (int px = o0; px < o1; px++) {
            if (px == a0) {
                if (thick <= 0)
                    for (; px < a1; px++) blend(row + px, rgb, (uint32_t)(1.0f * alpha));
                px = a1;
                if (px >= o1) break;
            }
            float dx = px + 0.5f - cx;
            float d = sqrtf(dx * dx + dy * dy) - r;
            float cov = clamp01(0.5f - d);
            if (thick > 0) cov -= clamp01(0.5f - (d + thick));
            if (cov > 0.004f) blend(row + px, rgb, (uint32_t)(cov * alpha));
        }
    }
}

// ---- background: a dark blue gradient with a soft glow, computed once --------------------------------------------

static uint32_t* bg;

// The four 8 MB screen copies (the frame being drawn, this background, the backdrop of modal screens and the
// keyboard's picture) come from a 32 MiB block of direct memory of their own, CPU cached (WB Onion) like malloc's, so
// they stay out of the system RAM (flexible memory), where the 9B's dictation weights leave ~20 MiB free.
// The engine then leaves 32 MiB less for the system (ui_screen_pool_mb). Without the block: malloc.
static uint8_t* scr_pool;
static int scr_tried, scr_used;
#define SCR_POOL (32u << 20)
#define SCR_ONE (((size_t)SCREEN_W * SCREEN_H * 4 + 0xFFFF) & ~(size_t)0xFFFF)

int ui_screen_pool_mb(void) {
#ifndef _WIN32
    if (!scr_tried) {
        scr_tried = 1;
        off_t off = 0;
        void* m = NULL;
        if (!sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), SCR_POOL, 0x200000, 0 /* WB Onion */, &off)) {
            if (!sceKernelMapDirectMemory(&m, SCR_POOL, 0x03 /* CPU read/write */, 0, off, 0x200000)) scr_pool = m;
            else sceKernelReleaseDirectMemory(off, SCR_POOL);
        }
    }
#endif
    return scr_pool ? (int)(SCR_POOL >> 20) : 0;
}

uint32_t* ui_screen_buf(void) {
    ui_screen_pool_mb();
    if (scr_pool && (size_t)(scr_used + 1) * SCR_ONE <= SCR_POOL) return (uint32_t*)(scr_pool + SCR_ONE * scr_used++);
    return malloc(SCREEN_W * SCREEN_H * 4);
}

static void make_background(void) {
    bg = ui_screen_buf();
    uint32_t seed = 1;
    for (int y = 0; y < SCREEN_H; y++) {
        float t = (float)y / (SCREEN_H - 1);
        float base[3] = {(C_BG_TOP >> 16 & 255) + ((C_BG_BOT >> 16 & 255) - (C_BG_TOP >> 16 & 255)) * t,
                         (C_BG_TOP >> 8 & 255) + ((C_BG_BOT >> 8 & 255) - (C_BG_TOP >> 8 & 255)) * t,
                         (C_BG_TOP & 255) + ((C_BG_BOT & 255) - (C_BG_TOP & 255)) * t};
        for (int x = 0; x < SCREEN_W; x++) {
            float dx = (x - SCREEN_W * 0.5f) / (SCREEN_W * 0.75f), dy = (y + 200.0f) / (SCREEN_H * 1.1f);
            float g = 1.0f - (dx * dx + dy * dy);
            g = g > 0 ? g * g * 0.55f : 0;
            seed = seed * 1664525u + 1013904223u;
            float dither = ((seed >> 24) / 255.0f - 0.5f) * 1.2f;   // breaks up banding in the gradient
            int r = (int)(base[0] + g * 6 + dither), gr = (int)(base[1] + g * 34 + dither), b = (int)(base[2] + g * 80 + dither);
            r = r < 0 ? 0 : r > 255 ? 255 : r, gr = gr < 0 ? 0 : gr > 255 ? 255 : gr, b = b < 0 ? 0 : b > 255 ? 255 : b;
            bg[y * SCREEN_W + x] = 0x80000000 | r << 16 | gr << 8 | b;
        }
    }
}

static void draw_background(void) { memcpy(back, bg, SCREEN_W * SCREEN_H * 4); }

// ---- fonts (PFN2 from pc/export_chat.py) ----------------------------------------------------------------------
// Header "<4sHHII" (magic, line height, ascent, glyph count, bitmap bytes); per glyph "<IHHhhHBBI" (codepoint,
// width, height, left, top, advance, kind 0 = alpha / 1 = RGBA, pad, bitmap offset), sorted by codepoint; then
// the bitmaps; then optionally "SEQS" + count + per sequence "<BBHI" (length, 0, 0, glyph index) and its
// codepoints: emoji made of several characters (flags, keycaps, joined emoji), drawn as one image.
// Emoji images are Twemoji (CC-BY 4.0, credited in assets/CREDITOS.txt).
#define FONT_HEADER 16
#define FONT_REC 20

typedef struct Font {
    uint16_t line, ascent;
    uint32_t n;
    const uint8_t* recs;
    const uint8_t* bitmaps;
    uint32_t* cps;
    int fallback;
    const struct Font* next;   // font to ask when this one lacks a character
    uint32_t n_seq;
    uint32_t* seq_first;
    uint8_t* seq_len;
    uint32_t* seq_glyph;
    const uint8_t** seq_cps;
    int space;                 // advance of ' '
} Font;

typedef struct {
    uint16_t w, h, adv;
    int16_t bx, by;
    uint8_t kind;
    const uint8_t* bm;
} Glyph;

static Font font, font_small, font_big;

static uint32_t utf8_next(const char** s) {
    const uint8_t* p = (const uint8_t*)*s;
    uint32_t c = *p++;
    int extra = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0;
    if (extra) c &= 0x3F >> extra;
    for (int k = 0; k < extra && (*p & 0xC0) == 0x80; k++) c = c << 6 | (*p++ & 0x3F);
    *s = (const char*)p;
    return c;
}

static void glyph_at(const Font* f, int i, Glyph* g) {
    const uint8_t* r = f->recs + i * FONT_REC;
    uint32_t off;
    memcpy(&g->w, r + 4, 2);
    memcpy(&g->h, r + 6, 2);
    memcpy(&g->bx, r + 8, 2);
    memcpy(&g->by, r + 10, 2);
    memcpy(&g->adv, r + 12, 2);
    g->kind = r[14];
    memcpy(&off, r + 16, 4);
    g->bm = f->bitmaps + off;
}

static int find_cp(const Font* f, uint32_t cp) {
    int lo = 0, hi = (int)f->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (f->cps[mid] == cp) return mid;
        if (f->cps[mid] < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int font_load(Font* f, const char* path) {
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < FONT_HEADER || memcmp(d, "PFN2", 4)) return -1;
    memset(f, 0, sizeof(*f));
    uint32_t bitmap_bytes;
    memcpy(&f->line, d + 4, 2);
    memcpy(&f->ascent, d + 6, 2);
    memcpy(&f->n, d + 8, 4);
    memcpy(&bitmap_bytes, d + 12, 4);
    f->recs = d + FONT_HEADER;
    f->bitmaps = d + FONT_HEADER + f->n * FONT_REC;
    f->cps = malloc(f->n * 4);
    for (uint32_t i = 0; i < f->n; i++) {
        memcpy(&f->cps[i], f->recs + i * FONT_REC, 4);
        if (f->cps[i] == 0xFFFD) f->fallback = (int)i;
    }
    const uint8_t* p = f->bitmaps + bitmap_bytes;
    if ((size_t)(p - d) + 8 <= size && !memcmp(p, "SEQS", 4)) {
        memcpy(&f->n_seq, p + 4, 4);
        p += 8;
        f->seq_first = malloc(f->n_seq * 4);
        f->seq_len = malloc(f->n_seq);
        f->seq_glyph = malloc(f->n_seq * 4);
        f->seq_cps = malloc(f->n_seq * sizeof(*f->seq_cps));
        for (uint32_t k = 0; k < f->n_seq; k++) {
            f->seq_len[k] = p[0];
            memcpy(&f->seq_glyph[k], p + 4, 4);
            memcpy(&f->seq_first[k], p + 8, 4);
            f->seq_cps[k] = p + 12;
            p += 8 + 4 * f->seq_len[k];
        }
    }
    int sp = find_cp(f, ' ');
    Glyph g;
    if (sp >= 0) glyph_at(f, sp, &g), f->space = g.adv;
    else f->space = f->line / 4;
    return 0;
}

// Joiners, variation selectors, keycap marks and skin-tone modifiers outside a known sequence draw as nothing.
static int invisible(uint32_t cp) {
    return (cp >= 0x200B && cp <= 0x200F) || (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0x1F3FB && cp <= 0x1F3FF) ||
           (cp >= 0xE0000 && cp <= 0xE007F) || cp == 0x20E3;
}

static void missing(uint32_t cp) {   // each character no font can draw is logged once
    static uint32_t seen[64];
    static int n;
    for (int i = 0; i < n; i++)
        if (seen[i] == cp) return;
    if (n < 64) seen[n++] = cp;
    say("no glyph for U+%04X", (unsigned)cp);
}

static int match_sequence(const Font* f, uint32_t cp, const char* s, const char* end, const char** after) {
    if (!f->n_seq) return -1;
    int lo = 0, hi = (int)f->n_seq;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (f->seq_first[mid] < cp) lo = mid + 1;
        else hi = mid;
    }
    int best = -1, best_len = 0;
    for (int k = lo; k < (int)f->n_seq && f->seq_first[k] == cp; k++) {
        const char* q = s;
        int ok = 1;
        for (int j = 1; j < f->seq_len[k] && ok; j++) {
            uint32_t want;
            memcpy(&want, f->seq_cps[k] + 4 * (j - 1), 4);
            uint32_t got = 0;
            do {
                if (q >= end || !*q) {
                    ok = 0;
                    break;
                }
                got = utf8_next(&q);
            } while (got == 0xFE0F);
            if (ok && got != want) ok = 0;
        }
        if (ok && f->seq_len[k] > best_len) {
            best = k, best_len = f->seq_len[k];
            *after = q;
        }
    }
    return best;
}

static int next_glyph(const Font* f, const char** s, const char* end, Glyph* g) {
    while (*s < end && **s) {
        uint32_t cp = utf8_next(s);
        if (invisible(cp)) continue;
        if (cp == '\t') cp = ' ';
        for (const Font* t = f; t; t = t->next) {
            const char* after;
            int k = match_sequence(t, cp, *s, end, &after);
            if (k >= 0) {
                *s = after;
                glyph_at(t, (int)t->seq_glyph[k], g);
                if (t != f) g->by += (int)f->ascent - (int)t->ascent;   // keep the baseline of the asking font
                return 1;
            }
            int i = find_cp(t, cp);
            if (i >= 0) {
                glyph_at(t, i, g);
                if (t != f) g->by += (int)f->ascent - (int)t->ascent;
                return 1;
            }
        }
        missing(cp);
        glyph_at(f, f->fallback, g);
        return 1;
    }
    return 0;
}

static int text_width(const Font* f, const char* s, const char* end) {
    int w = 0;
    Glyph g;
    while (next_glyph(f, &s, end, &g)) w += g.adv;
    return w;
}
static int str_width(const Font* f, const char* s) { return text_width(f, s, s + strlen(s)); }

// Draws s[0..end) with its top-left at (x, y); returns the x after the text. Emoji keep their own colours.
static int draw_text(const Font* f, int x, int y, const char* s, const char* end, uint32_t rgb) {
    Glyph g;
    while (next_glyph(f, &s, end, &g)) {
        int gx = x + g.bx, gy = y + g.by;
        if (gx + g.w > clip_x0 && gx < clip_x1 && gy + g.h > clip_y0 && gy < clip_y1)
            for (int j = 0; j < g.h; j++) {
                int py = gy + j;
                if (py < clip_y0 || py >= clip_y1) continue;
                uint32_t* row = back + py * SCREEN_W;
                for (int i = 0; i < g.w; i++) {
                    int px = gx + i;
                    if (px < clip_x0 || px >= clip_x1) continue;
                    if (g.kind == 1) {
                        const uint8_t* c = g.bm + (j * g.w + i) * 4;
                        if (c[3]) blend(row + px, (uint32_t)c[0] << 16 | c[1] << 8 | c[2], c[3] * text_alpha / 255);
                    } else {
                        uint32_t a = g.bm[j * g.w + i];
                        if (a) blend(row + px, rgb, a * text_alpha / 255);
                    }
                }
            }
        x += g.adv;
    }
    return x;
}
static int draw_str(const Font* f, int x, int y, const char* s, uint32_t rgb) { return draw_text(f, x, y, s, s + strlen(s), rgb); }
static void draw_center(const Font* f, int cx, int y, const char* s, uint32_t rgb) { draw_str(f, cx - str_width(f, s) / 2, y, s, rgb); }

// ---- images (PIMG from pc/make_brand.py) ----------------------------------------------------------------------

typedef struct {
    int w, h;
    const uint8_t* px;
} Image;
static Image logo_s, logo_l;

static int images_load(const char* path) {
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < 8 || memcmp(d, "PIMG", 4)) return -1;
    uint32_t n;
    memcpy(&n, d + 4, 4);
    Image* out[2] = {&logo_s, &logo_l};
    for (uint32_t i = 0; i < n && i < 2; i++) {
        uint16_t w, h;
        uint32_t off;
        memcpy(&w, d + 8 + i * 8, 2);
        memcpy(&h, d + 10 + i * 8, 2);
        memcpy(&off, d + 12 + i * 8, 4);
        out[i]->w = w, out[i]->h = h, out[i]->px = d + off;
    }
    return 0;
}

static void draw_image(const Image* im, int x, int y, int alpha) {
    if (!im->px) return;
    alpha = alpha * text_alpha / 255;
    for (int j = 0; j < im->h; j++) {
        int py = y + j;
        if (py < clip_y0 || py >= clip_y1) continue;
        for (int i = 0; i < im->w; i++) {
            int px = x + i;
            if (px < clip_x0 || px >= clip_x1) continue;
            const uint8_t* c = im->px + (j * im->w + i) * 4;
            if (c[3]) blend(back + py * SCREEN_W + px, (uint32_t)c[0] << 16 | c[1] << 8 | c[2], c[3] * alpha / 255);
        }
    }
}

int ui_init(const char* dir) {
    static const struct {
        Font* f;
        const char* name;
    } files[] = {{&font, "font.bin"}, {&font_small, "font_small.bin"}, {&font_big, "font_big.bin"}};
    char path[256];
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, files[i].name);
        if (font_load(files[i].f, path)) {
            ui_error = files[i].name;
            return -1;
        }
    }
    font_small.next = font_big.next = &font;
    snprintf(path, sizeof(path), "%s/logo.bin", dir);
    if (images_load(path)) say("logo.bin missing");
    make_background();
    return 0;
}

// ---- screens ---------------------------------------------------------------------------------------------------

static void brand_block(int y) {
    int cx = SCREEN_W / 2;
    circle(cx, y + logo_l.h / 2, 130, 0, C_ACCENT, 30);
    circle(cx, y + logo_l.h / 2, 98, 0, C_ACCENT_HI, 22);
    draw_image(&logo_l, cx - logo_l.w / 2, y, 255);
    y += logo_l.h + 28;
    int w1 = str_width(&font_big, "Garlic"), w2 = str_width(&font_big, "LLM");
    int x = cx - (w1 + font_big.space + w2) / 2;
    x = draw_str(&font_big, x, y, "Garlic", 0xFFFFFF);
    draw_str(&font_big, x + font_big.space, y, "LLM", C_ACCENT_HI);
}

void draw_loading(float frac, const char* what) {
    draw_background();
    brand_block(270);
    int cx = SCREEN_W / 2, y = 600, w = 560;
    rrect(cx - w / 2, y, w, 10, 5, 0x12264A, 255);
    int fw = (int)(w * clamp01(frac));
    if (fw > 0) {
        rrect(cx - w / 2, y, fw < 10 ? 10 : fw, 10, 5, C_ACCENT, 255);
        rrect(cx - w / 2 + (fw > 40 ? fw - 40 : 0), y, fw > 40 ? 40 : fw, 10, 5, C_ACCENT_HI, 200);
    }
    char line_[160];
    snprintf(line_, sizeof(line_), "%s…  %d %%", what, (int)(clamp01(frac) * 100));
    draw_center(&font_small, cx, y + 30, line_, C_TEXT_DIM);
    draw_center(&font_small, cx, SCREEN_H - 90, "Qwen3.5", C_TEXT_MUTED);
}

void draw_error(const char* what) {
    draw_background();
    brand_block(220);
    int cx = SCREEN_W / 2, y = 560;
    draw_center(&font_big, cx, y, tr(S_ERROR_TITLE), C_RED);
    draw_center(&font_small, cx, y + font_big.line + 20, what, C_TEXT_DIM);
    draw_center(&font_small, cx, y + font_big.line + 80, tr(S_ERROR_HINT), C_TEXT_MUTED);
}

void draw_server(const char* ip, int port, int busy, int served, const char* models) {
    if (!ip || !*ip) ip = "0.0.0.0";
    draw_background();
    brand_block(170);
    int cx = SCREEN_W / 2, y = 470;
    char t[160];
    draw_center(&font_small, cx, y, tr(S_SERVER_TITLE), C_ACCENT_HI);
    snprintf(t, sizeof(t), "http://%s:%d/v1", ip, port);
    draw_center(&font_big, cx, y + 60, t, 0xFFFFFF);
    snprintf(t, sizeof(t), tr(busy ? S_SERVER_BUSY : S_SERVER_IDLE), served);
    draw_center(&font_small, cx, y + 60 + font_big.line + 30, t, busy ? C_ACCENT_HI : C_TEXT_DIM);
    snprintf(t, sizeof(t), tr(S_SERVER_HINT), ip, port);
    draw_center(&font_small, cx, y + 60 + font_big.line + 90, t, C_TEXT_MUTED);
    if (models && *models) draw_center(&font_small, cx, y + 60 + font_big.line + 150, models, C_TEXT_DIM);
}
