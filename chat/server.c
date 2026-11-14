// The OpenAI-compatible API server (see server.h): sockets, request assembly, routing and the JSON replies.
// One connection at a time; it is read without blocking, a little per loop turn, so the screen stays live.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <orbis/Net.h>
#include <orbis/NetCtl.h>
#include <orbis/libkernel.h>

#include "server.h"
#include "ui.h"

#if defined(GARLIC_4B)
#define MODEL_ID "qwen3.5-4b"
#else
#define MODEL_ID "qwen3.5-9b"
#endif
#define CREATED 1763000000      // "created" in every reply: fixed for the run
#define HEAD_MAX 16384          // request line + headers
#define IDLE_S 30.0             // a client silent this long is dropped (the next one is waiting)

// ---- the sceNet layer: the only place that knows the console's socket calls ------------------------------------
// The constants are FreeBSD's (the console's network stack); sceNet errors are 0x80410100 | errno.
#define NET_SOL_SOCKET 0xffff
#define NET_SO_REUSEADDR 0x0004
#define NET_SO_NOSIGPIPE 0x0800
#define NET_SO_NBIO 0x1200
#define NET_EAGAIN ((int32_t)0x80410123)

typedef struct {
    uint8_t len, family;
    uint16_t port;
    uint32_t addr;
    char zero[8];
} SockIn;

static double now_s(void) { return (double)sceKernelReadTsc() / sceKernelGetTscFrequency(); }

static void net_nbio(int fd) {
    int on = 1;
    sceNetSetsockopt(fd, NET_SOL_SOCKET, NET_SO_NBIO, &on, sizeof(on));
    sceNetSetsockopt(fd, NET_SOL_SOCKET, NET_SO_NOSIGPIPE, &on, sizeof(on));
}

static int net_listen(int port, char* err, size_t errn) {
    sceNetInit();
    int fd = sceNetSocket("garlic api", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_STREAM, 0);
    if (fd < 0) {
        snprintf(err, errn, "server: socket failed (0x%08X)", (unsigned)fd);
        return -1;
    }
    int on = 1;
    sceNetSetsockopt(fd, NET_SOL_SOCKET, NET_SO_REUSEADDR, &on, sizeof(on));
    SockIn a = {sizeof(SockIn), ORBIS_NET_AF_INET, sceNetHtons((uint16_t)port), 0, {0}};
    int r = sceNetBind(fd, (const OrbisNetSockaddr*)&a, sizeof(a));
    if (r >= 0) r = sceNetListen(fd, 8);
    if (r < 0) {
        snprintf(err, errn, "server: port %d not available (0x%08X)", port, (unsigned)r);
        sceNetSocketClose(fd);
        return -1;
    }
    net_nbio(fd);
    return fd;
}

static void net_ip(char* ip) {
    static int ctl;
    OrbisNetCtlInfo info;
    if (!ctl) ctl = sceNetCtlInit() >= 0 ? 1 : -1;
    memset(&info, 0, sizeof(info));
    if (sceNetCtlGetInfo(ORBIS_NET_CTL_INFO_IP_ADDRESS, &info) >= 0) {
        memcpy(ip, info.ip_address, 15);
        ip[15] = 0;
    }
}

// Sends everything; a full socket buffer is waited out (up to IDLE_S). 0 = sent, -1 = the client is gone.
static int net_send(int fd, const char* p, size_t n) {
    double t0 = now_s();
    while (n) {
        int r = sceNetSend(fd, p, n, 0);
        if (r > 0) {
            p += r, n -= r, t0 = now_s();
            continue;
        }
        if (r != NET_EAGAIN || now_s() - t0 > IDLE_S) return -1;
        sceKernelUsleep(1000);
    }
    return 0;
}

// ---- a growing text buffer and the JSON writer -----------------------------------------------------------------

typedef struct {
    char* p;
    size_t n, cap;
} Buf;

static void buf_put(Buf* b, const char* s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 1024;
        while (cap < b->n + n + 1) cap *= 2;
        char* p = realloc(b->p, cap);
        if (!p) return;
        b->p = p, b->cap = cap;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void buf_puts(Buf* b, const char* s) { buf_put(b, s, strlen(s)); }

static void buf_printf(Buf* b, const char* fmt, ...) {
    char t[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(t, sizeof(t), fmt, ap);
    va_end(ap);
    if (n > 0) buf_put(b, t, n < (int)sizeof(t) ? (size_t)n : sizeof(t) - 1);
}

// s as a JSON string, quotes included; bytes >= 0x80 pass through (UTF-8 is valid JSON)
static void json_str(Buf* b, const char* s, size_t n) {
    buf_put(b, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            char e[2] = {'\\', (char)c};
            buf_put(b, e, 2);
        } else if (c == '\n') buf_put(b, "\\n", 2);
        else if (c == '\r') buf_put(b, "\\r", 2);
        else if (c == '\t') buf_put(b, "\\t", 2);
        else if (c < 0x20) buf_printf(b, "\\u%04x", c);
        else buf_put(b, (const char*)&c, 1);
    }
    buf_put(b, "\"", 1);
}

// ---- replies ---------------------------------------------------------------------------------------------------

static const char* status_text(int code) {
    switch (code) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    default: return "Internal Server Error";
    }
}

static void reply(int fd, int code, const char* type, const char* body, size_t n) {
    char head[256];
    int h = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %u\r\nConnection: close\r\n\r\n", code,
                     status_text(code), type, (unsigned)n);
    if (net_send(fd, head, h) == 0) net_send(fd, body, n);
}

static void reply_json(int fd, int code, const Buf* b) { reply(fd, code, "application/json", b->p ? b->p : "", b->n); }

static void reply_error(int fd, int code, const char* type, const char* message) {
    Buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    json_str(&b, message, strlen(message));
    buf_puts(&b, ",\"type\":");
    json_str(&b, type, strlen(type));
    buf_puts(&b, "}}");
    reply_json(fd, code, &b);
    free(b.p);
    say("server: %d %s", code, message);
}

static void models(int fd) {
    Buf b = {0};
    buf_printf(&b, "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"created\":%d,"
                   "\"owned_by\":\"garlic\"}]}", MODEL_ID, CREATED);
    reply_json(fd, 200, &b);
    free(b.p);
}

// ---- the JSON parser: reads the request in place, keeping only the fields the server uses -----------------------

typedef struct {
    const char* p;
    const char* end;
    int depth;
} Json;

static void j_ws(Json* j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) j->p++;
}

static int j_peek(Json* j) {
    j_ws(j);
    return j->p < j->end ? (unsigned char)*j->p : -1;
}

static int j_eat(Json* j, char c) {
    if (j_peek(j) != c) return 0;
    j->p++;
    return 1;
}

static void put_utf8(Buf* b, uint32_t c) {
    char u[4];
    int n;
    if (c < 0x80) u[0] = (char)c, n = 1;
    else if (c < 0x800) u[0] = (char)(0xC0 | c >> 6), u[1] = (char)(0x80 | (c & 63)), n = 2;
    else if (c < 0x10000)
        u[0] = (char)(0xE0 | c >> 12), u[1] = (char)(0x80 | (c >> 6 & 63)), u[2] = (char)(0x80 | (c & 63)), n = 3;
    else
        u[0] = (char)(0xF0 | c >> 18), u[1] = (char)(0x80 | (c >> 12 & 63)), u[2] = (char)(0x80 | (c >> 6 & 63)),
        u[3] = (char)(0x80 | (c & 63)), n = 4;
    if (b) buf_put(b, u, n);
}

static int j_hex4(Json* j, uint32_t* v) {
    if (j->end - j->p < 4) return 0;
    *v = 0;
    for (int i = 0; i < 4; i++) {
        char c = *j->p++;
        int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return 0;
        *v = *v << 4 | d;
    }
    return 1;
}

// A string, decoded into out (NULL = skipped). 1 = ok.
static int j_str(Json* j, Buf* out) {
    if (!j_eat(j, '"')) return 0;
    while (j->p < j->end) {
        const char* run = j->p;
        while (j->p < j->end && *j->p != '"' && *j->p != '\\' && (unsigned char)*j->p >= 0x20) j->p++;
        if (out && j->p > run) buf_put(out, run, j->p - run);
        if (j->p >= j->end || (unsigned char)*j->p < 0x20) return 0;
        char c = *j->p++;
        if (c == '"') return 1;
        if (j->p >= j->end) return 0;
        c = *j->p++;
        uint32_t u;
        switch (c) {
        case '"': case '\\': case '/': if (out) buf_put(out, &c, 1); break;
        case 'b': put_utf8(out, 8); break;
        case 'f': put_utf8(out, 12); break;
        case 'n': put_utf8(out, 10); break;
        case 'r': put_utf8(out, 13); break;
        case 't': put_utf8(out, 9); break;
        case 'u':
            if (!j_hex4(j, &u)) return 0;
            if (u >= 0xD800 && u < 0xDC00) {   // a surrogate pair, else U+FFFD
                uint32_t lo;
                if (j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                    const char* back = j->p;
                    j->p += 2;
                    if (j_hex4(j, &lo) && lo >= 0xDC00 && lo < 0xE000) u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                    else j->p = back, u = 0xFFFD;
                } else
                    u = 0xFFFD;
            } else if (u >= 0xDC00 && u < 0xE000)
                u = 0xFFFD;
            put_utf8(out, u);
            break;
        default: return 0;
        }
    }
    return 0;
}

static int j_num(Json* j, double* v) {
    j_ws(j);
    const char* s0 = j->p;
    while (j->p < j->end && strchr("+-0123456789.eE", *j->p)) j->p++;
    size_t n = j->p - s0;
    if (!n || n > 63) return 0;
    char t[64];
    memcpy(t, s0, n);
    t[n] = 0;
    char* e;
    *v = strtod(t, &e);
    return *e == 0;
}

static int j_lit(Json* j, const char* w) {
    size_t n = strlen(w);
    j_ws(j);
    if ((size_t)(j->end - j->p) < n || memcmp(j->p, w, n)) return 0;
    j->p += n;
    return 1;
}

static int j_skip(Json* j) {
    int c = j_peek(j);
    if (c == '"') return j_str(j, NULL);
    if (c == '{' || c == '[') {
        if (++j->depth > 64) return 0;
        j->p++;
        char close = c == '{' ? '}' : ']';
        if (!j_eat(j, close)) {
            do {
                if (c == '{' && (!j_str(j, NULL) || !j_eat(j, ':'))) return 0;
                if (!j_skip(j)) return 0;
            } while (j_eat(j, ','));
            if (!j_eat(j, close)) return 0;
        }
        j->depth--;
        return 1;
    }
    if (c == 't') return j_lit(j, "true");
    if (c == 'f') return j_lit(j, "false");
    if (c == 'n') return j_lit(j, "null");
    double v;
    return j_num(j, &v);
}

// Walks an object: for every key calls field(j, key, user), which reads the value (0 = bad). 1 = ok.
static int j_object(Json* j, int (*field)(Json*, const char*, void*), void* user) {
    if (!j_eat(j, '{')) return 0;
    if (j_eat(j, '}')) return 1;
    do {
        Buf key = {0};
        int ok = j_str(j, &key) && j_eat(j, ':') && field(j, key.p ? key.p : "", user);
        free(key.p);
        if (!ok) return 0;
    } while (j_eat(j, ','));
    return j_eat(j, '}');
}

// ---- the request ---------------------------------------------------------------------------------------------------

enum { ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT };
static const char* const role_names[] = {"system", "user", "assistant"};

typedef struct {
    int role;
    Buf text;
} Message;

typedef struct {
    Message* msg;
    int n, cap;
    double temperature;
    long max_tokens, max_completion;   // -1 = absent
    int stream, seed, has_seed;
} Request;

typedef struct {
    Buf type, text;
} Part;

static int part_field(Json* j, const char* key, void* user) {
    Part* p = user;
    if (!strcmp(key, "type")) return j_peek(j) == '"' ? j_str(j, &p->type) : j_skip(j);
    if (!strcmp(key, "text")) return j_peek(j) == '"' ? j_str(j, &p->text) : j_skip(j);
    return j_skip(j);
}

static int content(Json* j, Buf* out) {
    int c = j_peek(j);
    if (c == '"') return j_str(j, out);
    if (c != '[') return j_skip(j);   // null or anything else: no text
    j->p++;
    if (j_eat(j, ']')) return 1;
    do {
        Part p = {{0}, {0}};
        int ok = j_peek(j) == '{' ? j_object(j, part_field, &p) : j_skip(j);
        if (ok && p.type.p && !strcmp(p.type.p, "text") && p.text.n) buf_put(out, p.text.p, p.text.n);
        free(p.type.p), free(p.text.p);
        if (!ok) return 0;
    } while (j_eat(j, ','));
    return j_eat(j, ']');
}

static int message_field(Json* j, const char* key, void* user) {
    Message* m = user;
    if (!strcmp(key, "content")) return content(j, &m->text);
    if (!strcmp(key, "role")) {
        Buf r = {0};
        if (j_peek(j) != '"') return j_skip(j);
        int ok = j_str(j, &r);
        const char* s = r.p ? r.p : "";
        m->role = !strcmp(s, "system") || !strcmp(s, "developer") ? ROLE_SYSTEM : !strcmp(s, "assistant") ? ROLE_ASSISTANT : ROLE_USER;
        free(r.p);
        return ok;
    }
    return j_skip(j);
}

static int messages(Json* j, Request* r) {
    if (!j_eat(j, '[')) return 0;
    if (j_eat(j, ']')) return 1;
    do {
        if (r->n == r->cap) {
            int cap = r->cap ? r->cap * 2 : 16;
            Message* m = realloc(r->msg, cap * sizeof(Message));
            if (!m) return 0;
            r->msg = m, r->cap = cap;
        }
        Message* m = &r->msg[r->n++];
        memset(m, 0, sizeof(*m));
        m->role = ROLE_USER;
        if (!j_object(j, message_field, m)) return 0;
    } while (j_eat(j, ','));
    return j_eat(j, ']');
}

static int request_field(Json* j, const char* key, void* user) {
    Request* r = user;
    double v;
    if (j_peek(j) == 'n') return j_lit(j, "null");   // null = absent
    if (!strcmp(key, "messages")) return messages(j, r);
    if (!strcmp(key, "stream")) {
        if (j_lit(j, "true")) r->stream = 1;
        else if (j_lit(j, "false")) r->stream = 0;
        else return 0;
        return 1;
    }
    if (!strcmp(key, "temperature") || !strcmp(key, "max_tokens") || !strcmp(key, "max_completion_tokens") ||
        !strcmp(key, "seed")) {
        if (!j_num(j, &v)) return 0;
        if (key[0] == 't') r->temperature = v;
        else if (key[0] == 's') r->seed = (int)(long long)v, r->has_seed = 1;
        else if (!strcmp(key, "max_tokens")) r->max_tokens = v < 0 ? 0 : v > 1e9 ? 1000000000L : (long)v;
        else r->max_completion = v < 0 ? 0 : v > 1e9 ? 1000000000L : (long)v;
        return 1;
    }
    return j_skip(j);
}

static void request_free(Request* r) {
    for (int i = 0; i < r->n; i++) free(r->msg[i].text.p);
    free(r->msg);
}

// ---- chat completions ------------------------------------------------------------------------------------------

static int leaving;   // frame asked to leave while a request ran: the reply ends, then server mode
// The ids in the model's cache: the last prompt and every generated id that was fed (not the final stop token,
// sampled but never fed). A prompt that starts with exactly these is fed from where they end (server_run owns it).
static uint32_t* cache;
static int cached;

// The prompt as the chat renders it: every message, then a new assistant turn; returns its length.
static int render(ServerConfig* cfg, const Request* r, uint32_t* ids, int max) {
    int n = 0;
    for (int i = 0; i <= r->n && n < max - 16; i++) {
        int role = i < r->n ? r->msg[i].role : ROLE_ASSISTANT;
        ids[n++] = cfg->im_start;
        n = cfg->encode(ids, n, max - 16, role_names[role]);
        n = cfg->encode(ids, n, max - 16, "\n");
        if (role == ROLE_ASSISTANT) {   // the empty think block: the answer without thinking
            ids[n++] = cfg->think;
            n = cfg->encode(ids, n, max - 16, "\n\n");
            ids[n++] = cfg->think_end;
            n = cfg->encode(ids, n, max - 16, "\n\n");
        }
        if (i == r->n) break;
        if (r->msg[i].text.p) n = cfg->encode(ids, n, max - 16, r->msg[i].text.p);
        ids[n++] = cfg->im_end;
        n = cfg->encode(ids, n, max - 16, "\n");
    }
    return n;
}

typedef struct {
    Buf text;           // plain: the reply so far
    int completion;     // tokens sent as text
    int fd, stream, id;
    uint8_t pend[8];    // stream: the bytes of a UTF-8 character not complete yet
    int npend;
    int gone;           // stream: the client closed the connection
} Out;

// One chunk of the stream: delta is the JSON of the delta object, finish NULL = null; usage only with a finish.
static void chunk(Out* o, const char* delta, size_t dn, const char* finish, int prompt) {
    Buf b = {0};
    buf_printf(&b, "data: {\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion.chunk\",\"created\":%d,"
                   "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":", o->id, CREATED, MODEL_ID);
    buf_put(&b, delta, dn);
    if (finish)
        buf_printf(&b, ",\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                       "\"total_tokens\":%d}}\n\n", finish, prompt, o->completion, prompt + o->completion);
    else
        buf_puts(&b, ",\"finish_reason\":null}]}\n\n");
    if (!o->gone && b.p && net_send(o->fd, b.p, b.n)) o->gone = 1;
    free(b.p);
}

static void emit(ServerConfig* cfg, Out* o, uint32_t token) {
    const uint8_t* bytes;
    int nb = tok_piece(cfg->tok, token, &bytes);
    o->completion++;
    if (!o->stream) {
        if (nb > 0) buf_put(&o->text, (const char*)bytes, nb);
        return;
    }
    Buf text = {0};   // only complete UTF-8 characters go out
    for (int i = 0; i < nb; i++) {
        o->pend[o->npend++] = bytes[i];
        uint8_t c0 = o->pend[0];
        int need = c0 < 0x80 ? 1 : (c0 >> 5) == 6 ? 2 : (c0 >> 4) == 14 ? 3 : (c0 >> 3) == 30 ? 4 : 1;
        if (o->npend >= need || o->npend == 8) {
            buf_put(&text, (const char*)o->pend, o->npend);
            o->npend = 0;
        }
    }
    if (text.n) {
        Buf d = {0};
        buf_puts(&d, "{\"content\":");
        json_str(&d, text.p, text.n);
        buf_puts(&d, "}");
        chunk(o, d.p, d.n, NULL, 0);
        free(d.p);
    }
    free(text.p);
}

static void cache_add(uint32_t t, int pos) {
    if (cached == pos) cache[cached++] = t;
}

// The prompt (only its new tail when the cache holds the rest), then the reply one token per call, several per call
// with the MTP head. 0 = done (*finish set), -1 = engine error (eng->error).
static int generate(ServerConfig* cfg, ServerStatus* st, const uint32_t* ids, int n, int max_new, float temperature,
                    uint32_t seed, Out* o, const char** finish) {
    Engine* e = cfg->eng;
    uint32_t next;
    int from = cached > 0 && cached < n && memcmp(ids, cache, cached * sizeof(uint32_t)) == 0 ? cached : 0;
    if (!from) engine_reset(e);
    say("server: prompt %d tokens, %d of them cached: feeding %d", n, from, n - from);
    cached = 0;   // (valid again once the prompt is in)
    if (engine_run(e, ids + from, n - from, from, temperature, seed++, &next)) return -1;
    memcpy(cache, ids, n * sizeof(uint32_t));
    cached = n;
    int pos = n;
    for (;;) {
        if (next == cfg->im_end || next == e->eos2) {
            *finish = "stop";
            return 0;
        }
        if (o->completion >= max_new || pos >= (int)e->ctx_limit - 1 || leaving || o->gone) {
            *finish = "length";
            return 0;
        }
        emit(cfg, o, next);
        if (cfg->frame(cfg->user, st)) leaving = 1;
        uint32_t t = next;
#ifdef GARLIC_SPEC
        if (spec_active(e)) {   // the MTP head drafts, the model keeps what it would have drawn itself
            uint32_t ex[8];
            int ne = 0;
            if (engine_spec_step(e, t, pos, temperature, seed++, max_new - o->completion + 1, ex, &ne, &next))
                return -1;
            cache_add(t, pos++);
            for (int i = 0; i < ne; i++) {
                if (ex[i] == cfg->im_end || ex[i] == e->eos2) {   // fed already: the cache no longer matches
                    cached = 0;
                    *finish = "stop";
                    return 0;
                }
                emit(cfg, o, ex[i]);
                cache_add(ex[i], pos++);
            }
            continue;
        }
#endif
        if (engine_run(e, &t, 1, pos, temperature, seed++, &next)) return -1;
        cache_add(t, pos++);
    }
}

static void completions(ServerConfig* cfg, ServerStatus* st, int fd, const char* body, size_t len) {
    Engine* e = cfg->eng;
    Request r = {NULL, 0, 0, 0.7, -1, -1, 0, 0, 0};
    Json j = {body, body + len, 0};
    if (!j_object(&j, request_field, &r) || j_peek(&j) != -1) {
        reply_error(fd, 400, "invalid_request_error", "the body is not valid JSON");
        request_free(&r);
        return;
    }
    if (!r.n) {
        reply_error(fd, 400, "invalid_request_error", "messages is missing or empty");
        request_free(&r);
        return;
    }
    int max = (int)e->ctx_limit + 32;   // (the cache has the same size)
    uint32_t* ids = malloc(max * sizeof(uint32_t));
    int n = ids ? render(cfg, &r, ids, max) : max;
    if (n >= (int)e->ctx_limit - 1) {
        reply_error(fd, 400, "invalid_request_error", "the prompt does not fit the context");
        free(ids);
        request_free(&r);
        return;
    }
    float temperature = r.temperature < 0 ? 0.0f : r.temperature > 1.5 ? 1.5f : (float)r.temperature;
    long want = r.max_completion >= 0 ? r.max_completion : r.max_tokens >= 0 ? r.max_tokens : cfg->max_reply;
    int max_new = want < (long)e->ctx_limit - 1 - n ? (int)want : (int)e->ctx_limit - 1 - n;
    uint32_t seed = r.has_seed ? (uint32_t)r.seed : (uint32_t)st->served + 1;
    Out o = {{0}, 0, fd, r.stream, st->served + 1, {0}, 0, 0};
    const char* finish = "stop";
    double t0 = now_s();
    if (r.stream) {
        static const char head[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                                   "Connection: close\r\n\r\n";
        static const char first[] = "{\"role\":\"assistant\",\"content\":\"\"}";
        o.gone = net_send(fd, head, sizeof(head) - 1) != 0;
        chunk(&o, first, sizeof(first) - 1, NULL, 0);
    }
    int failed = generate(cfg, st, ids, n, max_new, temperature, seed, &o, &finish);
    if (failed) say("server: %s", e->error), cached = 0;
    if (r.stream) {
        if (!failed) {
            if (o.npend) {   // a character the reply ended in the middle of
                Buf d = {0};
                buf_puts(&d, "{\"content\":");
                json_str(&d, (const char*)o.pend, o.npend);
                buf_puts(&d, "}");
                chunk(&o, d.p, d.n, NULL, 0);
                free(d.p);
            }
            chunk(&o, "{}", 2, finish, n);
            if (!o.gone) net_send(fd, "data: [DONE]\n\n", 14);
        }
    } else if (failed) {
        reply_error(fd, 500, "server_error", e->error);
    } else {
        Buf b = {0};
        buf_printf(&b, "{\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion\",\"created\":%d,\"model\":\"%s\","
                       "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":", st->served + 1,
                   CREATED, MODEL_ID);
        json_str(&b, o.text.p ? o.text.p : "", o.text.n);
        buf_printf(&b, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                       "\"total_tokens\":%d}}", finish, n, o.completion, n + o.completion);
        reply_json(fd, 200, &b);
        free(b.p);
    }
    if (!failed) {
        double dt = now_s() - t0;
        say("server: %d messages, prompt %d tokens, reply %d tokens (%s%s%s) in %.1f s (%.1f tokens/s)", r.n, n,
            o.completion, finish, r.stream ? ", streamed" : "", o.gone ? ", client gone" : "", dt,
            dt > 0 ? o.completion / dt : 0.0);
    }
    free(o.text.p);
    free(ids);
    request_free(&r);
}

// ---- one connection --------------------------------------------------------------------------------------------

typedef struct {
    int fd;
    char* buf;              // HEAD_MAX + SERVER_MAX_BODY + 1
    size_t len;
    size_t head;            // bytes of request line + headers (with the blank line), 0 = not complete yet
    size_t body;            // Content-Length
    double last;            // when the client last sent something
} Conn;

static void conn_close(Conn* c) {
    if (c->fd >= 0) sceNetSocketClose(c->fd);
    c->fd = -1, c->len = c->head = c->body = 0;
}

// The value of header name (case-insensitive) in the head, or NULL; *n gets its length.
static const char* header(const char* head, size_t hn, const char* name, size_t* n) {
    size_t k = strlen(name);
    const char* p = memchr(head, '\n', hn);
    const char* end = head + hn;
    while (p && p + 1 < end) {
        p++;
        const char* eol = memchr(p, '\n', end - p);
        if (!eol) break;
        if ((size_t)(eol - p) > k && p[k] == ':' && strncasecmp(p, name, k) == 0) {
            const char* v = p + k + 1;
            while (v < eol && (*v == ' ' || *v == '\t')) v++;
            const char* e = eol;
            while (e > v && (e[-1] == '\r' || e[-1] == ' ' || e[-1] == '\t')) e--;
            *n = e - v;
            return v;
        }
        p = eol;
    }
    return NULL;
}

// The head is complete: checks the body's length. 0 = go on reading, -1 = answered with an error.
static int conn_head(Conn* c) {
    size_t n;
    const char* te = header(c->buf, c->head, "Transfer-Encoding", &n);
    if (te) {
        reply_error(c->fd, 400, "invalid_request_error", "chunked bodies are not supported: send Content-Length");
        return -1;
    }
    const char* cl = header(c->buf, c->head, "Content-Length", &n);
    c->body = 0;
    if (cl) {
        char t[24] = {0};
        memcpy(t, cl, n < 23 ? n : 23);
        char* e;
        unsigned long long v = strtoull(t, &e, 10);
        if (e == t || *e || v > SERVER_MAX_BODY) {
            reply_error(c->fd, 400, "invalid_request_error", "the body is too long (at most 2 MiB) or its length is invalid");
            return -1;
        }
        c->body = (size_t)v;
    } else if (strncmp(c->buf, "POST ", 5) == 0) {
        reply_error(c->fd, 400, "invalid_request_error", "a POST needs Content-Length");
        return -1;
    }
    return 0;
}

static void route(ServerConfig* cfg, ServerStatus* st, Conn* c) {
    char* sp1 = memchr(c->buf, ' ', c->head);
    char* sp2 = sp1 ? memchr(sp1 + 1, ' ', c->head - (sp1 + 1 - c->buf)) : NULL;
    if (!sp1 || !sp2) {
        reply_error(c->fd, 400, "invalid_request_error", "bad request line");
        return;
    }
    size_t mlen = sp1 - c->buf, plen = sp2 - (sp1 + 1);
    char* q = memchr(sp1 + 1, '?', plen);
    if (q) plen = q - (sp1 + 1);
    const char* path = sp1 + 1;
    int get = mlen == 3 && memcmp(c->buf, "GET", 3) == 0, post = mlen == 4 && memcmp(c->buf, "POST", 4) == 0;
    say("server: %.*s %.*s (%u bytes)", (int)mlen, c->buf, (int)plen, path, (unsigned)c->body);
    if (plen == 10 && memcmp(path, "/v1/models", 10) == 0) {
        if (get) models(c->fd);
        else reply_error(c->fd, 405, "method_not_allowed", "use GET for /v1/models");
    } else if (plen == 20 && memcmp(path, "/v1/chat/completions", 20) == 0) {
        if (post) completions(cfg, st, c->fd, c->buf + c->head, c->body);
        else reply_error(c->fd, 405, "method_not_allowed", "use POST for /v1/chat/completions");
    } else
        reply_error(c->fd, 404, "invalid_request_error", "unknown path");
}

// One step of the open connection: reads what has arrived; a complete request is answered and the connection closed.
static void conn_step(ServerConfig* cfg, ServerStatus* st, Conn* c) {
    size_t want = c->head ? c->head + c->body : HEAD_MAX;
    if (c->len < want) {
        int r = sceNetRecv(c->fd, c->buf + c->len, want - c->len, 0);
        if (r == NET_EAGAIN) {
            if (now_s() - c->last > IDLE_S) conn_close(c);
            return;
        }
        if (r <= 0) {   // closed by the client, or an error
            conn_close(c);
            return;
        }
        c->len += r, c->last = now_s();
        c->buf[c->len] = 0;
    }
    if (!c->head) {
        char* e = strstr(c->buf, "\r\n\r\n");
        if (!e) {
            if (c->len >= HEAD_MAX) {
                reply_error(c->fd, 400, "invalid_request_error", "headers too long");
                conn_close(c);
            }
            return;
        }
        c->head = e + 4 - c->buf;
        if (conn_head(c)) {
            conn_close(c);
            return;
        }
    }
    if (c->len < c->head + c->body) return;
    st->busy = 1;
    route(cfg, st, c);
    st->served++;
    st->busy = 0;
    conn_close(c);
}

int server_run(ServerConfig* cfg) {
    ServerStatus st = {0, 0, SERVER_PORT, ""};
    Conn c = {-1, NULL, 0, 0, 0, 0};
    engine_reset(cfg->eng);
    c.buf = malloc(HEAD_MAX + SERVER_MAX_BODY + 1);
    if (!c.buf) {
        snprintf(cfg->eng->error, sizeof(cfg->eng->error), "server: no memory for the request buffer");
        return -1;
    }
    int ls = net_listen(SERVER_PORT, cfg->eng->error, sizeof(cfg->eng->error));
    if (ls < 0) {
        free(c.buf);
        return -1;
    }
    leaving = 0, cached = 0;
    cache = malloc((cfg->eng->ctx_limit + 32) * sizeof(uint32_t));
    if (!cache) {
        snprintf(cfg->eng->error, sizeof(cfg->eng->error), "server: no memory for the cache ids");
        sceNetSocketClose(ls);
        free(c.buf);
        return -1;
    }
    net_ip(st.ip);
    say("server: listening on %s:%d", st.ip[0] ? st.ip : "0.0.0.0", SERVER_PORT);
    for (int turn = 1;; turn++) {
        if (!st.ip[0] && turn % 120 == 0) net_ip(st.ip);
        if (c.fd < 0) {
            int fd = sceNetAccept(ls, NULL, NULL);
            if (fd >= 0) {
                net_nbio(fd);
                c.fd = fd, c.len = c.head = c.body = 0, c.last = now_s();
            }
        }
        if (c.fd >= 0) conn_step(cfg, &st, &c);
        if (leaving || cfg->frame(cfg->user, &st)) break;
    }
    conn_close(&c);
    sceNetSocketClose(ls);
    free(c.buf);
    free(cache);
    cache = NULL, cached = 0;
    say("server: stopped after %d requests", st.served);
    return 0;
}
