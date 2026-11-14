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

static void completions(ServerConfig* cfg, ServerStatus* st, int fd, const char* body, size_t n) {
    (void)cfg, (void)st, (void)body, (void)n;
    reply_error(fd, 400, "invalid_request_error", "chat completions are not available yet");
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
        if (cfg->frame(cfg->user, &st)) break;
    }
    conn_close(&c);
    sceNetSocketClose(ls);
    free(c.buf);
    say("server: stopped after %d requests", st.served);
    return 0;
}
