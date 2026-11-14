// The OpenAI-compatible API server (server mode, L3): /v1/models and /v1/chat/completions over the console's LAN,
// one request at a time, in the app's main thread.
#pragma once
#include <stdint.h>

#include "engine.h"
#include "tokenizer.h"

#define SERVER_PORT 8080
#define SERVER_MAX_BODY (2u << 20)      // the longest accepted request body; longer is 400

typedef struct {
    int busy;                           // 1 while a request is being answered
    int served;                         // requests answered since server mode began
    int port;                           // SERVER_PORT
    char ip[16];                        // the console's IPv4 address, "" when it cannot be read
} ServerStatus;

typedef struct {
    Engine* eng;                        // the loaded model; its ctx_limit is the one enforced
    const Tokenizer* tok;
    uint32_t im_start, im_end;          // the model's chat-template tokens
    uint32_t think, think_end;          // Qwen3.5's think / end-of-think tokens
    int max_reply;                      // default completion length when the request gives none
    int (*encode)(uint32_t* ids, int n, int max, const char* text);  // appends text's tokens, returns the new count
    int (*frame)(void* user, const ServerStatus* st);                // returns 1 to leave server mode
    void* user;
    uint64_t snap;                      // GPU memory for engine_state_copy, 0 = none: the state at the end of the
                                        // last prompt, for a next prompt that only shares that much with the cache
} ServerConfig;

// Listens on 0.0.0.0:SERVER_PORT until frame asks to leave (0), or -1 when it cannot listen (cfg->eng->error).
// Calls engine_reset before listening; keeps nothing after it returns.
int server_run(ServerConfig* cfg);
