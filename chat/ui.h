// Garlic LLM's screens: loading, error and the API server. No console calls here.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define SCREEN_W 1920
#define SCREEN_H 1080

// ---- provided by the platform (main.c) ----
extern uint32_t* back;                               // the frame being drawn, 0x80RRGGBB
void say(const char* fmt, ...);                      // log
uint8_t* read_all(const char* path, size_t* size);   // whole file, malloc'd

// ---- languages: chosen from the console's system language ----
enum { LANG_ES, LANG_EN, LANG_PT, LANG_RU, N_LANGS };
extern int ui_lang;
int lang_from_system(int system_lang);   // OrbisSystemParamLanguage -> LANG_*

typedef enum {
    S_PREPARING, S_LOADING_MODEL, S_ERROR_TITLE, S_ERROR_HINT,
    S_SERVER_TITLE, S_SERVER_IDLE, S_SERVER_BUSY, S_SERVER_HINT, N_STRINGS
} StrId;
const char* tr(StrId id);

int ui_init(const char* assets_dir);   // fonts, logo, background; 0 = ok
extern const char* ui_error;
// An 8 MB screen buffer from the screen block (direct memory, CPU cached; outside the system RAM), else malloc.
// ui_screen_pool_mb sets the block up (the first call) and says its size, 0 without it.
uint32_t* ui_screen_buf(void);
int ui_screen_pool_mb(void);

void draw_loading(float frac, const char* what);
void draw_error(const char* what);
// the server screen: the base URL and the status (busy 1 while request number served is answered)
void draw_server(const char* ip, int port, int busy, int served, const char* models);   // models: a line, or NULL
