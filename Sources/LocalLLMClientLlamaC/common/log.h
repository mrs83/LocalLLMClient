#include "ggml.h"

#define LOG_WRN(...)
#define LOG_ERR(...)
#define LOG_DBG(...)
#define LOG_INF(...)
#define LOG_TRC(...)
#define LOG_CNT(...)

#define LOG_LEVEL_DEBUG  5
#define LOG_LEVEL_TRACE  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0 // output data from tools

#define LOG_DEFAULT_LLAMA 0

void common_log_set_verbosity_thold(int verbosity);
int  common_log_get_verbosity_thold(void);
extern int common_log_verbosity_thold;

struct common_log;

struct common_log * common_log_init();
struct common_log * common_log_main();
void common_log_add(struct common_log * log, enum ggml_log_level level, const char * fmt, ...);
void common_log_default_callback(enum ggml_log_level level, const char * text, void * user_data);
void common_log_set_prefix(struct common_log * log, bool prefix);
void common_log_set_timestamps(struct common_log * log, bool timestamps);
