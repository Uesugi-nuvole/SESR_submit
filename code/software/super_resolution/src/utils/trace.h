#ifndef SESR_TRACE_H
#define SESR_TRACE_H

#include <stdio.h>

#ifndef SESR_TRACE_OPS
#define SESR_TRACE_OPS 0
#endif

#if SESR_TRACE_OPS
#define SESR_TRACE_OP(backend, op, fmt, ...) \
    do { \
        printf("[TRACE][op] backend=%s op=%s " fmt "\n", backend, op, ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)

#define SESR_TRACE_STAGE(backend, stage, fmt, ...) \
    do { \
        printf("[TRACE][stage] backend=%s stage=%s " fmt "\n", backend, stage, ##__VA_ARGS__); \
        fflush(stdout); \
    } while (0)
#else
#define SESR_TRACE_OP(backend, op, fmt, ...) \
    do { \
    } while (0)

#define SESR_TRACE_STAGE(backend, stage, fmt, ...) \
    do { \
    } while (0)
#endif

#endif // SESR_TRACE_H
