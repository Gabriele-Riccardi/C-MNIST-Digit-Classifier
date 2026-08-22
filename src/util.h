#ifndef UTIL_H
#define UTIL_H

#include <time.h>

/* Wall-clock seconds. clock() measures CPU time, which would report the time the
   load benchmark spends blocked on I/O as free. */
static inline double now_seconds(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

#endif /* UTIL_H */
