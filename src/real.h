#ifndef REAL_H
#define REAL_H

/*
 * Precision switch.
 *
 * The network trains in float32 (NET_REAL_DOUBLE undefined). The numerical
 * gradient check compiles the *same* sources with -DNET_REAL_DOUBLE, because a
 * central difference (L(w+e) - L(w-e)) / 2e carries a cancellation error of
 * roughly eps/e: in float32 (eps ~ 1.2e-7) the best achievable relative error
 * is ~1e-3, which cannot resolve a 1e-5 threshold. In float64 (eps ~ 2.2e-16)
 * the same check settles around 1e-10.
 *
 * On-disk weights are always float32, whatever `real` is, so network.dat stays
 * portable between the two builds (see net_save / net_load).
 */

#include <math.h>

#ifdef NET_REAL_DOUBLE
typedef double real;
#define R_EXP(x)     exp(x)
#define R_LOG1P(x)   log1p(x)
#define R_SQRT(x)    sqrt(x)
#define R_FABS(x)    fabs(x)
#define REAL_NAME    "float64"
#else
typedef float real;
#define R_EXP(x)     expf(x)
#define R_LOG1P(x)   log1pf(x)
#define R_SQRT(x)    sqrtf(x)
#define R_FABS(x)    fabsf(x)
#define REAL_NAME    "float32"
#endif

#endif /* REAL_H */
