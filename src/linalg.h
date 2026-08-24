#ifndef LINALG_H
#define LINALG_H

/*
 * Dense linear algebra for the post-training quantiser.
 *
 * Everything here is `double`, unconditionally, including in the float32 build.
 * That is not a stylistic preference. The quantiser forms H = X X^T by
 * accumulating a rank-1 update over thousands of calibration samples and then
 * inverts it; a damped 784x784 MNIST Hessian has a condition number in the
 * tens of thousands, and float32 carries about seven decimal digits, so the
 * inverse would be indistinguishable from noise. src/real.h makes the same
 * argument for the gradient check, for the same reason.
 *
 * Matrices are n x n row-major, indexed a[i * n + j]. Vectors are length n.
 * Nothing here is destructive: inputs are const, outputs are separate
 * allocations owned by the caller.
 *
 * These are the textbook O(n^3) formulations -- no blocking, no BLAS. At n <=
 * 784 the whole factorise-invert-refactorise chain runs in well under a second,
 * and the code stays short enough to read line by line against the definition,
 * which is the property the tests need.
 */

#include <stddef.h>

/*
 * Cholesky factorisation of a symmetric positive definite `a` (n x n).
 *
 *   linalg_cholesky_lower  writes L, lower triangular, with L L^T = a
 *   linalg_cholesky_upper  writes U, upper triangular, with U^T U = a
 *
 * The two are transposes of each other. The upper form is the one GPTQ needs:
 * with U^T U = H^-1, row j of U carries exactly the weights that compensate a
 * rounding error at column j (see the derivation in src/quant.c).
 *
 * The strict other triangle of the output is written as zero, so the result can
 * be handed to a general matrix multiply without a mask.
 *
 * Returns 0, or -1 if a pivot came out non-positive -- which means `a` was not
 * positive definite (or is too ill-conditioned to tell the difference).
 */
int linalg_cholesky_lower(const double *a, double *l, int n);
int linalg_cholesky_upper(const double *a, double *u, int n);

/* Triangular solves. `x` and `b` may alias.
 *   lower:    L   x = b,  L lower triangular
 *   lower_t:  L^T x = b,  L lower triangular (avoids materialising L^T)
 *   upper:    U   x = b,  U upper triangular
 */
void linalg_solve_lower  (const double *l, const double *b, double *x, int n);
void linalg_solve_lower_t(const double *l, const double *b, double *x, int n);
void linalg_solve_upper  (const double *u, const double *b, double *x, int n);

/*
 * Inverse of a symmetric positive definite matrix.
 *
 * Goes through the Cholesky factor rather than Gaussian elimination: L is
 * inverted by back-substitution and the result formed as inv = L^-T L^-1, which
 * is symmetric *by construction* rather than symmetric up to rounding. That
 * matters here -- the inverse is immediately handed back to Cholesky, and a
 * factorisation of a matrix whose two triangles disagree in the last few bits
 * silently uses only one of them.
 *
 * Returns 0, or -1 if `a` is not positive definite or the workspace could not
 * be allocated.
 */
int linalg_spd_inverse(const double *a, double *inv, int n);

/* Solves a x = b for symmetric positive definite `a`. Returns 0, or -1. */
int linalg_spd_solve(const double *a, const double *b, double *x, int n);

/* max |a[i] - b[i]|. Used by the tests to report residuals. */
double linalg_max_abs_diff(const double *a, const double *b, size_t count);

/* Frobenius norm of an n x n matrix, and of a - b. */
double linalg_frob(const double *a, size_t count);
double linalg_frob_diff(const double *a, const double *b, size_t count);

#endif /* LINALG_H */
