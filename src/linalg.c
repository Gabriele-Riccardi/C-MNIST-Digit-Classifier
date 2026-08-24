#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"

int linalg_cholesky_lower(const double *a, double *l, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = a[(size_t)i * n + j];
            for (int k = 0; k < j; k++)
                s -= l[(size_t)i * n + k] * l[(size_t)j * n + k];

            if (i == j) {
                /* A non-positive pivot is the whole positive-definiteness test:
                   there is no square root to take and no way to continue. The
                   caller's damping term exists to keep this from happening. */
                if (!(s > 0.0)) return -1;
                l[(size_t)i * n + i] = sqrt(s);
            } else {
                l[(size_t)i * n + j] = s / l[(size_t)j * n + j];
            }
        }
        for (int j = i + 1; j < n; j++)
            l[(size_t)i * n + j] = 0.0;
    }
    return 0;
}

int linalg_cholesky_upper(const double *a, double *u, int n) {
    if (linalg_cholesky_lower(a, u, n) != 0) return -1;

    /* U = L^T, in place: swap the two triangles. */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            const double t = u[(size_t)i * n + j];
            u[(size_t)i * n + j] = 0.0;
            u[(size_t)j * n + i] = t;
        }
    return 0;
}

void linalg_solve_lower(const double *l, const double *b, double *x, int n) {
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++)
            s -= l[(size_t)i * n + k] * x[k];
        x[i] = s / l[(size_t)i * n + i];
    }
}

void linalg_solve_lower_t(const double *l, const double *b, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < n; k++)
            s -= l[(size_t)k * n + i] * x[k];   /* (L^T)[i][k] = L[k][i] */
        x[i] = s / l[(size_t)i * n + i];
    }
}

void linalg_solve_upper(const double *u, const double *b, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < n; k++)
            s -= u[(size_t)i * n + k] * x[k];
        x[i] = s / u[(size_t)i * n + i];
    }
}

int linalg_spd_inverse(const double *a, double *inv, int n) {
    double *l = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!l) return -1;

    if (linalg_cholesky_lower(a, l, n) != 0) { free(l); return -1; }

    /* Invert L in place. L^-1 is lower triangular with 1/L[i][i] on the
       diagonal; the strict lower part follows from row i of L L^-1 = I. */
    for (int i = 0; i < n; i++) {
        const double d = l[(size_t)i * n + i];
        l[(size_t)i * n + i] = 1.0 / d;
        for (int j = 0; j < i; j++) {
            double s = 0.0;
            for (int k = j; k < i; k++)
                s += l[(size_t)i * n + k] * l[(size_t)k * n + j];
            l[(size_t)i * n + j] = -s / d;
        }
    }

    /* inv = L^-T L^-1. Only the upper triangle is computed and then mirrored,
       so the two halves are bit-identical rather than merely close. */
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            double s = 0.0;
            for (int k = j; k < n; k++)   /* L^-1[k][i] is zero for k < i <= j */
                s += l[(size_t)k * n + i] * l[(size_t)k * n + j];
            inv[(size_t)i * n + j] = s;
            inv[(size_t)j * n + i] = s;
        }
    }

    free(l);
    return 0;
}

int linalg_spd_solve(const double *a, const double *b, double *x, int n) {
    double *l = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    if (!l || !y) { free(l); free(y); return -1; }

    if (linalg_cholesky_lower(a, l, n) != 0) { free(l); free(y); return -1; }

    linalg_solve_lower  (l, b, y, n);   /* L y = b   */
    linalg_solve_lower_t(l, y, x, n);   /* L^T x = y */

    free(l);
    free(y);
    return 0;
}

double linalg_max_abs_diff(const double *a, const double *b, size_t count) {
    double worst = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double d = fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

double linalg_frob(const double *a, size_t count) {
    double s = 0.0;
    for (size_t i = 0; i < count; i++) s += a[i] * a[i];
    return sqrt(s);
}

double linalg_frob_diff(const double *a, const double *b, size_t count) {
    double s = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);
}
