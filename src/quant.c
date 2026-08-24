#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"
#include "prng.h"
#include "quant.h"

const char *quant_method_name(quant_method m) {
    return (m == QUANT_GPTQ) ? "gptq" : "rtn";
}

/* ---------------- the quantisation grid ---------------- */

/*
 * Asymmetric, per row, uniform:
 *
 *     scale = (max - min) / (2^b - 1),   zero = min
 *     q(w)  = zero + scale * clamp(round((w - zero) / scale), 0, 2^b - 1)
 *
 * Computed once from the row's ORIGINAL weights, before the sweep starts. It
 * has to be: the sweep moves the weights around as it compensates, and a grid
 * that chased them would be fitting itself to its own error. Grouping the grid
 * by blocks of columns is the obvious next refinement and is not done here.
 */
typedef struct {
    double scale;
    double zero;
    int    levels;    /* 2^bits, or 0 for the pass-through grid */
} row_grid;

static row_grid grid_from_row(const double *row, int cols, int bits) {
    row_grid g = { 0.0, 0.0, 0 };

    if (bits <= 0) return g;              /* pass-through; see QUANT_BITS_PASSTHROUGH */

    double lo = row[0], hi = row[0];
    for (int j = 1; j < cols; j++) {
        if (row[j] < lo) lo = row[j];
        if (row[j] > hi) hi = row[j];
    }

    g.levels = 1 << bits;
    g.zero   = lo;
    g.scale  = (hi - lo) / (double)(g.levels - 1);
    return g;
}

static double grid_quantize(const row_grid *g, double w) {
    if (g->levels <= 0)  return w;         /* pass-through */
    if (!(g->scale > 0.0)) return g->zero; /* a constant row encodes to one level */

    double t = round((w - g->zero) / g->scale);
    if (t < 0.0)                       t = 0.0;
    if (t > (double)(g->levels - 1))   t = (double)(g->levels - 1);
    return g->zero + t * g->scale;
}

/* ---------------- Hessian ---------------- */

int quant_hessian(const double *x, int n_samples, int cols,
                  double damping, double *h_out) {
    const size_t n2 = (size_t)cols * (size_t)cols;
    memset(h_out, 0, n2 * sizeof(double));

    /*
     * H = X X^T accumulated one sample at a time. The objective's Hessian with
     * respect to a row of W is 2 X X^T; the factor of two changes neither the
     * minimiser nor the ratio the update below is built from, so it is dropped
     * -- consistently, in the damping and in the error report as well.
     *
     * Only the upper triangle is accumulated and it is mirrored at the end, so
     * H is exactly symmetric rather than symmetric to within rounding. Columns
     * whose activation is zero for this sample contribute nothing and are
     * skipped: on MNIST the border pixels are zero in every image and roughly
     * four fifths of the rest are zero in any given one, which is most of the
     * work in this loop.
     */
    for (int s = 0; s < n_samples; s++) {
        const double *v = x + (size_t)s * (size_t)cols;
        for (int i = 0; i < cols; i++) {
            const double vi = v[i];
            if (vi == 0.0) continue;
            double *hrow = h_out + (size_t)i * (size_t)cols;
            for (int k = i; k < cols; k++)
                hrow[k] += vi * v[k];
        }
    }

    for (int i = 0; i < cols; i++)
        for (int k = i + 1; k < cols; k++)
            h_out[(size_t)k * cols + i] = h_out[(size_t)i * cols + k];

    if (damping > 0.0) {
        double trace = 0.0;
        for (int i = 0; i < cols; i++) trace += h_out[(size_t)i * cols + i];

        /*
         * lambda * mean(diag(H)) on the diagonal. Scaling the damping by the
         * Hessian's own magnitude is what makes lambda = 0.01 mean the same
         * thing for 128 calibration samples and for 2048; an absolute constant
         * would be heavy damping in one case and none in the other.
         *
         * It is also not optional. MNIST's border pixels are zero in every
         * image, so those rows and columns of X X^T are exactly zero and H is
         * singular before damping -- 67 of the 784 columns for the first layer.
         */
        const double mean_diag = trace / (double)cols;
        const double ridge = (mean_diag > 0.0) ? damping * mean_diag : damping;
        for (int i = 0; i < cols; i++)
            h_out[(size_t)i * cols + i] += ridge;
    }

    return 0;
}

/* ---------------- the sweep ---------------- */

/*
 * One pass over the columns of every row.
 *
 *   for each column j:
 *       q   = quantize(w[j])
 *       err = (w[j] - q) / u[j][j]
 *       w[j] = q
 *       w[j+1:] -= err * u[j][j+1:]
 *
 * where u is upper triangular with u^T u = H_damped^-1.
 *
 * Why that last line is the optimal compensation and not a plausible-looking
 * approximation. Fix a row and write the objective in terms of d = w_hat - w:
 * ||dX||^2 = d H d^T. Once columns 0..j-1 are frozen, the free block F =
 * {j..in-1} sits at the minimiser of that quadratic, so around the current w it
 * is (1/2) c^T A c with A = H_FF, the trailing submatrix. Constraining c_j to
 * -e, the Optimal Brain Surgeon solution is
 *
 *       c = -(e / [A^-1]_jj) * (A^-1)_{:,j}
 *
 * and GPTQ's contribution is that this needs no per-column submatrix inverse.
 * Split the indices into Q = {0..j-1} and F, and let M = H^-1 = u^T u with u
 * partitioned to match. The block-inverse identity gives
 *
 *       (H_FF)^-1 = M_FF - M_FQ M_QQ^-1 M_QF
 *                 = u_QF^T u_QF + u_FF^T u_FF - u_QF^T u_QF
 *                 = u_FF^T u_FF
 *
 * and since j is the first index of F and u_FF is upper triangular, the first
 * row of u_FF^T u_FF is u[j][j] * u[j][:]. So
 *
 *       [A^-1]_jk / [A^-1]_jj = u[j][k] / u[j][j]
 *
 * and the update collapses to w[k] -= (e / u[j][j]) * u[j][k]. One Cholesky of
 * one inverse, computed once per layer, serves every column and every row.
 *
 * Check C4 verifies this against a brute-force least-squares solve rather than
 * taking the derivation's word for it.
 *
 * Rows are independent -- they share H and each has its own grid -- so the loop
 * runs row-outer even though the algorithm is stated column-outer. Same
 * arithmetic in the same order per row; W is row-major, so this walks it
 * forwards instead of striding.
 */
static void gptq_sweep(double *w, int rows, int cols,
                       const double *u, const row_grid *grids,
                       int prefix, int propagate) {
    for (int r = 0; r < rows; r++) {
        double        *row = w + (size_t)r * (size_t)cols;
        const row_grid g   = grids[r];

        for (int j = 0; j < prefix; j++) {
            const double q = grid_quantize(&g, row[j]);

            if (!propagate) {                 /* round-to-nearest: no compensation */
                row[j] = q;
                continue;
            }

            const double err = (row[j] - q) / u[(size_t)j * cols + j];
            row[j] = q;

            /* Exactly zero error must leave the tail bit-identical, which
               "subtract err * u" does not guarantee for a weight of -0.0.
               Check C3 is what noticed. */
            if (err == 0.0) continue;

            const double *urow = u + (size_t)j * (size_t)cols;
            for (int k = j + 1; k < cols; k++)
                row[k] -= err * urow[k];
        }
    }
}

/* H_damped -> its inverse -> the upper Cholesky factor of that inverse. */
static int build_factor(const double *x, int n_samples, int cols,
                        double damping, double *u_out) {
    const size_t n2 = (size_t)cols * (size_t)cols;
    double *h    = (double *)malloc(n2 * sizeof(double));
    double *hinv = (double *)malloc(n2 * sizeof(double));
    if (!h || !hinv) { free(h); free(hinv); return -1; }

    int rc = quant_hessian(x, n_samples, cols, damping, h);
    if (rc == 0) rc = linalg_spd_inverse(h, hinv, cols);
    if (rc == 0) rc = linalg_cholesky_upper(hinv, u_out, cols);

    free(h);
    free(hinv);
    return rc;
}

static int build_grids(const double *w, int rows, int cols, int bits, row_grid *grids) {
    for (int r = 0; r < rows; r++)
        grids[r] = grid_from_row(w + (size_t)r * (size_t)cols, cols, bits);
    return 0;
}

int quant_gptq_prefix(const double *w, int rows, int cols,
                      const double *x, int n_samples, double damping,
                      int bits, int prefix, double *w_out) {
    if (prefix < 0 || prefix > cols) return -1;

    const size_t wn = (size_t)rows * (size_t)cols;
    double   *u     = (double *)malloc((size_t)cols * (size_t)cols * sizeof(double));
    row_grid *grids = (row_grid *)malloc((size_t)rows * sizeof(row_grid));
    if (!u || !grids) { free(u); free(grids); return -1; }

    if (build_factor(x, n_samples, cols, damping, u) != 0) {
        free(u); free(grids);
        return -1;
    }

    build_grids(w, rows, cols, bits, grids);
    memcpy(w_out, w, wn * sizeof(double));
    gptq_sweep(w_out, rows, cols, u, grids, prefix, 1);

    free(u);
    free(grids);
    return 0;
}

/* ---------------- one layer, end to end ---------------- */

/*
 * Quantises `w` (rows x cols, `real`) in place and reports
 * ||W X - W_hat X||_F^2 and the largest single weight movement.
 *
 * The error is evaluated as sum_r d_r H d_r^T with the UNDAMPED H, which is
 * algebraically the same number as summing ||W x - W_hat x||^2 over the
 * calibration samples but costs rows*cols^2 instead of rows*cols*N, and is
 * exactly zero when d is exactly zero -- which check C3 relies on.
 */
static int quantize_layer(real *w, int rows, int cols,
                          const double *x, int n_samples,
                          const quant_config *cfg,
                          double *err_out, double *max_delta_out) {
    const size_t wn = (size_t)rows * (size_t)cols;
    const size_t n2 = (size_t)cols * (size_t)cols;

    double    err = 0.0, max_delta = 0.0;
    double   *w_ref  = (double *)malloc(wn * sizeof(double));
    double   *w_work = (double *)malloc(wn * sizeof(double));
    double   *h      = (double *)malloc(n2 * sizeof(double));
    row_grid *grids  = (row_grid *)malloc((size_t)rows * sizeof(row_grid));
    double   *u      = NULL;

    if (!w_ref || !w_work || !h || !grids) goto fail;

    for (size_t i = 0; i < wn; i++) w_ref[i] = (double)w[i];
    memcpy(w_work, w_ref, wn * sizeof(double));

    if (quant_hessian(x, n_samples, cols, 0.0, h) != 0) goto fail;
    build_grids(w_ref, rows, cols, cfg->bits, grids);

    if (cfg->method == QUANT_GPTQ) {
        u = (double *)malloc(n2 * sizeof(double));
        if (!u) goto fail;
        if (build_factor(x, n_samples, cols, cfg->damping, u) != 0) goto fail;
    }

    gptq_sweep(w_work, rows, cols, u, grids, cols, cfg->method == QUANT_GPTQ);

    /* d H d^T, row by row, and the largest movement while we are here. */
    for (int r = 0; r < rows; r++) {
        const size_t base = (size_t)r * (size_t)cols;
        for (int i = 0; i < cols; i++) {
            const double di = w_ref[base + i] - w_work[base + i];
            const double ad = fabs(di);
            if (ad > max_delta) max_delta = ad;
            if (di == 0.0) continue;

            const double *hrow = h + (size_t)i * (size_t)cols;
            double acc = 0.0;
            for (int k = 0; k < cols; k++)
                acc += hrow[k] * (w_ref[base + k] - w_work[base + k]);
            err += di * acc;
        }
    }

    for (size_t i = 0; i < wn; i++) w[i] = (real)w_work[i];

    *err_out       = err;
    *max_delta_out = max_delta;

    free(w_ref); free(w_work); free(h); free(grids); free(u);
    return 0;

fail:
    free(w_ref); free(w_work); free(h); free(grids); free(u);
    return -1;
}

int quant_apply(network *n, const quant_config *cfg,
                const double *x1, const double *x2, quant_report *rep) {
    memset(rep, 0, sizeof(*rep));

    if (cfg->bits != QUANT_BITS_PASSTHROUGH && (cfg->bits < 2 || cfg->bits > 16)) {
        fprintf(stderr, "quant: %d bits is outside the supported 2..16\n", cfg->bits);
        return -1;
    }
    if (cfg->calib_n <= 0) {
        fprintf(stderr, "quant: %d calibration samples is not a calibration set\n", cfg->calib_n);
        return -1;
    }
    if (cfg->damping < 0.0) {
        fprintf(stderr, "quant: damping must not be negative\n");
        return -1;
    }
    if ((cfg->quantize_w1 && !x1) || (cfg->quantize_w2 && !x2)) {
        fprintf(stderr, "quant: asked to quantise a layer with no activations for it\n");
        return -1;
    }

    if (cfg->quantize_w1) {
        if (quantize_layer(n->w1, n->hidden, n->input, x1, cfg->calib_n, cfg,
                           &rep->layer_err_w1, &rep->max_abs_delta_w1) != 0) {
            fprintf(stderr, "quant: w1 failed\n");
            return -1;
        }
    }

    if (cfg->quantize_w2) {
        if (quantize_layer(n->w2, n->output, n->hidden, x2, cfg->calib_n, cfg,
                           &rep->layer_err_w2, &rep->max_abs_delta_w2) != 0) {
            fprintf(stderr, "quant: w2 failed\n");
            return -1;
        }
    }

    return 0;
}

double quant_layer_error(const real *w_ref, const real *w_quant,
                         int rows, int cols, const double *x, int n_samples) {
    double total = 0.0;

    /* Straight from the definition here, sample by sample: this one is used on
       activation sets the calibration never saw, where forming another
       cols x cols Hessian would cost more than the sum it replaces. */
    for (int s = 0; s < n_samples; s++) {
        const double *v = x + (size_t)s * (size_t)cols;
        for (int r = 0; r < rows; r++) {
            const size_t base = (size_t)r * (size_t)cols;
            double acc = 0.0;
            for (int i = 0; i < cols; i++)
                acc += ((double)w_ref[base + i] - (double)w_quant[base + i]) * v[i];
            total += acc * acc;
        }
    }
    return total;
}

size_t quant_packed_bytes(const network *n, const quant_config *cfg) {
    const int bits = (cfg->bits > 0) ? cfg->bits : 32;

    const size_t w1_codes = ((size_t)n->hidden * (size_t)n->input  * (size_t)bits + 7) / 8;
    const size_t w2_codes = ((size_t)n->output * (size_t)n->hidden * (size_t)bits + 7) / 8;

    size_t total = 0;
    total += cfg->quantize_w1 ? w1_codes + (size_t)n->hidden * 2 * sizeof(float)
                              : (size_t)n->hidden * (size_t)n->input * sizeof(float);
    total += cfg->quantize_w2 ? w2_codes + (size_t)n->output * 2 * sizeof(float)
                              : (size_t)n->output * (size_t)n->hidden * sizeof(float);
    total += (size_t)(n->hidden + n->output) * sizeof(float);   /* biases stay float32 */
    return total;
}

/* ---------------- activations ---------------- */

int quant_collect_activations(network *n, const dataset *train,
                              const int *idx, int calib_n,
                              double **x1_out, double **x2_out) {
    if (calib_n <= 0 || n->input != train->pixels) {
        fprintf(stderr, "quant: %d samples of %d pixels do not fit a %d-input network\n",
                calib_n, train->pixels, n->input);
        return -1;
    }

    double *x1      = (double *)malloc((size_t)calib_n * (size_t)n->input  * sizeof(double));
    double *x2      = (double *)malloc((size_t)calib_n * (size_t)n->hidden * sizeof(double));
    real   *scratch = (real   *)malloc((size_t)n->output * sizeof(real));

    if (!x1 || !x2 || !scratch) {
        free(x1); free(x2); free(scratch);
        return -1;
    }

    for (int s = 0; s < calib_n; s++) {
        const int i = idx[s];
        if (i < 0 || i >= train->count) {
            /* A calibration index that is not a sample of the set handed in is
               either a bug or a leak; neither should reach the numbers. */
            fprintf(stderr, "quant: calibration index %d is outside the %d samples given\n",
                    i, train->count);
            abort();
        }

        const real *img = dataset_image(train, i);
        double     *r1  = x1 + (size_t)s * (size_t)n->input;
        for (int p = 0; p < n->input; p++) r1[p] = (double)img[p];

        net_forward(n, img, scratch);
        double *r2 = x2 + (size_t)s * (size_t)n->hidden;
        for (int hcol = 0; hcol < n->hidden; hcol++) r2[hcol] = (double)n->h[hcol];
    }

    free(scratch);
    *x1_out = x1;
    *x2_out = x2;
    return 0;
}

/* ---------------- calibration selection and the contamination guard ---------------- */

int quant_indices_are_training_only(const subset *train_set, const subset *val_set,
                                    const int *idx, int count) {
    const dataset *ds = train_set->ds;
    if (val_set && val_set->ds != ds) return 0;

    /* bit 0: in the training split. bit 1: in the validation split. */
    unsigned char *mark = (unsigned char *)calloc((size_t)ds->count, 1);
    if (!mark) {
        fprintf(stderr, "quant: could not allocate the contamination check\n");
        abort();     /* a check that cannot run has not passed */
    }

    for (int k = 0; k < train_set->count; k++) {
        const int i = train_set->index[k];
        if (i < 0 || i >= ds->count) { free(mark); return 0; }
        mark[i] |= 1u;
    }
    if (val_set) {
        for (int k = 0; k < val_set->count; k++) {
            const int i = val_set->index[k];
            if (i < 0 || i >= ds->count) { free(mark); return 0; }
            mark[i] |= 2u;
        }
    }

    int ok = 1;

    /* The split itself must be a split. If the two arms overlap, every
       downstream guarantee is already gone. */
    for (int i = 0; i < ds->count && ok; i++)
        if (mark[i] == 3u) ok = 0;

    for (int k = 0; k < count && ok; k++) {
        const int i = idx[k];
        if (i < 0 || i >= ds->count) { ok = 0; break; }
        if ((mark[i] & 1u) == 0u)    { ok = 0; break; }   /* not in the training split */
        if ((mark[i] & 2u) != 0u)    { ok = 0; break; }   /* held out for validation */
    }

    free(mark);
    return ok;
}

int quant_select_calibration(const subset *train_set, const subset *val_set,
                             int calib_n, unsigned long long seed, int *idx_out) {
    if (calib_n <= 0 || calib_n > train_set->count) {
        fprintf(stderr, "quant: cannot draw %d calibration samples from %d training samples\n",
                calib_n, train_set->count);
        return -1;
    }

    int *perm = (int *)malloc((size_t)train_set->count * sizeof(int));
    if (!perm) return -1;
    for (int i = 0; i < train_set->count; i++) perm[i] = i;

    /* Private stream, as in subset_split: which samples calibrate must depend
       on `seed` and not on how many draws the training loop happened to make. */
    prng_state rng;
    prng_seed_r(&rng, seed, 3ULL);
    for (int i = 0; i < calib_n; i++) {
        const int j = i + (int)prng_below_r(&rng, (u32)(train_set->count - i));
        const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        idx_out[i] = train_set->index[perm[i]];
    }
    free(perm);

    if (!quant_indices_are_training_only(train_set, val_set, idx_out, calib_n)) {
        fprintf(stderr,
                "quant: the calibration set is not drawn purely from the training split.\n"
                "       Calibrating on held-out data makes every number downstream\n"
                "       meaningless, so this aborts rather than warns.\n");
        abort();
    }

    return 0;
}
