/*
 * Correctness suite for the post-training quantiser.
 *
 * Same standard as tests/gradcheck.c, and for the same reason. A quantiser that
 * is subtly wrong still produces a network, still classifies digits, and still
 * loses a plausible-looking amount of accuracy as the bit width drops. The
 * error-propagation line in particular -- w[j+1:] -= err * u[j][j+1:] -- has a
 * sign, an index offset and a divisor, and getting any of the three wrong
 * degrades gracefully into round-to-nearest or into something slightly worse
 * than it. None of that is visible in an accuracy table.
 *
 * So the algorithm is checked against independent references rather than
 * against expectations:
 *
 *   C1  the Cholesky factor reconstructs the matrix it factorised
 *   C2  the inverse is an inverse
 *   C3  a quantiser that rounds nothing changes nothing, exactly
 *   C4  the incremental update equals a brute-force least-squares solve
 *   C5  GPTQ does not lose to round-to-nearest on the objective it minimises
 *   C6  16 bits is very nearly transparent
 *   C7  same inputs, byte-identical weight file
 *   C8  the calibration set cannot contain held-out data
 *
 * C4 is the one that matters. It is the analogue of the gradient check: it
 * takes the state in the middle of the sweep, solves the normal equations
 * directly for the columns that are still free, and requires the two to agree
 * to 1e-9. That is what distinguishes "the compensation is optimal" from "the
 * compensation is in the right direction".
 *
 * Hermetic by default -- every fixture is synthesised, including a 784x784
 * Hessian built from images with MNIST's dead border. Pass --data DIR to
 * additionally run C1 and C2 on a Hessian built from the real training images.
 * Nothing here opens the test set.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "idx.h"
#include "linalg.h"
#include "net.h"
#include "prng.h"
#include "quant.h"

static int failures = 0;
static int checks   = 0;

#define CHECK(cond, ...) do {                                      \
    checks++;                                                      \
    if (!(cond)) {                                                 \
        failures++;                                                \
        printf("    FAIL %s:%d: ", __FILE__, __LINE__);            \
        printf(__VA_ARGS__);                                       \
        printf("\n");                                              \
    }                                                              \
} while (0)

static void section(const char *name) { printf("  %s\n", name); }

/* ---------------- fixtures ---------------- */

/* A random symmetric positive definite matrix: B^T B is positive semidefinite,
   and n on the diagonal makes it definite with a condition number that stays
   moderate, so C1 and C2 measure the algorithm and not the fixture. */
static void make_spd(double *a, int n, prng_state *rng) {
    double *b = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    for (int i = 0; i < n * n; i++) b[i] = (double)prng_randf_r(rng) * 2.0 - 1.0;

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) s += b[k * n + i] * b[k * n + j];
            if (i == j) s += (double)n;
            a[i * n + j] = s;
            a[j * n + i] = s;
        }
    free(b);
}

/*
 * Images with MNIST's shape: a `border`-pixel frame that is zero in every
 * sample, and a sparse, non-negative interior. The dead frame is not
 * decoration -- it makes X X^T exactly singular, which is the case the damping
 * term exists for, and a fixture without it would not exercise it.
 */
static void synth_images(dataset *ds, int count, int rows, int cols, int border,
                         unsigned long long seed) {
    memset(ds, 0, sizeof(*ds));
    ds->count  = count;
    ds->rows   = rows;
    ds->cols   = cols;
    ds->pixels = rows * cols;
    ds->images = (real *)calloc((size_t)count * (size_t)ds->pixels, sizeof(real));
    ds->labels = (int  *)calloc((size_t)count, sizeof(int));

    prng_state rng;
    prng_seed_r(&rng, seed, 5ULL);

    for (int s = 0; s < count; s++) {
        real *img = ds->images + (size_t)s * (size_t)ds->pixels;
        for (int r = border; r < rows - border; r++)
            for (int c = border; c < cols - border; c++)
                if (prng_randf_r(&rng) < 0.25f)         /* sparse, like a digit */
                    img[r * cols + c] = (real)prng_randf_r(&rng);
        ds->labels[s] = (int)prng_below_r(&rng, 10u);
    }
}

static double *images_as_double(const dataset *ds, int count) {
    double *x = (double *)malloc((size_t)count * (size_t)ds->pixels * sizeof(double));
    for (int s = 0; s < count; s++)
        for (int p = 0; p < ds->pixels; p++)
            x[(size_t)s * ds->pixels + p] = (double)dataset_image(ds, s)[p];
    return x;
}

static int *identity_index(int count) {
    int *idx = (int *)malloc((size_t)count * sizeof(int));
    for (int i = 0; i < count; i++) idx[i] = i;
    return idx;
}

/* ---------------- C1 / C2 ---------------- */

/* ||U^T U - H||_F / ||H||_F  and  ||H H^-1 - I||_max, for one matrix. */
static void factor_and_invert(const char *name, const double *h, int n,
                              double recon_tol, double inv_tol) {
    double *u    = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *prod = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *hinv = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!u || !prod || !hinv) { printf("    out of memory\n"); failures++; goto done; }

    CHECK(linalg_cholesky_upper(h, u, n) == 0, "%s: Cholesky rejected a positive definite matrix", name);

    /* C1: U^T U must be the matrix that went in. */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k <= (i < j ? i : j); k++)   /* U is upper: U^T[i][k] = U[k][i] */
                s += u[(size_t)k * n + i] * u[(size_t)k * n + j];
            prod[(size_t)i * n + j] = s;
        }

    const size_t n2 = (size_t)n * (size_t)n;
    const double rel = linalg_frob_diff(prod, h, n2) / linalg_frob(h, n2);

    /* C2: and the inverse must be an inverse. */
    CHECK(linalg_spd_inverse(h, hinv, n) == 0, "%s: inversion failed", name);

    double worst = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++)
                s += h[(size_t)i * n + k] * hinv[(size_t)k * n + j];
            const double d = fabs(s - (i == j ? 1.0 : 0.0));
            if (d > worst) worst = d;
        }

    printf("    %-28s n=%-4d  ||U^T U - H||/||H|| %8.2e   ||H H^-1 - I||_max %8.2e\n",
           name, n, rel, worst);

    CHECK(rel   < recon_tol, "%s: reconstruction %.3e exceeds %.1e", name, rel, recon_tol);
    CHECK(worst < inv_tol,   "%s: inverse residual %.3e exceeds %.1e", name, worst, inv_tol);

done:
    free(u); free(prod); free(hinv);
}

static void test_c1_c2_random(void) {
    section("C1/C2  Cholesky reconstructs, and the inverse is an inverse");

    prng_state rng;
    prng_seed_r(&rng, 90210ULL, 1ULL);

    const int sizes[] = { 8, 64 };
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        const int n = sizes[s];
        double *a = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        make_spd(a, n, &rng);
        factor_and_invert("random SPD", a, n, 1e-12, 1e-10);
        free(a);
    }
}

/*
 * The 784x784 case, on a Hessian built exactly the way quant.c builds one: the
 * damped X X^T of 2048 images with a dead border. The tolerances are looser
 * than for the random fixtures and that is the finding, not a concession -- a
 * damped MNIST Hessian has a condition number around 1e5, and the residual of a
 * computed inverse grows with it. The printed number is the measurement.
 */
static void test_c1_c2_hessian(const dataset *ds, int count, const char *label,
                               double recon_tol, double inv_tol) {
    double *x = images_as_double(ds, count);
    double *h = (double *)malloc((size_t)ds->pixels * (size_t)ds->pixels * sizeof(double));
    if (!x || !h) { printf("    out of memory\n"); failures++; free(x); free(h); return; }

    quant_hessian(x, count, ds->pixels, 0.01, h);

    int dead = 0;
    for (int i = 0; i < ds->pixels; i++) {
        double row = 0.0;
        for (int k = 0; k < ds->pixels; k++)
            if (k != i) row += fabs(h[(size_t)i * ds->pixels + k]);
        if (row == 0.0) dead++;
    }
    printf("    %s: %d of %d columns are dead (zero in every calibration image)\n",
           label, dead, ds->pixels);
    CHECK(dead > 0, "%s: a fixture with no dead columns does not exercise the damping", label);

    factor_and_invert(label, h, ds->pixels, recon_tol, inv_tol);

    free(x); free(h);
}

/* ---------------- C3 ---------------- */

static void test_c3_passthrough(void) {
    section("C3  a quantiser that rounds nothing changes nothing, exactly");

    dataset ds;
    synth_images(&ds, 96, 12, 12, 2, 7ULL);

    prng_seed(4242ULL, 1ULL);
    network *n = net_create(ds.pixels, 16, 10);
    net_init_weights(n);

    const size_t n1 = (size_t)n->hidden * (size_t)n->input;
    const size_t n2 = (size_t)n->output * (size_t)n->hidden;
    real *w1_before = (real *)malloc(n1 * sizeof(real));
    real *w2_before = (real *)malloc(n2 * sizeof(real));
    memcpy(w1_before, n->w1, n1 * sizeof(real));
    memcpy(w2_before, n->w2, n2 * sizeof(real));

    int    *idx = identity_index(ds.count);
    double *x1 = NULL, *x2 = NULL;
    CHECK(quant_collect_activations(n, &ds, idx, ds.count, &x1, &x2) == 0, "activation collection");

    const quant_config cfg = { QUANT_GPTQ, QUANT_BITS_PASSTHROUGH, ds.count, 0.01, 1, 1 };
    quant_report rep;
    CHECK(quant_apply(n, &cfg, x1, x2, &rep) == 0, "quant_apply");

    CHECK(memcmp(w1_before, n->w1, n1 * sizeof(real)) == 0,
          "w1 moved under a pass-through grid -- the error propagation has a sign or index bug");
    CHECK(memcmp(w2_before, n->w2, n2 * sizeof(real)) == 0,
          "w2 moved under a pass-through grid");
    CHECK(rep.layer_err_w1 == 0.0, "layer_err_w1 = %g, expected exactly 0", rep.layer_err_w1);
    CHECK(rep.layer_err_w2 == 0.0, "layer_err_w2 = %g, expected exactly 0", rep.layer_err_w2);
    CHECK(rep.max_abs_delta_w1 == 0.0, "max_abs_delta_w1 = %g, expected exactly 0", rep.max_abs_delta_w1);

    printf("    weights bit-identical, layer error exactly 0.0 in both layers\n");

    free(w1_before); free(w2_before); free(idx); free(x1); free(x2);
    net_free(n);
    dataset_free(&ds);
}

/* ---------------- C4 ---------------- */

/*
 * The central check.
 *
 * Stop the sweep after `prefix` columns. Those columns are now fixed at their
 * quantised values; the rest have been moved only by the incremental update.
 * Independently, solve for what the remaining columns *should* be: minimising
 * ||(W_hat - W) X||_F^2 over the free block with the prefix held fixed is an
 * unconstrained quadratic, whose normal equations are
 *
 *     H_FF d_F = -H_FQ d_Q ,      d = W_hat - W
 *
 * solved here by Cholesky, with no reference to the update rule. If the two
 * disagree, the compensation is not the optimal one.
 */
static void c4_case(const char *name, int rows, int cols, int n_samples, int bits,
                    double damping, double tol) {
    prng_state rng;
    prng_seed_r(&rng, 13579ULL, 2ULL);

    double *w     = (double *)malloc((size_t)rows * (size_t)cols * sizeof(double));
    double *w_out = (double *)malloc((size_t)rows * (size_t)cols * sizeof(double));
    double *x     = (double *)malloc((size_t)n_samples * (size_t)cols * sizeof(double));
    double *h     = (double *)malloc((size_t)cols * (size_t)cols * sizeof(double));

    for (int i = 0; i < rows * cols; i++)      w[i] = (double)prng_randf_r(&rng) * 2.0 - 1.0;
    for (int i = 0; i < n_samples * cols; i++) x[i] = (double)prng_randf_r(&rng) * 2.0 - 1.0;

    quant_hessian(x, n_samples, cols, damping, h);

    double worst = 0.0;
    int    worst_prefix = 0;

    for (int prefix = 1; prefix < cols; prefix++) {
        CHECK(quant_gptq_prefix(w, rows, cols, x, n_samples, damping, bits, prefix, w_out) == 0,
              "%s: prefix sweep failed at %d", name, prefix);

        const int nf = cols - prefix;
        double *hff = (double *)malloc((size_t)nf * (size_t)nf * sizeof(double));
        double *rhs = (double *)malloc((size_t)nf * sizeof(double));
        double *sol = (double *)malloc((size_t)nf * sizeof(double));

        for (int a = 0; a < nf; a++)
            for (int b = 0; b < nf; b++)
                hff[(size_t)a * nf + b] = h[(size_t)(prefix + a) * cols + (prefix + b)];

        for (int r = 0; r < rows; r++) {
            const size_t base = (size_t)r * (size_t)cols;

            /* -H_FQ d_Q, over the columns already fixed. */
            for (int a = 0; a < nf; a++) {
                double s = 0.0;
                for (int q = 0; q < prefix; q++)
                    s += h[(size_t)(prefix + a) * cols + q] * (w_out[base + q] - w[base + q]);
                rhs[a] = -s;
            }

            CHECK(linalg_spd_solve(hff, rhs, sol, nf) == 0, "%s: normal equations", name);

            for (int a = 0; a < nf; a++) {
                const double expected = w[base + prefix + a] + sol[a];
                const double d = fabs(expected - w_out[base + prefix + a]);
                if (d > worst) { worst = d; worst_prefix = prefix; }
            }
        }

        free(hff); free(rhs); free(sol);
    }

    printf("    %-34s %d x %d, N=%d, %d bits   worst |incremental - exact| %8.2e  (at column %d)\n",
           name, rows, cols, n_samples, bits, worst, worst_prefix);
    CHECK(worst < tol, "%s: %.3e exceeds %.1e -- the compensation is not the optimal one",
          name, worst, tol);

    free(w); free(w_out); free(x); free(h);
}

static void test_c4_bruteforce(void) {
    section("C4  the incremental update equals a brute-force least-squares solve");
    c4_case("the spec's fixture",       3,  6,  40, 3, 0.01, 1e-9);
    c4_case("wider, more calibration",  4, 16, 200, 4, 0.01, 1e-9);
    c4_case("2 bits, heavy rounding",   5, 12, 120, 2, 0.01, 1e-9);
    c4_case("light damping",            3, 10,  80, 4, 0.001, 1e-9);
}

/* ---------------- shared fixture for C5 / C6 / C7 ---------------- */

typedef struct {
    dataset  ds;
    network *net;
    real    *w1_ref, *w2_ref;
    double  *x1, *x2;
    int     *idx;
    size_t   n1, n2;
} qfixture;

static void fixture_init(qfixture *f, int samples, int side, int hidden, unsigned long long seed) {
    synth_images(&f->ds, samples, side, side, 2, seed);

    prng_seed(seed, 9ULL);
    f->net = net_create(f->ds.pixels, hidden, 10);
    net_init_weights(f->net);

    /* A few SGD steps so the weights have structure. Freshly initialised
       weights are uniform on a symmetric interval, which is the single case a
       min/max grid handles best -- the easiest possible input for the thing
       under test. */
    gradients *g     = grad_create(f->net);
    real      *probs = (real *)malloc((size_t)f->net->output * sizeof(real));
    for (int step = 0; step < 200; step++) {
        const int s = step % f->ds.count;
        const real *img = dataset_image(&f->ds, s);
        net_forward(f->net, img, probs);
        softmax(probs, f->net->output);
        net_backward(f->net, img, f->ds.labels[s], probs, g);
        net_sgd_step(f->net, g, (real)0.05);
    }
    free(probs);
    grad_free(g);

    f->n1 = (size_t)f->net->hidden * (size_t)f->net->input;
    f->n2 = (size_t)f->net->output * (size_t)f->net->hidden;
    f->w1_ref = (real *)malloc(f->n1 * sizeof(real));
    f->w2_ref = (real *)malloc(f->n2 * sizeof(real));
    memcpy(f->w1_ref, f->net->w1, f->n1 * sizeof(real));
    memcpy(f->w2_ref, f->net->w2, f->n2 * sizeof(real));

    f->idx = identity_index(f->ds.count);
    f->x1 = f->x2 = NULL;
    quant_collect_activations(f->net, &f->ds, f->idx, f->ds.count, &f->x1, &f->x2);
}

static void fixture_restore(qfixture *f) {
    memcpy(f->net->w1, f->w1_ref, f->n1 * sizeof(real));
    memcpy(f->net->w2, f->w2_ref, f->n2 * sizeof(real));
}

static void fixture_free(qfixture *f) {
    free(f->w1_ref); free(f->w2_ref); free(f->x1); free(f->x2); free(f->idx);
    net_free(f->net);
    dataset_free(&f->ds);
}

/* ---------------- C5 ---------------- */

static void test_c5_gptq_beats_rtn(void) {
    section("C5  GPTQ does not lose to round-to-nearest on the objective it minimises");

    qfixture f;
    fixture_init(&f, 512, 20, 24, 20260823ULL);

    const int bit_list[] = { 8, 6, 4, 3, 2 };
    for (size_t b = 0; b < sizeof(bit_list) / sizeof(bit_list[0]); b++) {
        quant_config cfg = { QUANT_RTN, bit_list[b], f.ds.count, 0.01, 1, 1 };
        quant_report rtn, gptq;

        fixture_restore(&f);
        CHECK(quant_apply(f.net, &cfg, f.x1, f.x2, &rtn) == 0, "rtn at %d bits", bit_list[b]);

        cfg.method = QUANT_GPTQ;
        fixture_restore(&f);
        CHECK(quant_apply(f.net, &cfg, f.x1, f.x2, &gptq) == 0, "gptq at %d bits", bit_list[b]);

        printf("    %2d bits   w1  rtn %11.4e  gptq %11.4e  (%5.2fx)   "
               "w2  rtn %11.4e  gptq %11.4e  (%5.2fx)\n",
               bit_list[b],
               rtn.layer_err_w1, gptq.layer_err_w1,
               gptq.layer_err_w1 > 0.0 ? rtn.layer_err_w1 / gptq.layer_err_w1 : 0.0,
               rtn.layer_err_w2, gptq.layer_err_w2,
               gptq.layer_err_w2 > 0.0 ? rtn.layer_err_w2 / gptq.layer_err_w2 : 0.0);

        CHECK(gptq.layer_err_w1 <= rtn.layer_err_w1,
              "%d bits: gptq w1 error %.6e is worse than rtn's %.6e",
              bit_list[b], gptq.layer_err_w1, rtn.layer_err_w1);
        CHECK(gptq.layer_err_w2 <= rtn.layer_err_w2,
              "%d bits: gptq w2 error %.6e is worse than rtn's %.6e",
              bit_list[b], gptq.layer_err_w2, rtn.layer_err_w2);
    }

    fixture_free(&f);
}

/* ---------------- C6 ---------------- */

static void test_c6_high_precision(void) {
    section("C6  16 bits is very nearly transparent");

    qfixture f;
    fixture_init(&f, 512, 20, 24, 31337ULL);

    /* Predictions before quantisation, to compare against afterwards. */
    real *scratch = (real *)malloc((size_t)f.net->output * sizeof(real));
    int  *before  = (int *)malloc((size_t)f.ds.count * sizeof(int));
    for (int s = 0; s < f.ds.count; s++)
        before[s] = net_predict(f.net, dataset_image(&f.ds, s), scratch);

    double max_w1 = 0.0, max_w2 = 0.0;
    for (size_t i = 0; i < f.n1; i++) if (fabs((double)f.w1_ref[i]) > max_w1) max_w1 = fabs((double)f.w1_ref[i]);
    for (size_t i = 0; i < f.n2; i++) if (fabs((double)f.w2_ref[i]) > max_w2) max_w2 = fabs((double)f.w2_ref[i]);

    const quant_config cfg = { QUANT_GPTQ, 16, f.ds.count, 0.01, 1, 1 };
    quant_report rep;
    fixture_restore(&f);
    CHECK(quant_apply(f.net, &cfg, f.x1, f.x2, &rep) == 0, "quant_apply at 16 bits");

    const double rel1 = rep.max_abs_delta_w1 / max_w1;
    const double rel2 = rep.max_abs_delta_w2 / max_w2;

    int changed = 0;
    for (int s = 0; s < f.ds.count; s++)
        if (net_predict(f.net, dataset_image(&f.ds, s), scratch) != before[s]) changed++;

    printf("    max|dW| / max|W|   w1 %8.2e   w2 %8.2e     predictions changed: %d of %d\n",
           rel1, rel2, changed, f.ds.count);

    CHECK(rel1 < 1e-3, "w1 moved by %.3e of its range at 16 bits", rel1);
    CHECK(rel2 < 1e-3, "w2 moved by %.3e of its range at 16 bits", rel2);
    /* The test-set half of C6 -- accuracy within one standard deviation of the
       baseline -- is a protocol claim and belongs to scripts/quant_seeds.sh,
       which is the only thing allowed to open the test files. This is the
       hermetic half: at 16 bits the network must classify identically. */
    CHECK(changed == 0, "%d of %d predictions changed at 16 bits", changed, f.ds.count);

    free(scratch); free(before);
    fixture_free(&f);
}

/* ---------------- C7 ---------------- */

static void run_and_save(unsigned long long seed, const quant_config *cfg, const char *path) {
    qfixture f;
    fixture_init(&f, 256, 16, 20, seed);
    quant_report rep;
    quant_apply(f.net, cfg, f.x1, f.x2, &rep);
    net_save(f.net, path);
    fixture_free(&f);
}

static long slurp(const char *path, unsigned char **out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    const long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    *out = (unsigned char *)malloc((size_t)(len > 0 ? len : 1));
    const size_t got = fread(*out, 1, (size_t)len, fp);
    fclose(fp);
    return (got == (size_t)len) ? len : -1;
}

static void test_c7_determinism(void) {
    section("C7  same seed, same calibration set, same bits -> byte-identical weights");

    const quant_config cfg = { QUANT_GPTQ, 4, 256, 0.01, 1, 1 };
    run_and_save(555ULL, &cfg, "quantcheck_a.tmp");
    run_and_save(555ULL, &cfg, "quantcheck_b.tmp");

    unsigned char *a = NULL, *b = NULL;
    const long la = slurp("quantcheck_a.tmp", &a);
    const long lb = slurp("quantcheck_b.tmp", &b);

    CHECK(la > 0 && la == lb, "weight files differ in length: %ld vs %ld", la, lb);
    if (la > 0 && la == lb) {
        const int same = (memcmp(a, b, (size_t)la) == 0);
        CHECK(same, "two identical runs produced different weight files");
        if (same) printf("    %ld bytes, identical\n", la);
    }

    free(a); free(b);
    remove("quantcheck_a.tmp");
    remove("quantcheck_b.tmp");
}

/* ---------------- C8 ---------------- */

static void test_c8_no_contamination(void) {
    section("C8  the calibration set cannot contain held-out data");

    dataset ds;
    synth_images(&ds, 400, 8, 8, 1, 606ULL);

    subset train_set, val_set;
    CHECK(subset_split(&ds, 100, 20260822ULL, &train_set, &val_set) == 0, "split");

    int *idx = (int *)malloc(64 * sizeof(int));
    CHECK(quant_select_calibration(&train_set, &val_set, 64, 11ULL, idx) == 0, "selection");
    CHECK(quant_indices_are_training_only(&train_set, &val_set, idx, 64) == 1,
          "a set drawn from the training split must pass its own check");

    /* Distinct samples: drawing the same image twice would weight it twice in
       X X^T without saying so. */
    int repeats = 0;
    for (int i = 0; i < 64; i++)
        for (int j = i + 1; j < 64; j++)
            if (idx[i] == idx[j]) repeats++;
    CHECK(repeats == 0, "%d repeated calibration samples", repeats);

    /* Now the case the guard exists for. One validation index, swapped in. */
    const int saved = idx[7];
    idx[7] = val_set.index[0];
    CHECK(quant_indices_are_training_only(&train_set, &val_set, idx, 64) == 0,
          "a validation sample in the calibration set must be rejected");
    idx[7] = saved;

    /* And an index that is in no split at all. */
    idx[7] = ds.count + 5;
    CHECK(quant_indices_are_training_only(&train_set, &val_set, idx, 64) == 0,
          "an out-of-range index must be rejected");
    idx[7] = saved;

    printf("    64 calibration indices, all in the 300 training samples, none of the 100 held out\n");
    printf("    (quant_select_calibration calls abort() on a violation; this drives the predicate\n"
           "     behind that abort, which is the part a passing test can observe)\n");

    free(idx);
    subset_free(&train_set);
    subset_free(&val_set);
    dataset_free(&ds);
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    const char *data_dir = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) data_dir = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--data DIR]\n", argv[0]);
            return 2;
        }
    }

    printf("quantcheck  (%s; the Hessian, its inverse and the Cholesky factor are double in both builds)\n",
           REAL_NAME);

    test_c1_c2_random();

    /* The 784x784 case, hermetically. */
    dataset synth;
    synth_images(&synth, 2048, 28, 28, 3, 2024ULL);
    test_c1_c2_hessian(&synth, 2048, "synthetic 28x28 Hessian", 1e-12, 1e-10);
    dataset_free(&synth);

    if (data_dir) {
        char ipath[1024], lpath[1024];
        snprintf(ipath, sizeof(ipath), "%s/train-images.idx3-ubyte", data_dir);
        snprintf(lpath, sizeof(lpath), "%s/train-labels.idx1-ubyte", data_dir);

        dataset real_ds;
        if (dataset_load(&real_ds, ipath, lpath) == 0) {
            test_c1_c2_hessian(&real_ds, 2048, "MNIST training Hessian", 1e-12, 1e-10);
            dataset_free(&real_ds);
        } else {
            printf("    (could not read %s -- skipping the real Hessian)\n", ipath);
        }
    } else {
        printf("    (pass --data dataset to also run C1/C2 on a Hessian from the real images)\n");
    }

    test_c3_passthrough();
    test_c4_bruteforce();
    test_c5_gptq_beats_rtn();
    test_c6_high_precision();
    test_c7_determinism();
    test_c8_no_contamination();

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
