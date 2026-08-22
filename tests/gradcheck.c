/*
 * Numerical gradient check.
 *
 * Backpropagation written by hand fails silently: a wrong sign or a transposed
 * index still trains, only worse, and a good-looking accuracy proves nothing
 * beyond "not catastrophically broken". The only real evidence is to compare
 * the analytic gradient against a central difference of the loss it claims to
 * differentiate:
 *
 *     dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 eps)
 *
 * and require the relative error to be small.
 *
 * Two details make the difference between a check that means something and one
 * that only looks rigorous:
 *
 *   Precision. A central difference costs about eps_machine/h in cancellation.
 *   In float32 (eps ~ 1.2e-7) the floor is around 1e-3, so a float32 check can
 *   never demonstrate 1e-5. This binary is built twice: once as `gradcheck`
 *   with the float32 the network trains in, reported for information, and once
 *   as `gradcheck64` from the same sources with -DNET_REAL_DOUBLE, which is the
 *   build that has to pass.
 *
 *   ReLU kinks. The loss is only piecewise differentiable. If perturbing a
 *   parameter flips a hidden unit across zero, the two sides of the difference
 *   lie on different pieces and the numeric "gradient" is meaningless -- a real
 *   effect that is routinely mistaken for a backprop bug. Every parameter is
 *   therefore checked for a change in the ReLU activation pattern between the
 *   two evaluations and skipped, and counted, if one occurred.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "net.h"
#include "train.h"
#include "prng.h"

#ifdef NET_REAL_DOUBLE
static const real  EPS       = (real)1e-5;
static const real  THRESHOLD = (real)1e-6;   /* the conventional bar is 1e-5 */
#else
static const real  EPS       = (real)1e-2;
static const real  THRESHOLD = (real)5e-2;   /* float32 cancellation floor, informational */
#endif

typedef struct {
    real max_rel;
    real worst_analytic;   /* the gradient at the worst point, so a large */
    real worst_numeric;    /* relative error on a near-zero entry is visible */
    int  checked;
    int  skipped;
} tensor_report;

/* Snapshot of which hidden units the ReLU let through. */
static void relu_pattern(const network *n, unsigned char *out) {
    for (int h = 0; h < n->hidden; h++)
        out[h] = (n->h[h] > (real)0.0) ? 1u : 0u;
}

static real relative_error(real a, real b) {
    const real denom = R_FABS(a) + R_FABS(b);
    if (denom < (real)1e-30) return (real)0.0;      /* both zero: agreement */
    return R_FABS(a - b) / denom;
}

/*
 * Checks `samples` randomly chosen entries of one parameter tensor.
 * `param` is the live tensor, `analytic` the gradient backprop produced for it.
 */
static tensor_report check_tensor(network *n, const real *x, int label,
                                  real *param, const real *analytic, size_t count,
                                  int samples, real *scratch,
                                  unsigned char *pat_plus, unsigned char *pat_minus) {
    tensor_report rep = { (real)0.0, (real)0.0, (real)0.0, 0, 0 };

    for (int s = 0; s < samples; s++) {
        const size_t idx  = (size_t)prng_below((u32)(count > 0xFFFFFFFFu ? 0xFFFFFFFFu : count));
        const real   save = param[idx];

        param[idx] = save + EPS;
        const real loss_plus = net_loss(n, x, label, scratch);
        relu_pattern(n, pat_plus);

        param[idx] = save - EPS;
        const real loss_minus = net_loss(n, x, label, scratch);
        relu_pattern(n, pat_minus);

        param[idx] = save;

        if (memcmp(pat_plus, pat_minus, (size_t)n->hidden) != 0) {
            rep.skipped++;      /* the perturbation crossed a ReLU kink */
            continue;
        }

        const real numeric = (loss_plus - loss_minus) / ((real)2.0 * EPS);
        const real rel     = relative_error(analytic[idx], numeric);

        rep.checked++;
        if (rel > rep.max_rel) {
            rep.max_rel        = rel;
            rep.worst_analytic = analytic[idx];
            rep.worst_numeric  = numeric;
        }
    }

    return rep;
}

/*
 * Builds a fixture and checks it. A network whose ReLU has switched every
 * hidden unit off has all-zero gradients, and then every comparison agrees for
 * the wrong reason -- so the number of live units is printed, and a fixture
 * with none is a failure rather than a very clean pass.
 */
static int check_case(const char *name, int in_nodes, int hid_nodes, int out_nodes,
                      int train_steps) {
    network   *n = net_create(in_nodes, hid_nodes, out_nodes);
    gradients *g = n ? grad_create(n) : NULL;

    real *x       = (real *)malloc((size_t)in_nodes  * sizeof(real));
    real *probs   = (real *)malloc((size_t)out_nodes * sizeof(real));
    real *scratch = (real *)malloc((size_t)out_nodes * sizeof(real));
    unsigned char *pat_plus  = (unsigned char *)malloc((size_t)hid_nodes);
    unsigned char *pat_minus = (unsigned char *)malloc((size_t)hid_nodes);

    if (!n || !g || !x || !probs || !scratch || !pat_plus || !pat_minus) {
        fprintf(stderr, "gradcheck: out of memory\n");
        return 1;
    }

    net_init_weights(n);

    /* Pixel-like input in [0,1]. */
    for (int i = 0; i < in_nodes; i++) x[i] = (real)prng_randf();
    const int label = (int)prng_below((u32)out_nodes);

    /*
     * Optionally take a few SGD steps first. Freshly initialised weights are
     * symmetric and near zero; a bug that only shows once the weights have
     * structure would slip past a check run on the initial point alone.
     */
    for (int st = 0; st < train_steps; st++) {
        net_forward(n, x, probs);
        softmax(probs, n->output);
        net_backward(n, x, label, probs, g);
        net_sgd_step(n, g, (real)0.05);
    }

    net_forward(n, x, probs);

    int active = 0;
    for (int h = 0; h < hid_nodes; h++) if (n->h[h] > (real)0.0) active++;

    softmax(probs, n->output);
    net_backward(n, x, label, probs, g);

    printf("  %s  (%d-%d-%d, p(label)=%.4f, %d warm-up steps, %d/%d hidden units active)\n",
           name, in_nodes, hid_nodes, out_nodes, (double)probs[label],
           train_steps, active, hid_nodes);

    int failed = 0;
    if (active == 0) {
        printf("    every hidden unit is off -- this fixture would verify nothing\n");
        failed = 1;
    }

    const size_t n1 = (size_t)n->hidden * (size_t)n->input;
    const size_t n2 = (size_t)n->output * (size_t)n->hidden;

    const struct { const char *label; real *p; const real *d; size_t count; int samples; } tensors[] = {
        { "w1", n->w1, g->dw1, n1,                 200 },
        { "b1", n->b1, g->db1, (size_t)n->hidden,   64 },
        { "w2", n->w2, g->dw2, n2,                 200 },
        { "b2", n->b2, g->db2, (size_t)n->output,  n->output },
    };

    for (size_t t = 0; t < sizeof(tensors) / sizeof(tensors[0]); t++) {
        const tensor_report r = check_tensor(n, x, label, tensors[t].p, tensors[t].d,
                                             tensors[t].count, tensors[t].samples,
                                             scratch, pat_plus, pat_minus);

        const int ok = (r.checked > 0) && (r.max_rel <= THRESHOLD);
        printf("    %-3s max rel err %8.2e  (analytic %11.4e vs numeric %11.4e)  "
               "%4d checked, %3d at a kink   %s\n",
               tensors[t].label, (double)r.max_rel,
               (double)r.worst_analytic, (double)r.worst_numeric,
               r.checked, r.skipped, ok ? "ok" : "FAIL");

        if (!ok) failed = 1;
        if (r.checked == 0)
            printf("        (nothing checked -- every sampled entry hit a ReLU kink)\n");
    }

    free(x); free(probs); free(scratch); free(pat_plus); free(pat_minus);
    grad_free(g);
    net_free(n);
    return failed;
}

int main(void) {
    prng_seed(424242ULL, 7ULL);

    printf("gradcheck  %s  eps=%g  threshold=%g\n",
           REAL_NAME, (double)EPS, (double)THRESHOLD);

    int failed = 0;

    const struct { const char *name; int in, hid, out; int steps; } cases[] = {
        { "tiny, fresh weights",       12,   7,  4,   0 },
        { "tiny, after 25 SGD steps",  12,   7,  4,  25 },
        { "MNIST geometry, fresh",    784, 128, 10,   0 },
        { "MNIST geometry, trained",  784, 128, 10,  50 },
        { "narrow hidden layer",        9,   3,  3,   5 },
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++)
        failed |= check_case(cases[c].name, cases[c].in, cases[c].hid,
                             cases[c].out, cases[c].steps);

#ifdef NET_REAL_DOUBLE
    printf("\n%s\n", failed ? "gradcheck FAILED" : "gradcheck passed");
    return failed;
#else
    /* The float32 build exists to show the cancellation floor, not to gate CI. */
    printf("\nfloat32 build is informational; gradcheck64 is the one that gates.\n");
    printf("%s\n", failed ? "float32 check outside its (loose) threshold" : "float32 check within its (loose) threshold");
    return failed;
#endif
}
