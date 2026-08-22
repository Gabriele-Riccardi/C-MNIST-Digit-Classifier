#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "net.h"
#include "prng.h"

network *net_create(int input, int hidden, int output) {
    if (input <= 0 || hidden <= 0 || output <= 0) {
        fprintf(stderr, "net: bad geometry %d-%d-%d\n", input, hidden, output);
        return NULL;
    }

    network *n = (network *)calloc(1, sizeof(network));
    if (!n) return NULL;

    n->input  = input;
    n->hidden = hidden;
    n->output = output;

    n->w1 = (real *)malloc((size_t)hidden * (size_t)input  * sizeof(real));
    n->b1 = (real *)calloc((size_t)hidden,                   sizeof(real));
    n->w2 = (real *)malloc((size_t)output * (size_t)hidden * sizeof(real));
    n->b2 = (real *)calloc((size_t)output,                   sizeof(real));
    n->h  = (real *)malloc((size_t)hidden                  * sizeof(real));

    if (!n->w1 || !n->b1 || !n->w2 || !n->b2 || !n->h) {
        fprintf(stderr, "net: out of memory for a %d-%d-%d network\n", input, hidden, output);
        net_free(n);
        return NULL;
    }
    return n;
}

void net_free(network *n) {
    if (!n) return;
    free(n->w1); free(n->b1); free(n->w2); free(n->b2); free(n->h);
    free(n);
}

void net_init_weights(network *n) {
    /* Xavier/Glorot uniform on the fan-in for the hidden layer... */
    const real limit1 = R_SQRT((real)6.0 / (real)n->input);
    for (size_t i = 0; i < (size_t)n->hidden * (size_t)n->input; i++)
        n->w1[i] = ((real)prng_randf() * (real)2.0 - (real)1.0) * limit1;

    /* ...and on fan-in + fan-out for the layer that feeds softmax. */
    const real limit2 = R_SQRT((real)6.0 / (real)(n->hidden + n->output));
    for (size_t i = 0; i < (size_t)n->output * (size_t)n->hidden; i++)
        n->w2[i] = ((real)prng_randf() * (real)2.0 - (real)1.0) * limit2;

    memset(n->b1, 0, (size_t)n->hidden * sizeof(real));
    memset(n->b2, 0, (size_t)n->output * sizeof(real));
}

gradients *grad_create(const network *n) {
    gradients *g = (gradients *)calloc(1, sizeof(gradients));
    if (!g) return NULL;

    g->dw1 = (real *)calloc((size_t)n->hidden * (size_t)n->input,  sizeof(real));
    g->db1 = (real *)calloc((size_t)n->hidden,                     sizeof(real));
    g->dw2 = (real *)calloc((size_t)n->output * (size_t)n->hidden, sizeof(real));
    g->db2 = (real *)calloc((size_t)n->output,                     sizeof(real));

    if (!g->dw1 || !g->db1 || !g->dw2 || !g->db2) {
        grad_free(g);
        return NULL;
    }
    return g;
}

void grad_free(gradients *g) {
    if (!g) return;
    free(g->dw1); free(g->db1); free(g->dw2); free(g->db2);
    free(g);
}

void net_forward(network *n, const real *x, real *out) {
    for (int h = 0; h < n->hidden; h++) {
        const real *w   = &n->w1[(size_t)h * n->input];
        real        sum = n->b1[h];
        for (int i = 0; i < n->input; i++)
            sum += x[i] * w[i];
        n->h[h] = sum > (real)0.0 ? sum : (real)0.0;     /* ReLU */
    }

    for (int o = 0; o < n->output; o++) {
        const real *w   = &n->w2[(size_t)o * n->hidden];
        real        sum = n->b2[o];
        for (int h = 0; h < n->hidden; h++)
            sum += n->h[h] * w[h];
        out[o] = sum;
    }
}

void softmax(real *v, int n) {
    real max = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > max) max = v[i];

    real sum = (real)0.0;
    for (int i = 0; i < n; i++) {
        v[i]  = R_EXP(v[i] - max);
        sum  += v[i];
    }

    for (int i = 0; i < n; i++)
        v[i] /= sum;
}

/*
 * Cross-entropy computed from the logits rather than from the softmax output.
 *
 *   L = logsumexp(z) - z[label]
 *
 * Going through p = softmax(z) and then -log(p) loses precision exactly where
 * it matters most: once the network is confident, p(label) approaches 1 and
 * -log(p) is a small number computed by cancelling two numbers near 1. The
 * error that leaves behind is invisible in training -- the loss is only printed
 * -- but it is the noise floor of any finite-difference check, and it is what
 * puts a hand-rolled gradient check out of reach of a 1e-5 threshold.
 *
 * Shifting by the maximum keeps exp() from overflowing, and accumulating the
 * sum without its largest term lets log1p() carry the small case at full
 * relative precision. It also removes the need to clamp p away from zero, and
 * with it the kink that clamp put in the loss surface.
 */
real cross_entropy_from_logits(const real *logits, int n, int label) {
    int m = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[m]) m = i;

    real rest = (real)0.0;              /* sum over j != m of exp(z_j - z_m) */
    for (int i = 0; i < n; i++)
        if (i != m) rest += R_EXP(logits[i] - logits[m]);

    /* logsumexp(z) - z[label], with z[m] cancelled analytically. */
    return R_LOG1P(rest) - (logits[label] - logits[m]);
}

real net_loss(network *n, const real *x, int label, real *scratch) {
    net_forward(n, x, scratch);
    return cross_entropy_from_logits(scratch, n->output, label);
}

int net_predict(network *n, const real *x, real *scratch) {
    net_forward(n, x, scratch);
    int best = 0;
    for (int o = 1; o < n->output; o++)
        if (scratch[o] > scratch[best]) best = o;
    return best;   /* softmax is monotone, so argmax needs no normalisation */
}

void net_backward(const network *n, const real *x, int label,
                  const real *probs, gradients *g) {
    /* Softmax + cross-entropy collapse: dL/dlogit = p - onehot(label). */
    for (int o = 0; o < n->output; o++)
        g->db2[o] = probs[o];
    g->db2[label] -= (real)1.0;

    /* Hidden delta uses the CURRENT w2. Computing it after a weight update --
       the classic silent bug -- leaves the network training, only worse. */
    for (int h = 0; h < n->hidden; h++)
        g->db1[h] = (real)0.0;

    for (int o = 0; o < n->output; o++) {
        const real  d = g->db2[o];
        const real *w = &n->w2[(size_t)o * n->hidden];
        real       *r = &g->dw2[(size_t)o * n->hidden];
        for (int h = 0; h < n->hidden; h++) {
            g->db1[h] += d * w[h];
            r[h]       = d * n->h[h];      /* dL/dw2 = delta2 (x) hidden */
        }
    }

    /* ReLU derivative: h[i] > 0 exactly when the pre-activation was positive. */
    for (int h = 0; h < n->hidden; h++)
        if (n->h[h] <= (real)0.0) g->db1[h] = (real)0.0;

    /* dL/dw1 = delta1 (x) input. Rows behind a switched-off unit are zeroed
       rather than skipped so every entry of dw1 is defined after the call. */
    for (int h = 0; h < n->hidden; h++) {
        real *r = &g->dw1[(size_t)h * n->input];
        const real d = g->db1[h];
        if (d == (real)0.0) {
            memset(r, 0, (size_t)n->input * sizeof(real));
            continue;
        }
        for (int i = 0; i < n->input; i++)
            r[i] = d * x[i];
    }
}

void net_sgd_step(network *n, const gradients *g, real lr) {
    const size_t n1 = (size_t)n->hidden * (size_t)n->input;
    const size_t n2 = (size_t)n->output * (size_t)n->hidden;

    for (size_t i = 0; i < n1; i++) n->w1[i] -= lr * g->dw1[i];
    for (int    h = 0; h < n->hidden; h++) n->b1[h] -= lr * g->db1[h];
    for (size_t i = 0; i < n2; i++) n->w2[i] -= lr * g->dw2[i];
    for (int    o = 0; o < n->output; o++) n->b2[o] -= lr * g->db2[o];
}

/* ---------------- serialisation ---------------- */

static const char NET_MAGIC[4] = { 'M', 'N', 'N', '1' };

/* Convert to/from the on-disk float32 through a staging buffer. */
static int write_floats(FILE *f, const real *src, size_t count) {
    float *buf = (float *)malloc(count * sizeof(float));
    if (!buf) return -1;
    for (size_t i = 0; i < count; i++) buf[i] = (float)src[i];
    const int ok = (fwrite(buf, sizeof(float), count, f) == count);
    free(buf);
    return ok ? 0 : -1;
}

static int read_floats(FILE *f, real *dst, size_t count) {
    float *buf = (float *)malloc(count * sizeof(float));
    if (!buf) return -1;
    const int ok = (fread(buf, sizeof(float), count, f) == count);
    if (ok) for (size_t i = 0; i < count; i++) dst[i] = (real)buf[i];
    free(buf);
    return ok ? 0 : -1;
}

int net_save(const network *n, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "%s: ", path); perror("fopen"); return -1; }

    const int32_t geom[3] = { (int32_t)n->input, (int32_t)n->hidden, (int32_t)n->output };
    const size_t  n1 = (size_t)n->hidden * (size_t)n->input;
    const size_t  n2 = (size_t)n->output * (size_t)n->hidden;

    if (fwrite(NET_MAGIC, 1, sizeof(NET_MAGIC), f) != sizeof(NET_MAGIC) ||
        fwrite(geom, sizeof(int32_t), 3, f) != 3 ||
        write_floats(f, n->w1, n1)                 != 0 ||
        write_floats(f, n->b1, (size_t)n->hidden)  != 0 ||
        write_floats(f, n->w2, n2)                 != 0 ||
        write_floats(f, n->b2, (size_t)n->output)  != 0) {
        fprintf(stderr, "%s: write failed\n", path);
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

int net_load(network *n, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;      /* a missing file is a normal cold start, not an error */

    char    magic[4];
    int32_t geom[3];

    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        fread(geom, sizeof(int32_t), 3, f) != 3) {
        fprintf(stderr, "%s: truncated header\n", path);
        fclose(f);
        return -1;
    }

    if (memcmp(magic, NET_MAGIC, sizeof(magic)) != 0) {
        fprintf(stderr, "%s: not a network file\n", path);
        fclose(f);
        return -1;
    }

    /* Refuse a mismatched geometry: reading it would run past the arrays. */
    if (geom[0] != n->input || geom[1] != n->hidden || geom[2] != n->output) {
        fprintf(stderr, "%s: shape mismatch -- file %d-%d-%d, network %d-%d-%d\n",
                path, geom[0], geom[1], geom[2], n->input, n->hidden, n->output);
        fclose(f);
        return -1;
    }

    const size_t n1 = (size_t)n->hidden * (size_t)n->input;
    const size_t n2 = (size_t)n->output * (size_t)n->hidden;

    if (read_floats(f, n->w1, n1)                 != 0 ||
        read_floats(f, n->b1, (size_t)n->hidden)  != 0 ||
        read_floats(f, n->w2, n2)                 != 0 ||
        read_floats(f, n->b2, (size_t)n->output)  != 0) {
        fprintf(stderr, "%s: incomplete data\n", path);
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}
