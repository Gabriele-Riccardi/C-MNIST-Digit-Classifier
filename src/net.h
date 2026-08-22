#ifndef NET_H
#define NET_H

/*
 * A fully connected two-layer network: input -> hidden (ReLU) -> output (softmax),
 * trained with cross-entropy.
 *
 * Parameters live in flat arrays, one contiguous allocation per layer, indexed
 * as w1[h * input + i] and w2[o * hidden + h].
 *
 * Gradients are computed into an explicit `gradients` struct rather than being
 * folded into the weight update. That is what makes tests/gradcheck.c possible:
 * the numbers it verifies are the numbers the training loop uses, not a second
 * derivation written for the test.
 */

#include "real.h"

typedef struct {
    int   input, hidden, output;
    real *w1, *b1;      /* hidden x input, hidden */
    real *w2, *b2;      /* output x hidden, output */
    real *h;            /* hidden activations, forward-pass scratch */
} network;

typedef struct {
    real *dw1, *db1;
    real *dw2, *db2;
} gradients;

network *net_create(int input, int hidden, int output);
void     net_free(network *n);

/* Xavier/Glorot uniform, drawn from the global PRNG. Biases start at zero. */
void     net_init_weights(network *n);

gradients *grad_create(const network *n);
void       grad_free(gradients *g);

/*
 * Forward pass. Writes the hidden activations into n->h and the raw logits into
 * `out` (which must hold n->output values).
 */
void  net_forward(network *n, const real *x, real *out);

/* In-place softmax; subtracts the maximum first so large logits cannot overflow. */
void  softmax(real *v, int n);

/*
 * Cross-entropy of softmax(logits) against `label`, evaluated as
 * logsumexp(logits) - logits[label] so that a confident prediction does not
 * lose its low-order bits to cancellation. See the comment in net.c: this is
 * what makes a 1e-5 gradient check reachable at all.
 */
real  cross_entropy_from_logits(const real *logits, int n, int label);

/*
 * Loss for one sample: forward pass, then the loss above. `scratch` must hold
 * n->output values and comes back holding the logits. This is the function the
 * gradient check differentiates numerically, and it shares net_forward with
 * training.
 */
real  net_loss(network *n, const real *x, int label, real *scratch);

/*
 * Gradients of the cross-entropy loss for one sample, given the softmax output
 * of the forward pass that produced n->h.
 *
 * For a single sample the bias gradient of a layer is exactly that layer's
 * delta, so db1/db2 double as the delta buffers -- there is nothing else to
 * store. dw1 is written in full, including the all-zero rows of hidden units
 * the ReLU switched off, so the caller can read any entry unconditionally.
 */
void  net_backward(const network *n, const real *x, int label,
                   const real *probs, gradients *g);

/* theta <- theta - lr * grad */
void  net_sgd_step(network *n, const gradients *g, real lr);

/* argmax over the output layer; runs a forward pass. `scratch` holds n->output. */
int   net_predict(network *n, const real *x, real *scratch);

/*
 * Serialisation. Weights are always stored as float32 regardless of `real`, so
 * a file written by the float build loads into the float64 build and back.
 * Header: "MNN1", then input, hidden, output as int32.
 */
int   net_save(const network *n, const char *path);
int   net_load(network *n, const char *path);

#endif /* NET_H */
