#ifndef QUANT_H
#define QUANT_H

/*
 * Post-training quantisation with an output-reconstruction objective.
 *
 * For a layer with weights W (out x in) and calibration activations X (in x N),
 * this looks for a low-precision W_hat minimising
 *
 *     || W X  -  W_hat X ||_F^2
 *
 * and NOT || W - W_hat ||. The distinction is the whole point: two weight
 * matrices equally far from the original in parameter space can produce very
 * different layer outputs, and it is the output that the next layer sees.
 *
 * The method follows Optimal Brain Quantisation and GPTQ (Frantar, Ashkboos,
 * Hoefler, Alistarh, arXiv:2210.17323): quantise one column at a time, and
 * after each one, correct the columns that have not been quantised yet so that
 * they absorb the error just introduced. Removing that correction leaves plain
 * round-to-nearest, which is the baseline this compares against.
 *
 * This is not "GPTQ in C". It is per-layer post-training quantisation with an
 * output-reconstruction objective, on a 784-128-10 MLP. The algorithm is the
 * paper's; the setting is four orders of magnitude smaller, and conclusions
 * drawn here do not transfer to a transformer. See the README.
 *
 * Precision: the Hessian, its inverse and the Cholesky factor are double
 * throughout, in both builds (src/linalg.h explains why). The network's weights
 * stay `real`; the sweep runs on a double working copy and is written back at
 * the end, because the correction term accumulates across 784 columns and that
 * accumulation is exactly what float32 would lose.
 */

#include "idx.h"
#include "net.h"

typedef enum { QUANT_RTN = 0, QUANT_GPTQ = 1 } quant_method;

typedef struct {
    quant_method method;
    int          bits;          /* 2..16; 0 selects the pass-through grid (see below) */
    int          calib_n;       /* calibration samples: the N in X (in x N) */
    double       damping;       /* fraction of mean(diag(H)) added to the diagonal, default 0.01 */
    int          quantize_w1;   /* for the per-layer ablation */
    int          quantize_w2;
} quant_config;

/*
 * bits = 0 selects a grid that returns its input unchanged. Nothing outside
 * tests/quantcheck.c should use it: it exists so that check C3 can assert that
 * a quantiser which rounds nothing leaves W bit-identical and the layer error
 * exactly zero, which is what pins down the sign and the indexing of the error
 * propagation.
 */
#define QUANT_BITS_PASSTHROUGH 0

typedef struct {
    double layer_err_w1, layer_err_w2;       /* ||WX - W_hat X||_F^2 on the calibration set */
    double max_abs_delta_w1, max_abs_delta_w2;
} quant_report;

const char *quant_method_name(quant_method m);

/*
 * Runs the calibration samples through the network and hands back the input
 * activations of both layers: x1 is the images (calib_n x n->input), x2 is the
 * post-ReLU hidden layer (calib_n x n->hidden), both sample-major and both
 * allocated here for the caller to free.
 *
 * Sample-major rather than the (in x N) of the maths, because H is accumulated
 * as a rank-1 update per sample and that wants each sample contiguous.
 *
 * Both are collected from the unquantised network in a single pass. For a deep
 * model GPTQ re-collects each layer's inputs from the already-quantised layers
 * below, so that the errors do not compound; with two layers the difference is
 * small, and taking one pass keeps the API honest about what it does. The
 * README lists this as a stated limitation rather than hiding it here.
 *
 * Indices used to CALIBRATE must come from the training split, and
 * quant_select_calibration below turns that "must" into an abort. This
 * function itself collects activations for whatever indices it is handed:
 * tools/quant_sweep.c also uses it on validation indices, to score the same
 * objective on samples the calibration never saw, which is a measurement and
 * not a fit. What neither path can do is reach the test set -- the only dataset
 * either is given is the one loaded from the training files.
 *
 * Out-of-range indices abort here.
 *
 * Returns 0, or -1 on allocation failure (nothing is allocated on failure).
 */
int quant_collect_activations(network *n, const dataset *train,
                              const int *idx, int calib_n,
                              double **x1_out, double **x2_out);

/*
 * Quantises the network in place. x1 and x2 are what
 * quant_collect_activations produced, cfg->calib_n rows each.
 *
 * A layer the config leaves out is untouched and its report entries are zero.
 * Biases are never quantised: 138 of the 101,770 parameters, and no part of
 * the objective describes them.
 *
 * Returns 0, or -1 if the configuration is out of range, an allocation failed,
 * or a damped Hessian came out indefinite.
 */
int quant_apply(network *n, const quant_config *cfg,
                const double *x1, const double *x2, quant_report *rep);

/*
 * || W_ref X - W_quant X ||_F^2 for an arbitrary activation matrix (n_samples x
 * cols, sample-major). quant_apply reports this on the calibration set; this
 * lets the driver measure the same objective on activations the calibration
 * never saw, which is the only way to tell a well-estimated X X^T from an
 * overfitted one.
 */
double quant_layer_error(const real *w_ref, const real *w_quant,
                         int rows, int cols, const double *x, int n_samples);

/*
 * Bytes the quantised parameters would occupy if they were actually packed:
 * out*in*bits/8 for the codes, one float32 scale and zero point per row, and
 * the biases in float32. The weight file this project writes is still float32
 * -- the quantised values are stored dequantised -- so this is the size the
 * grid implies, not the size of anything on disk, and the README says so.
 */
size_t quant_packed_bytes(const network *n, const quant_config *cfg);

/* ---------------- calibration set selection (check C8) ---------------- */

/*
 * Draws `calib_n` distinct samples from the training split, using a private
 * PRNG stream so the choice depends on `seed` and on nothing else.
 *
 * Then it checks the result: every drawn index must be a member of train_set
 * and must not be a member of val_set, and the two subsets must themselves be
 * disjoint. A violation calls abort(). Not a warning, not a return code -- a
 * calibration set contaminated with held-out data makes every number downstream
 * meaningless, and a run that produces meaningless numbers should not finish.
 *
 * Returns 0, or -1 if calib_n does not fit in the training split or a
 * workspace could not be allocated.
 */
int quant_select_calibration(const subset *train_set, const subset *val_set,
                             int calib_n, unsigned long long seed, int *idx_out);

/*
 * The predicate behind that abort, exposed so tests/quantcheck.c can drive it
 * with a deliberately contaminated list and watch it say no. Returns 1 if every
 * index is training-only, 0 otherwise.
 */
int quant_indices_are_training_only(const subset *train_set, const subset *val_set,
                                    const int *idx, int count);

/* ---------------- exposed for tests/quantcheck.c ---------------- */

/*
 * H = X X^T (cols x cols), plus `damping` * mean(diag(H)) on the diagonal when
 * damping > 0. Same code path quant_apply uses, so the brute-force reference in
 * check C4 solves the normal equations of the matrix the algorithm actually ran
 * on rather than one the test derived for itself.
 */
int quant_hessian(const double *x, int n_samples, int cols,
                  double damping, double *h_out);

/*
 * The GPTQ sweep stopped after `prefix` columns, writing the full working
 * weights (rows x cols) to w_out. Check C4 needs the state in the *middle* of
 * the sweep: after j columns are fixed, the exact least-squares solution for
 * the remaining ones has to equal what the incremental update produced.
 *
 * This shares the sweep with quant_apply; it is a different entry point, not a
 * second implementation. Returns 0, or -1.
 */
int quant_gptq_prefix(const double *w, int rows, int cols,
                      const double *x, int n_samples, double damping,
                      int bits, int prefix, double *w_out);

#endif /* QUANT_H */
