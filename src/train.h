#ifndef TRAIN_H
#define TRAIN_H

#include "idx.h"
#include "net.h"

typedef struct {
    int  epochs;
    real lr;
    real aug_prob;    /* probability that a training sample is translated */
    int  max_shift;   /* translation is drawn uniformly from [-max_shift, +max_shift] */
} train_config;

/* Scratch buffers for one training run: allocated once, reused every sample. */
typedef struct {
    gradients *grad;
    real      *probs;     /* net->output */
    real      *shifted;   /* one image */
    int       *order;     /* presentation order, reshuffled each epoch */
} trainer;

int  trainer_init(trainer *t, const network *n, const subset *train_set);
void trainer_free(trainer *t);

/* Translates by (dx, dy), padding the exposed border with zeros. */
void shift_image(const real *src, real *dst, int rows, int cols, int dx, int dy);

/* Runs one epoch of per-sample SGD and returns the mean training loss. */
real train_epoch(network *n, const subset *s, const train_config *cfg, trainer *t);

/* Fraction of correctly classified samples in `s`. No augmentation, no updates. */
real evaluate(network *n, const subset *s, real *scratch);

#endif /* TRAIN_H */
