#include <stdlib.h>
#include <string.h>

#include "train.h"
#include "prng.h"

int trainer_init(trainer *t, const network *n, const subset *train_set) {
    memset(t, 0, sizeof(*t));

    t->grad      = grad_create(n);
    t->probs     = (real *)malloc((size_t)n->output * sizeof(real));
    t->shifted   = (real *)malloc((size_t)train_set->ds->pixels * sizeof(real));
    t->order     = (int  *)malloc((size_t)train_set->count * sizeof(int));

    if (!t->grad || !t->probs || !t->shifted || !t->order) {
        trainer_free(t);
        return -1;
    }

    for (int i = 0; i < train_set->count; i++) t->order[i] = i;
    return 0;
}

void trainer_free(trainer *t) {
    if (!t) return;
    grad_free(t->grad);
    free(t->probs);
    free(t->shifted);
    free(t->order);
    memset(t, 0, sizeof(*t));
}

void shift_image(const real *src, real *dst, int rows, int cols, int dx, int dy) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const int sr = r - dy, sc = c - dx;
            dst[r * cols + c] = (sr >= 0 && sr < rows && sc >= 0 && sc < cols)
                              ? src[sr * cols + sc]
                              : (real)0.0;
        }
    }
}

real train_epoch(network *n, const subset *s, const train_config *cfg, trainer *t) {
    const dataset *ds = s->ds;
    real total_loss = (real)0.0;

    /* Fisher-Yates over the position list: a fixed presentation order biases
       the updates towards whatever the set happens to end with. */
    for (int i = s->count - 1; i > 0; i--) {
        const int j = (int)prng_below((u32)(i + 1));
        const int tmp = t->order[i]; t->order[i] = t->order[j]; t->order[j] = tmp;
    }

    for (int k = 0; k < s->count; k++) {
        const int    i     = s->index[t->order[k]];
        const int    label = ds->labels[i];
        const real  *x     = dataset_image(ds, i);

        /* Augmentation on a fraction of the samples. A fully connected network
           has no translation invariance, so shifting every image would train it
           on a distribution the evaluation set does not share. */
        if (cfg->aug_prob > (real)0.0 && (real)prng_randf() < cfg->aug_prob) {
            const int span = 2 * cfg->max_shift + 1;
            const int dx = (int)prng_below((u32)span) - cfg->max_shift;
            const int dy = (int)prng_below((u32)span) - cfg->max_shift;
            shift_image(x, t->shifted, ds->rows, ds->cols, dx, dy);
            x = t->shifted;   /* both passes must see the same buffer */
        }

        /* The loss reads the logits, so it is taken before softmax overwrites
           them in place; backprop then works from the probabilities. */
        net_forward(n, x, t->probs);
        total_loss += cross_entropy_from_logits(t->probs, n->output, label);
        softmax(t->probs, n->output);

        net_backward(n, x, label, t->probs, t->grad);
        net_sgd_step(n, t->grad, cfg->lr);
    }

    return total_loss / (real)s->count;
}

real evaluate(network *n, const subset *s, real *scratch) {
    const dataset *ds = s->ds;
    int correct = 0;

    for (int k = 0; k < s->count; k++) {
        const int i = s->index[k];
        if (net_predict(n, dataset_image(ds, i), scratch) == ds->labels[i])
            correct++;
    }

    return (real)correct / (real)s->count;
}
