#ifndef IDX_H
#define IDX_H

/*
 * MNIST IDX reader.
 *
 * Images: 16-byte header, then one unsigned byte per pixel.
 *     offset  0  magic number (2051)
 *     offset  4  number of images
 *     offset  8  number of rows
 *     offset 12  number of columns
 *     offset 16  pixel data
 *
 * Labels: 8-byte header, then one unsigned byte per label.
 *     offset  0  magic number (2049)
 *     offset  4  number of items
 *     offset  8  label data
 *
 * Every header field is big-endian and is byte-swapped on read.
 */

#include <stddef.h>
#include "real.h"

typedef struct {
    int   count;    /* number of samples */
    int   rows;
    int   cols;
    int   pixels;   /* rows * cols */
    real *images;   /* count * pixels, row-major, normalised to [0,1] */
    int  *labels;   /* count */
} dataset;

/* A view over a dataset: an index list, no pixel data of its own. */
typedef struct {
    const dataset *ds;
    int           *index;
    int            count;
} subset;

int  dataset_load(dataset *ds, const char *images_path, const char *labels_path);
void dataset_free(dataset *ds);

static inline const real *dataset_image(const dataset *ds, int i) {
    return ds->images + (size_t)i * (size_t)ds->pixels;
}

/*
 * Deterministic hold-out split. `val_count` samples go to `val`, the rest to
 * `train`; the two are disjoint and together cover every index exactly once.
 *
 * The permutation is drawn from a private PRNG stream seeded by `split_seed`,
 * independent of the global generator, so that changing the training seed does
 * not change which samples are held out.
 */
int  subset_split(const dataset *ds, int val_count, unsigned long long split_seed,
                  subset *train, subset *val);
void subset_free(subset *s);

/* A view over the whole dataset, in file order. */
int  subset_all(const dataset *ds, subset *s);

int  reverse_int(int i);

#endif /* IDX_H */
