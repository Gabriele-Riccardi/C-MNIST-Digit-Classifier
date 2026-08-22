#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "idx.h"
#include "prng.h"

int reverse_int(int i) {
    unsigned char c1 =  (unsigned)i        & 255u;
    unsigned char c2 = ((unsigned)i >> 8)  & 255u;
    unsigned char c3 = ((unsigned)i >> 16) & 255u;
    unsigned char c4 = ((unsigned)i >> 24) & 255u;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

static int read_be_int(FILE *fp, int *out) {
    int raw = 0;
    if (fread(&raw, sizeof(raw), 1, fp) != 1) return -1;
    *out = reverse_int(raw);
    return 0;
}

/*
 * Reads the whole pixel block in one call and converts it in place afterwards.
 * The historical version issued one fread per pixel -- 47 million calls for the
 * training set. tools/bench_load.c measures the difference.
 */
static int load_images(dataset *ds, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "%s: ", path); perror("fopen"); return -1; }

    int magic = 0, count = 0, rows = 0, cols = 0;
    if (read_be_int(fp, &magic) || read_be_int(fp, &count) ||
        read_be_int(fp, &rows)  || read_be_int(fp, &cols)) {
        fprintf(stderr, "%s: truncated header\n", path);
        fclose(fp);
        return -1;
    }

    if (magic != 2051) {
        fprintf(stderr, "%s: not an IDX image file (magic %d, expected 2051)\n", path, magic);
        fclose(fp);
        return -1;
    }
    if (count <= 0 || rows <= 0 || cols <= 0) {
        fprintf(stderr, "%s: nonsensical geometry %dx%dx%d\n", path, count, rows, cols);
        fclose(fp);
        return -1;
    }

    const size_t pixels = (size_t)rows * (size_t)cols;
    const size_t total  = (size_t)count * pixels;

    /* One contiguous allocation for the whole set instead of `count` of them:
       60,000 mallocs become 1, and the pixels end up sequential in memory. */
    real          *images = (real *)malloc(total * sizeof(real));
    unsigned char *raw    = (unsigned char *)malloc(total);
    if (!images || !raw) {
        fprintf(stderr, "%s: out of memory for %zu pixels\n", path, total);
        free(images); free(raw); fclose(fp);
        return -1;
    }

    if (fread(raw, 1, total, fp) != total) {
        fprintf(stderr, "%s: truncated pixel data (expected %zu bytes)\n", path, total);
        free(images); free(raw); fclose(fp);
        return -1;
    }
    fclose(fp);

    /* Scale to [0,1]: unnormalised inputs make the first-layer gradients blow up. */
    for (size_t p = 0; p < total; p++)
        images[p] = (real)raw[p] / (real)255.0;

    free(raw);

    ds->count  = count;
    ds->rows   = rows;
    ds->cols   = cols;
    ds->pixels = (int)pixels;
    ds->images = images;
    return 0;
}

static int load_labels(dataset *ds, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "%s: ", path); perror("fopen"); return -1; }

    int magic = 0, count = 0;
    if (read_be_int(fp, &magic) || read_be_int(fp, &count)) {
        fprintf(stderr, "%s: truncated header\n", path);
        fclose(fp);
        return -1;
    }

    if (magic != 2049) {
        fprintf(stderr, "%s: not an IDX label file (magic %d, expected 2049)\n", path, magic);
        fclose(fp);
        return -1;
    }
    if (count <= 0) {
        fprintf(stderr, "%s: nonsensical label count %d\n", path, count);
        fclose(fp);
        return -1;
    }

    int           *labels = (int *)malloc((size_t)count * sizeof(int));
    unsigned char *raw    = (unsigned char *)malloc((size_t)count);
    if (!labels || !raw) {
        fprintf(stderr, "%s: out of memory for %d labels\n", path, count);
        free(labels); free(raw); fclose(fp);
        return -1;
    }

    if (fread(raw, 1, (size_t)count, fp) != (size_t)count) {
        fprintf(stderr, "%s: truncated label data\n", path);
        free(labels); free(raw); fclose(fp);
        return -1;
    }
    fclose(fp);

    for (int i = 0; i < count; i++) {
        if (raw[i] > 9) {
            fprintf(stderr, "%s: label %d out of range at index %d\n", path, raw[i], i);
            free(labels); free(raw);
            return -1;
        }
        labels[i] = (int)raw[i];
    }

    free(raw);

    /* load_images ran first and set ds->count; the two files must agree. */
    if (ds->images && ds->count != count) {
        fprintf(stderr, "%s: %d labels for %d images -- mismatched pair\n", path, count, ds->count);
        free(labels);
        return -1;
    }

    ds->count  = count;
    ds->labels = labels;
    return 0;
}

int dataset_load(dataset *ds, const char *images_path, const char *labels_path) {
    memset(ds, 0, sizeof(*ds));
    if (load_images(ds, images_path) != 0) return -1;
    if (load_labels(ds, labels_path) != 0) { dataset_free(ds); return -1; }
    return 0;
}

void dataset_free(dataset *ds) {
    if (!ds) return;
    free(ds->images);
    free(ds->labels);
    memset(ds, 0, sizeof(*ds));
}

int subset_all(const dataset *ds, subset *s) {
    s->ds    = ds;
    s->count = ds->count;
    s->index = (int *)malloc((size_t)ds->count * sizeof(int));
    if (!s->index) return -1;
    for (int i = 0; i < ds->count; i++) s->index[i] = i;
    return 0;
}

int subset_split(const dataset *ds, int val_count, unsigned long long split_seed,
                 subset *train, subset *val) {
    if (val_count < 0 || val_count >= ds->count) {
        fprintf(stderr, "split: %d validation samples out of %d is not a split\n",
                val_count, ds->count);
        return -1;
    }

    int *perm = (int *)malloc((size_t)ds->count * sizeof(int));
    if (!perm) return -1;
    for (int i = 0; i < ds->count; i++) perm[i] = i;

    /* Private stream: the hold-out set must not move when the training seed does,
       or validation numbers from two runs would not be comparable. */
    prng_state rng;
    prng_seed_r(&rng, split_seed, 0ULL);
    for (int i = ds->count - 1; i > 0; i--) {
        int j = (int)prng_below_r(&rng, (u32)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }

    const int train_count = ds->count - val_count;

    val->ds      = ds;
    val->count   = val_count;
    val->index   = (int *)malloc((size_t)(val_count   ? val_count   : 1) * sizeof(int));
    train->ds    = ds;
    train->count = train_count;
    train->index = (int *)malloc((size_t)(train_count ? train_count : 1) * sizeof(int));

    if (!val->index || !train->index) {
        free(perm); free(val->index); free(train->index);
        val->index = train->index = NULL;
        return -1;
    }

    memcpy(val->index,   perm,             (size_t)val_count   * sizeof(int));
    memcpy(train->index, perm + val_count, (size_t)train_count * sizeof(int));

    free(perm);
    return 0;
}

void subset_free(subset *s) {
    if (!s) return;
    free(s->index);
    s->index = NULL;
    s->count = 0;
    s->ds    = NULL;
}
