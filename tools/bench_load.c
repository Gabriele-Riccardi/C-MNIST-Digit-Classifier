/*
 * Reproducible measurement of the two changes claimed as optimisations:
 * block I/O in the IDX loader, and PCG in place of rand().
 *
 * The point is that anyone can run this and get the numbers in the README,
 * rather than taking a "5x faster" on trust. Three loader variants are timed
 * against the same file:
 *
 *   v0  one fread per pixel, one malloc per image   (the original)
 *   v1  one fread per image,  one malloc per image
 *   v2  one fread for the file, one allocation      (current, src/idx.c)
 *
 * Each is run --repeat times and the fastest run of each is reported, since the
 * minimum is the measurement least polluted by other work on the machine.
 *
 * usage: bench_load [--data DIR] [--repeat N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "idx.h"
#include "util.h"
#include "prng.h"

/* Somewhere for the timed loops to leave their results, so the optimiser cannot
   decide the work was pointless and delete it. */
static volatile double g_sink;

/* ---------------- v0: the original, one fread per pixel ---------------- */

static int load_v0(const char *path, double *seconds) {
    const double t0 = now_seconds();

    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); return -1; }

    int magic = 0, count = 0, rows = 0, cols = 0;
    if (fread(&magic, 4, 1, fp) != 1) { fclose(fp); return -1; }
    magic = reverse_int(magic);
    if (magic != 2051) { fclose(fp); return -1; }
    if (fread(&count, 4, 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&rows,  4, 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&cols,  4, 1, fp) != 1) { fclose(fp); return -1; }
    count = reverse_int(count); rows = reverse_int(rows); cols = reverse_int(cols);

    float **images = (float **)calloc((size_t)count, sizeof(float *));
    if (!images) { fclose(fp); return -1; }

    for (int i = 0; i < count; i++) {
        images[i] = (float *)malloc((size_t)rows * cols * sizeof(float));
        if (!images[i]) { fclose(fp); return -1; }
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                unsigned char px = 0;
                if (fread(&px, 1, 1, fp) != 1) { fclose(fp); return -1; }
                images[i][r * cols + c] = px / 255.0f;
            }
        }
    }
    fclose(fp);

    double sink = 0.0;
    for (int i = 0; i < count; i += 997) sink += images[i][0];
    g_sink = sink;

    for (int i = 0; i < count; i++) free(images[i]);
    free(images);

    *seconds = now_seconds() - t0;
    return 0;
}

/* ---------------- v1: block read per image, malloc per image ---------------- */

static int load_v1(const char *path, double *seconds) {
    const double t0 = now_seconds();

    FILE *fp = fopen(path, "rb");
    if (!fp) { perror("fopen"); return -1; }

    int magic = 0, count = 0, rows = 0, cols = 0;
    if (fread(&magic, 4, 1, fp) != 1) { fclose(fp); return -1; }
    magic = reverse_int(magic);
    if (magic != 2051) { fclose(fp); return -1; }
    if (fread(&count, 4, 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&rows,  4, 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&cols,  4, 1, fp) != 1) { fclose(fp); return -1; }
    count = reverse_int(count); rows = reverse_int(rows); cols = reverse_int(cols);

    const size_t pixels = (size_t)rows * cols;
    float        **images = (float **)calloc((size_t)count, sizeof(float *));
    unsigned char *row    = (unsigned char *)malloc(pixels);
    if (!images || !row) { fclose(fp); return -1; }

    for (int i = 0; i < count; i++) {
        images[i] = (float *)malloc(pixels * sizeof(float));
        if (!images[i] || fread(row, 1, pixels, fp) != pixels) { fclose(fp); return -1; }
        for (size_t p = 0; p < pixels; p++) images[i][p] = row[p] / 255.0f;
    }
    fclose(fp);
    free(row);

    double sink = 0.0;
    for (int i = 0; i < count; i += 997) sink += images[i][0];
    g_sink = sink;

    for (int i = 0; i < count; i++) free(images[i]);
    free(images);

    *seconds = now_seconds() - t0;
    return 0;
}

/* ---------------- v2: the current loader ---------------- */

static int load_v2(const char *images_path, const char *labels_path, double *seconds) {
    const double t0 = now_seconds();

    dataset ds;
    if (dataset_load(&ds, images_path, labels_path) != 0) return -1;

    double sink = 0.0;
    for (int i = 0; i < ds.count; i += 997) sink += (double)dataset_image(&ds, i)[0];
    g_sink = sink;

    dataset_free(&ds);
    *seconds = now_seconds() - t0;
    return 0;
}

/* ---------------- RNG throughput ---------------- */

static void bench_rng(int repeat) {
    const long draws = 20000000L;

    double best_rand = 1e30, best_pcg = 1e30;

    for (int r = 0; r < repeat; r++) {
        srand(1234u);
        double t0 = now_seconds();
        unsigned long acc = 0;
        for (long i = 0; i < draws; i++) acc += (unsigned long)rand();
        double dt = now_seconds() - t0;
        if (dt < best_rand) best_rand = dt;
        g_sink = (double)acc;

        prng_seed(1234ULL, 1ULL);
        t0 = now_seconds();
        acc = 0;
        for (long i = 0; i < draws; i++) acc += (unsigned long)prng_rand();
        dt = now_seconds() - t0;
        if (dt < best_pcg) best_pcg = dt;
        g_sink = (double)acc;
    }

    printf("\nrandom number generation  (%ld draws, best of %d)\n", draws, repeat);
    printf("  rand()        %8.3f s   %6.2f ns/draw\n", best_rand, best_rand / draws * 1e9);
    printf("  prng_rand()   %8.3f s   %6.2f ns/draw   %.2fx\n",
           best_pcg, best_pcg / draws * 1e9, best_rand / best_pcg);

    /* What that is worth per epoch: the training loop draws about four numbers
       per sample (one for the shuffle, one for the augmentation coin, two for
       the offsets), so 60,000 samples cost 240,000 draws. */
    const double per_epoch_rand = best_rand / draws * 240000.0;
    const double per_epoch_pcg  = best_pcg  / draws * 240000.0;
    printf("  ~4 draws x 60,000 samples per epoch: %.4f s with rand(), %.4f s with PCG\n",
           per_epoch_rand, per_epoch_pcg);
    printf("  difference per epoch: %.4f s\n", per_epoch_rand - per_epoch_pcg);
}

int main(int argc, char **argv) {
    const char *data_dir = "dataset";
    int repeat = 5;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i + 1 < argc)        data_dir = argv[++i];
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat   = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--data DIR] [--repeat N]\n", argv[0]); return 2; }
    }
    if (repeat < 1) repeat = 1;

    char images[1024], labels[1024];
    snprintf(images, sizeof(images), "%s/train-images.idx3-ubyte", data_dir);
    snprintf(labels, sizeof(labels), "%s/train-labels.idx1-ubyte", data_dir);

    printf("bench_load  %s  best of %d\n", images, repeat);

    /* One discarded pass first. Otherwise the first timed read also pays for
       pulling 47 MB off disk into the page cache, and the numbers below would
       be measuring the storage rather than the parsing, which is the part that
       changed. */
    double warm = 0.0;
    if (load_v2(images, labels, &warm) != 0) {
        fprintf(stderr, "could not read %s -- run scripts/get_dataset.sh first\n", images);
        return 1;
    }
    printf("(cold pass, discarded: %.3f s)\n", warm);

    double best[3] = { 1e30, 1e30, 1e30 };
    for (int r = 0; r < repeat; r++) {
        double t;
        if (load_v0(images, &t) != 0) { fprintf(stderr, "v0 failed\n"); return 1; }
        if (t < best[0]) best[0] = t;
        if (load_v1(images, &t) != 0) { fprintf(stderr, "v1 failed\n"); return 1; }
        if (t < best[1]) best[1] = t;
        if (load_v2(images, labels, &t) != 0) { fprintf(stderr, "v2 failed\n"); return 1; }
        if (t < best[2]) best[2] = t;
    }

    printf("\ntraining-set load  (60,000 images, 47 MB)\n");
    printf("  v0  fread per pixel,  malloc per image   %7.3f s   1.00x\n", best[0]);
    printf("  v1  fread per image,  malloc per image   %7.3f s   %5.2fx\n", best[1], best[0] / best[1]);
    printf("  v2  one read, one allocation  (current)  %7.3f s   %5.2fx\n", best[2], best[0] / best[2]);

    bench_rng(repeat);
    return 0;
}
