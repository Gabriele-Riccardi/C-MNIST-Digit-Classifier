/*
 * Unit tests for the pieces the gradient check does not cover: the IDX parser,
 * the train/validation split, serialisation, augmentation, and the numerical
 * guards in softmax and the loss.
 *
 * No dataset required -- every fixture is synthesised, so CI stays hermetic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "idx.h"
#include "net.h"
#include "train.h"
#include "prng.h"

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

#define CHECK_NEAR(a, b, tol) \
    CHECK(fabs((double)(a) - (double)(b)) <= (tol), \
          "%s = %g, expected %g (tol %g)", #a, (double)(a), (double)(b), (double)(tol))

static void section(const char *name) { printf("  %s\n", name); }

/* ---------------- helpers: synthetic IDX files ---------------- */

static void put_be_int(unsigned char *p, int v) {
    p[0] = (unsigned char)((v >> 24) & 0xFF);
    p[1] = (unsigned char)((v >> 16) & 0xFF);
    p[2] = (unsigned char)((v >>  8) & 0xFF);
    p[3] = (unsigned char)( v        & 0xFF);
}

static int write_image_file(const char *path, int magic, int count, int rows, int cols,
                            const unsigned char *pixels, size_t pixel_bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char hdr[16];
    put_be_int(hdr + 0,  magic);
    put_be_int(hdr + 4,  count);
    put_be_int(hdr + 8,  rows);
    put_be_int(hdr + 12, cols);
    fwrite(hdr, 1, sizeof(hdr), f);
    if (pixel_bytes) fwrite(pixels, 1, pixel_bytes, f);
    fclose(f);
    return 0;
}

static int write_label_file(const char *path, int magic, int count,
                            const unsigned char *labels, size_t label_bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char hdr[8];
    put_be_int(hdr + 0, magic);
    put_be_int(hdr + 4, count);
    fwrite(hdr, 1, sizeof(hdr), f);
    if (label_bytes) fwrite(labels, 1, label_bytes, f);
    fclose(f);
    return 0;
}

/* ---------------- tests ---------------- */

static void test_reverse_int(void) {
    section("byte swapping");
    CHECK(reverse_int(0x01000000) == 1, "0x01000000 should swap to 1");
    CHECK(reverse_int(reverse_int(0x12345678)) == 0x12345678, "swap must be an involution");
    /* The IDX magic numbers as they appear on disk. */
    CHECK(reverse_int((int)0x03080000) == 2051, "image magic");
    CHECK(reverse_int((int)0x01080000) == 2049, "label magic");
}

static void test_softmax(void) {
    section("softmax");
    real v[4] = { (real)1.0, (real)2.0, (real)3.0, (real)4.0 };
    softmax(v, 4);

    real sum = 0;
    for (int i = 0; i < 4; i++) sum += v[i];
    CHECK_NEAR(sum, 1.0, 1e-6);
    CHECK(v[3] > v[2] && v[2] > v[1] && v[1] > v[0], "softmax must preserve order");

    /* Shift invariance: adding a constant to every logit changes nothing. */
    real a[4] = { (real)1.0, (real)2.0, (real)3.0, (real)4.0 };
    real b[4] = { (real)101.0, (real)102.0, (real)103.0, (real)104.0 };
    softmax(a, 4); softmax(b, 4);
    for (int i = 0; i < 4; i++) CHECK_NEAR(a[i], b[i], 1e-6);

    /* The reason the maximum is subtracted: exp(800) is +inf. */
    real big[3] = { (real)800.0, (real)799.0, (real)-800.0 };
    softmax(big, 3);
    real big_sum = 0;
    for (int i = 0; i < 3; i++) {
        CHECK(!isnan((double)big[i]) && !isinf((double)big[i]), "big[%d] must stay finite", i);
        big_sum += big[i];
    }
    CHECK_NEAR(big_sum, 1.0, 1e-6);
}

static void test_cross_entropy(void) {
    section("cross-entropy");

    /* Equal logits: every class has probability 1/n, so the loss is log(n). */
    real uniform[10];
    for (int i = 0; i < 10; i++) uniform[i] = (real)0.0;
    CHECK_NEAR(cross_entropy_from_logits(uniform, 10, 3), log(10.0), 1e-5);

    /* Shift invariance, as for softmax. */
    real shifted[10];
    for (int i = 0; i < 10; i++) shifted[i] = (real)7.5;
    CHECK_NEAR(cross_entropy_from_logits(shifted, 10, 3), log(10.0), 1e-5);

    /* A confident hit gives a small positive loss, never a negative one. */
    real hit[3] = { (real)20.0, (real)0.0, (real)0.0 };
    const double small = (double)cross_entropy_from_logits(hit, 3, 0);
    CHECK(small > 0.0 && small < 1e-7, "confident hit gives %g, expected a tiny positive loss", small);

    /* A confident miss stays finite -- the old formulation needed a clamp to
       avoid log(0); computing from logits does not. */
    real miss[2] = { (real)200.0, (real)-200.0 };
    const double big = (double)cross_entropy_from_logits(miss, 2, 1);
    CHECK(!isinf(big) && !isnan(big) && big > 100.0,
          "confident miss gives %g, expected a large finite loss", big);

    /* It must agree with -log(softmax(z)[label]) wherever that is accurate. */
    real z[4] = { (real)1.0, (real)-0.5, (real)2.0, (real)0.25 };
    real p[4];
    memcpy(p, z, sizeof(z));
    softmax(p, 4);
    for (int i = 0; i < 4; i++)
        CHECK_NEAR(cross_entropy_from_logits(z, 4, i), -log((double)p[i]), 1e-5);
}

static void test_shift_image(void) {
    section("augmentation");
    real src[9] = { 1,2,3, 4,5,6, 7,8,9 };
    real dst[9];

    shift_image(src, dst, 3, 3, 0, 0);
    for (int i = 0; i < 9; i++) CHECK_NEAR(dst[i], src[i], 1e-9);

    /* dx = +1 moves content right; the exposed left column becomes zero. */
    shift_image(src, dst, 3, 3, 1, 0);
    CHECK_NEAR(dst[0], 0.0, 1e-9);
    CHECK_NEAR(dst[1], 1.0, 1e-9);
    CHECK_NEAR(dst[2], 2.0, 1e-9);
    CHECK_NEAR(dst[3], 0.0, 1e-9);
    CHECK_NEAR(dst[4], 4.0, 1e-9);

    /* dy = +1 moves content down; the top row becomes zero. */
    shift_image(src, dst, 3, 3, 0, 1);
    CHECK_NEAR(dst[0], 0.0, 1e-9);
    CHECK_NEAR(dst[1], 0.0, 1e-9);
    CHECK_NEAR(dst[2], 0.0, 1e-9);
    CHECK_NEAR(dst[3], 1.0, 1e-9);

    /* A shift larger than the image blanks it entirely. */
    shift_image(src, dst, 3, 3, 5, 0);
    for (int i = 0; i < 9; i++) CHECK_NEAR(dst[i], 0.0, 1e-9);
}

static void test_sgd_step(void) {
    section("SGD step");
    network *n = net_create(4, 3, 2);
    gradients *g = grad_create(n);
    CHECK(n != NULL && g != NULL, "allocation");
    if (!n || !g) return;

    for (size_t i = 0; i < 12; i++) { n->w1[i] = (real)1.0; g->dw1[i] = (real)2.0; }
    for (int i = 0; i < 3; i++)     { n->b1[i] = (real)1.0; g->db1[i] = (real)0.5; }
    for (size_t i = 0; i < 6; i++)  { n->w2[i] = (real)1.0; g->dw2[i] = (real)-1.0; }
    for (int i = 0; i < 2; i++)     { n->b2[i] = (real)1.0; g->db2[i] = (real)0.0; }

    net_sgd_step(n, g, (real)0.1);

    CHECK_NEAR(n->w1[0], 0.8, 1e-6);
    CHECK_NEAR(n->b1[0], 0.95, 1e-6);
    CHECK_NEAR(n->w2[0], 1.1, 1e-6);
    CHECK_NEAR(n->b2[0], 1.0, 1e-6);

    grad_free(g);
    net_free(n);
}

static void test_backward_zeroes_dead_rows(void) {
    section("dead ReLU rows");
    network *n = net_create(3, 4, 2);
    gradients *g = grad_create(n);
    CHECK(n != NULL && g != NULL, "allocation");
    if (!n || !g) return;

    /* Force hidden unit 0 off and unit 1 on, whatever the input. */
    for (size_t i = 0; i < 12; i++) n->w1[i] = (real)0.0;
    for (int h = 0; h < 4; h++)     n->b1[h] = (h == 0) ? (real)-1.0 : (real)1.0;
    for (size_t i = 0; i < 8; i++)  n->w2[i] = (real)0.1;
    for (int o = 0; o < 2; o++)     n->b2[o] = (real)0.0;

    real x[3] = { (real)0.5, (real)0.5, (real)0.5 };
    real probs[2];

    /* Leave rubbish in dw1 first: net_backward must overwrite all of it. */
    for (size_t i = 0; i < 12; i++) g->dw1[i] = (real)9999.0;

    net_forward(n, x, probs);
    softmax(probs, 2);
    net_backward(n, x, 0, probs, g);

    CHECK_NEAR(n->h[0], 0.0, 1e-9);
    CHECK(n->h[1] > (real)0.0, "hidden unit 1 should be active");
    CHECK_NEAR(g->db1[0], 0.0, 1e-9);
    for (int i = 0; i < 3; i++)
        CHECK_NEAR(g->dw1[0 * 3 + i], 0.0, 1e-9);

    grad_free(g);
    net_free(n);
}

static void test_save_load(void) {
    section("serialisation");
    const char *path = "test_network.tmp";

    network *a = net_create(6, 5, 3);
    network *b = net_create(6, 5, 3);
    CHECK(a != NULL && b != NULL, "allocation");
    if (!a || !b) return;

    prng_seed(99ULL, 1ULL);
    net_init_weights(a);

    CHECK(net_save(a, path) == 0, "save should succeed");
    CHECK(net_load(b, path) == 0, "load should succeed");

    /* float32 on disk, so the float build round-trips exactly. */
    const real tol = (sizeof(real) == sizeof(float)) ? (real)0.0 : (real)1e-7;
    for (size_t i = 0; i < 30; i++)
        CHECK(R_FABS(a->w1[i] - b->w1[i]) <= tol, "w1[%zu] differs after round-trip", i);
    for (size_t i = 0; i < 15; i++)
        CHECK(R_FABS(a->w2[i] - b->w2[i]) <= tol, "w2[%zu] differs after round-trip", i);

    /* A different geometry must be refused, not read into the wrong arrays. */
    network *wrong = net_create(6, 9, 3);
    CHECK(wrong != NULL, "allocation");
    if (wrong) {
        CHECK(net_load(wrong, path) != 0, "a 6-9-3 network must reject a 6-5-3 file");
        net_free(wrong);
    }

    /* So must a file that is not a network file at all. */
    FILE *junk = fopen("test_junk.tmp", "wb");
    if (junk) { fputs("not a network", junk); fclose(junk); }
    CHECK(net_load(b, "test_junk.tmp") != 0, "a bad magic must be rejected");
    CHECK(net_load(b, "test_missing.tmp") != 0, "a missing file must be reported");

    remove(path);
    remove("test_junk.tmp");
    net_free(a);
    net_free(b);
}

static void test_idx_loader(void) {
    section("IDX parser");
    const char *ipath = "test_images.tmp";
    const char *lpath = "test_labels.tmp";

    unsigned char pixels[3 * 4] = { 0, 255, 128, 0,  1, 2, 3, 4,  255, 255, 0, 0 };
    unsigned char labels[3]     = { 7, 0, 9 };

    write_image_file(ipath, 2051, 3, 2, 2, pixels, sizeof(pixels));
    write_label_file(lpath, 2049, 3, labels, sizeof(labels));

    dataset ds;
    CHECK(dataset_load(&ds, ipath, lpath) == 0, "a well-formed pair should load");
    CHECK(ds.count == 3 && ds.rows == 2 && ds.cols == 2 && ds.pixels == 4, "geometry");
    CHECK_NEAR(dataset_image(&ds, 0)[0], 0.0, 1e-6);
    CHECK_NEAR(dataset_image(&ds, 0)[1], 1.0, 1e-6);
    CHECK_NEAR(dataset_image(&ds, 0)[2], 128.0 / 255.0, 1e-6);
    CHECK_NEAR(dataset_image(&ds, 2)[0], 1.0, 1e-6);
    CHECK(ds.labels[0] == 7 && ds.labels[2] == 9, "labels");
    dataset_free(&ds);

    /* Wrong magic. */
    write_image_file(ipath, 1234, 3, 2, 2, pixels, sizeof(pixels));
    CHECK(dataset_load(&ds, ipath, lpath) != 0, "a wrong image magic must be refused");

    /* Truncated pixel data: the header promises more than the file holds. */
    write_image_file(ipath, 2051, 3, 2, 2, pixels, 5);
    CHECK(dataset_load(&ds, ipath, lpath) != 0, "truncated pixel data must be refused");

    /* Counts that disagree between the two files. */
    write_image_file(ipath, 2051, 3, 2, 2, pixels, sizeof(pixels));
    write_label_file(lpath, 2049, 2, labels, 2);
    CHECK(dataset_load(&ds, ipath, lpath) != 0, "a mismatched pair must be refused");

    /* A label outside 0..9. */
    unsigned char bad_labels[3] = { 7, 42, 9 };
    write_label_file(lpath, 2049, 3, bad_labels, sizeof(bad_labels));
    CHECK(dataset_load(&ds, ipath, lpath) != 0, "an out-of-range label must be refused");

    remove(ipath);
    remove(lpath);
}

static void test_split(void) {
    section("train / validation split");
    const int count = 200;

    dataset ds;
    memset(&ds, 0, sizeof(ds));
    ds.count = count; ds.rows = 1; ds.cols = 1; ds.pixels = 1;
    ds.images = (real *)calloc((size_t)count, sizeof(real));
    ds.labels = (int  *)calloc((size_t)count, sizeof(int));

    subset tr, va;
    CHECK(subset_split(&ds, 50, 1234ULL, &tr, &va) == 0, "split should succeed");
    CHECK(tr.count == 150 && va.count == 50, "sizes: %d / %d", tr.count, va.count);

    /* Disjoint, and together they cover every index exactly once. */
    int *seen = (int *)calloc((size_t)count, sizeof(int));
    for (int i = 0; i < tr.count; i++) seen[tr.index[i]]++;
    for (int i = 0; i < va.count; i++) seen[va.index[i]]++;
    int covered = 1;
    for (int i = 0; i < count; i++) if (seen[i] != 1) covered = 0;
    CHECK(covered, "every index must appear in exactly one side of the split");
    free(seen);

    /* The same split seed must reproduce the same hold-out set. */
    subset tr2, va2;
    CHECK(subset_split(&ds, 50, 1234ULL, &tr2, &va2) == 0, "second split");
    int identical = 1;
    for (int i = 0; i < va.count; i++) if (va.index[i] != va2.index[i]) identical = 0;
    CHECK(identical, "the split must be deterministic in its seed");

    /* A different split seed must move it. */
    subset tr3, va3;
    CHECK(subset_split(&ds, 50, 999ULL, &tr3, &va3) == 0, "third split");
    int same_again = 1;
    for (int i = 0; i < va.count; i++) if (va.index[i] != va3.index[i]) same_again = 0;
    CHECK(!same_again, "a different split seed should produce a different hold-out set");

    /* Degenerate requests are rejected rather than silently clamped. */
    subset t4, v4;
    CHECK(subset_split(&ds, count, 1ULL, &t4, &v4) != 0, "an empty training side must be refused");
    CHECK(subset_split(&ds, -1, 1ULL, &t4, &v4) != 0, "a negative hold-out must be refused");

    subset_free(&tr);  subset_free(&va);
    subset_free(&tr2); subset_free(&va2);
    subset_free(&tr3); subset_free(&va3);
    dataset_free(&ds);
}

static void test_split_independent_of_run_seed(void) {
    section("split is independent of the run seed");
    const int count = 500;

    dataset ds;
    memset(&ds, 0, sizeof(ds));
    ds.count = count; ds.rows = 1; ds.cols = 1; ds.pixels = 1;
    ds.images = (real *)calloc((size_t)count, sizeof(real));
    ds.labels = (int  *)calloc((size_t)count, sizeof(int));

    /* Drive the global generator to different states between the two splits.
       If subset_split used it, the hold-out sets would differ. */
    subset a_tr, a_va, b_tr, b_va;
    prng_seed(1ULL, 1ULL);
    subset_split(&ds, 100, 7ULL, &a_tr, &a_va);
    prng_seed(2ULL, 1ULL);
    for (int i = 0; i < 1000; i++) (void)prng_rand();
    subset_split(&ds, 100, 7ULL, &b_tr, &b_va);

    int identical = 1;
    for (int i = 0; i < a_va.count; i++) if (a_va.index[i] != b_va.index[i]) identical = 0;
    CHECK(identical, "the hold-out set must not move when the training seed does");

    subset_free(&a_tr); subset_free(&a_va);
    subset_free(&b_tr); subset_free(&b_va);
    dataset_free(&ds);
}

static void test_prng_bounds(void) {
    section("PRNG");
    prng_seed(5ULL, 3ULL);
    int hist[7] = {0};
    for (int i = 0; i < 70000; i++) {
        const u32 v = prng_below(7);
        CHECK(v < 7u, "prng_below(7) returned %u", v);
        if (v < 7u) hist[v]++;
    }
    /* 10,000 expected per bucket; a broken modulo shows up far outside this. */
    for (int i = 0; i < 7; i++)
        CHECK(hist[i] > 9000 && hist[i] < 11000, "bucket %d got %d, expected ~10000", i, hist[i]);

    for (int i = 0; i < 1000; i++) {
        const float f = prng_randf();
        CHECK(f >= 0.0f && f <= 1.0f, "prng_randf() returned %g", (double)f);
    }
}

static void test_predict_matches_softmax(void) {
    section("prediction");
    network *n = net_create(8, 6, 4);
    CHECK(n != NULL, "allocation");
    if (!n) return;
    prng_seed(11ULL, 1ULL);
    net_init_weights(n);

    real x[8], probs[4], scratch[4];
    for (int i = 0; i < 8; i++) x[i] = (real)prng_randf();

    net_forward(n, x, probs);
    softmax(probs, 4);
    int expected = 0;
    for (int o = 1; o < 4; o++) if (probs[o] > probs[expected]) expected = o;

    CHECK(net_predict(n, x, scratch) == expected, "argmax must agree with the softmax argmax");
    net_free(n);
}

/* One epoch on a tiny separable problem must reduce the loss. */
static void test_training_reduces_loss(void) {
    section("training smoke test");
    const int count = 400, pixels = 16;

    dataset ds;
    memset(&ds, 0, sizeof(ds));
    ds.count = count; ds.rows = 4; ds.cols = 4; ds.pixels = pixels;
    ds.images = (real *)calloc((size_t)count * pixels, sizeof(real));
    ds.labels = (int  *)calloc((size_t)count, sizeof(int));

    prng_seed(31ULL, 1ULL);
    /* Class k lights up pixel k, plus noise: linearly separable, learnable fast. */
    for (int i = 0; i < count; i++) {
        const int k = i % 4;
        ds.labels[i] = k;
        for (int p = 0; p < pixels; p++)
            ds.images[(size_t)i * pixels + p] = (real)(0.1 * prng_randf());
        ds.images[(size_t)i * pixels + k] = (real)1.0;
    }

    subset all;
    subset_all(&ds, &all);

    network *n = net_create(pixels, 8, 4);
    net_init_weights(n);

    trainer t;
    trainer_init(&t, n, &all);

    const train_config cfg = { 1, (real)0.05, (real)0.0, 0 };
    const real first = train_epoch(n, &all, &cfg, &t);
    real last = first;
    for (int e = 0; e < 15; e++) last = train_epoch(n, &all, &cfg, &t);

    real scratch[4];
    const real acc = evaluate(n, &all, scratch);

    CHECK(last < first, "loss should fall: %g -> %g", (double)first, (double)last);
    CHECK(acc > (real)0.95, "a separable problem should be learned, got %.3f", (double)acc);

    trainer_free(&t);
    net_free(n);
    subset_free(&all);
    dataset_free(&ds);
}

int main(void) {
    printf("unit tests  (%s)\n", REAL_NAME);

    test_reverse_int();
    test_softmax();
    test_cross_entropy();
    test_shift_image();
    test_sgd_step();
    test_backward_zeroes_dead_rows();
    test_save_load();
    test_idx_loader();
    test_split();
    test_split_independent_of_run_seed();
    test_prng_bounds();
    test_predict_matches_softmax();
    test_training_reduces_loss();

    printf("\n%d checks, %d failures\n", checks, failures);
    return failures != 0;
}
