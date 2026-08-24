/*
 * Post-training quantisation sweep.
 *
 * Loads a trained network, quantises it under a grid of configurations, and
 * reports for each one: the per-layer reconstruction error on the calibration
 * set, the same error on activations the calibration never saw, the largest
 * weight movement, validation accuracy, and the size the quantised parameters
 * would occupy if they were packed.
 *
 * It does not open the test files, and nothing in it can: the only dataset it
 * loads is the training pair, and the test number for a quantised network comes
 * from `./mnist test --weights ...` afterwards, exactly as it does for an
 * unquantised one. scripts/quant_seeds.sh drives both halves. Keeping the
 * separation is worth more here than in the baseline, because the number of
 * configurations to choose between is much larger and the temptation to pick
 * one by test accuracy is correspondingly stronger.
 *
 * The held-out error is the interesting column. The calibration error is the
 * quantity the algorithm minimises, so it can only go down as the calibration
 * set is tailored; the same objective measured on the validation split is what
 * says whether X X^T was estimated or memorised.
 *
 * usage: build/quant_sweep [options]   (--help for the list)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "idx.h"
#include "net.h"
#include "quant.h"
#include "train.h"
#include "util.h"
#include "prng.h"

#define MAX_LIST 16

#define DEFAULT_DATA_DIR   "dataset"
#define DEFAULT_WEIGHTS    "network.dat"
#define DEFAULT_OUT_DIR    "runs/quant"
#define DEFAULT_SPLIT_SEED 20260822ULL

typedef struct {
    const char        *data_dir;
    const char        *weights;
    const char        *out_dir;
    const char        *tag;
    int                hidden;
    int                val_count;
    unsigned long long split_seed;
    unsigned long long calib_seed;
    int                calib_n;
    int                holdout_n;
    double             damping;
    int                bits[MAX_LIST];      int bits_n;
    quant_method       methods[MAX_LIST];   int methods_n;
    int                layers[MAX_LIST];    int layers_n;   /* bit 1 = w1, bit 2 = w2 */
    int                save;
} sweep_options;

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s [options]\n"
        "\n"
        "  --data DIR       directory with the training IDX files (default \"%s\")\n"
        "  --weights FILE   trained network to quantise (default \"%s\")\n"
        "  --out DIR        where quantised weight files are written (default \"%s\")\n"
        "  --tag NAME       prefix for the written filenames (default \"q\")\n"
        "  --hidden N       hidden units, must match the weight file (default 128)\n"
        "  --val N          validation samples held out of the training set (default 6000)\n"
        "  --split-seed N   seed for the train/validation split (default %llu)\n"
        "  --calib N        calibration samples, drawn from the training split (default 512)\n"
        "  --calib-seed N   seed choosing which training samples calibrate (default 1)\n"
        "  --holdout N      validation samples used to score the objective off-calibration (default 2048)\n"
        "  --damping F      ridge as a fraction of mean(diag(H)) (default 0.01)\n"
        "  --bits LIST      comma-separated bit widths (default 8,6,4,3,2)\n"
        "  --method LIST    rtn, gptq, or both (default rtn,gptq)\n"
        "  --layers LIST    w1, w2, both -- comma-separated, for the per-layer ablation (default both)\n"
        "  --no-save        do not write quantised weight files\n"
        "\n"
        "The test set is not opened by this program. Score a quantised file with\n"
        "  ./mnist test --weights FILE\n",
        argv0, DEFAULT_DATA_DIR, DEFAULT_WEIGHTS, DEFAULT_OUT_DIR, DEFAULT_SPLIT_SEED);
}

/* ---------------- option parsing ---------------- */

static int parse_int_list(const char *s, int *out, int cap) {
    int n = 0;
    while (*s && n < cap) {
        char *end = NULL;
        const long v = strtol(s, &end, 10);
        if (end == s) return -1;
        out[n++] = (int)v;
        s = end;
        if (*s == ',') s++;
        else if (*s)   return -1;
    }
    return (*s == '\0') ? n : -1;
}

static int parse_method_list(const char *s, quant_method *out, int cap) {
    int n = 0;
    char buf[128];
    if (strlen(s) >= sizeof(buf)) return -1;
    strcpy(buf, s);

    for (char *tok = strtok(buf, ","); tok && n < cap; tok = strtok(NULL, ",")) {
        if      (!strcmp(tok, "rtn"))  out[n++] = QUANT_RTN;
        else if (!strcmp(tok, "gptq")) out[n++] = QUANT_GPTQ;
        else if (!strcmp(tok, "both")) { out[n++] = QUANT_RTN; if (n < cap) out[n++] = QUANT_GPTQ; }
        else return -1;
    }
    return n;
}

static int parse_layer_list(const char *s, int *out, int cap) {
    int n = 0;
    char buf[128];
    if (strlen(s) >= sizeof(buf)) return -1;
    strcpy(buf, s);

    for (char *tok = strtok(buf, ","); tok && n < cap; tok = strtok(NULL, ",")) {
        if      (!strcmp(tok, "w1"))   out[n++] = 1;
        else if (!strcmp(tok, "w2"))   out[n++] = 2;
        else if (!strcmp(tok, "both")) out[n++] = 3;
        else return -1;
    }
    return n;
}

static const char *layer_name(int mask) {
    return (mask == 1) ? "w1" : (mask == 2) ? "w2" : "both";
}

static int parse_options(int argc, char **argv, sweep_options *o) {
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) return 1;
        if (!strcmp(a, "--no-save")) { o->save = 0; continue; }

        if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a); return -1; }
        const char *v = argv[++i];

        if      (!strcmp(a, "--data"))       o->data_dir   = v;
        else if (!strcmp(a, "--weights"))    o->weights    = v;
        else if (!strcmp(a, "--out"))        o->out_dir    = v;
        else if (!strcmp(a, "--tag"))        o->tag        = v;
        else if (!strcmp(a, "--hidden"))     o->hidden     = atoi(v);
        else if (!strcmp(a, "--val"))        o->val_count  = atoi(v);
        else if (!strcmp(a, "--calib"))      o->calib_n    = atoi(v);
        else if (!strcmp(a, "--holdout"))    o->holdout_n  = atoi(v);
        else if (!strcmp(a, "--damping"))    o->damping    = atof(v);
        else if (!strcmp(a, "--split-seed")) o->split_seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--calib-seed")) o->calib_seed = strtoull(v, NULL, 10);
        else if (!strcmp(a, "--bits")) {
            o->bits_n = parse_int_list(v, o->bits, MAX_LIST);
            if (o->bits_n <= 0) { fprintf(stderr, "bad --bits: %s\n", v); return -1; }
        } else if (!strcmp(a, "--method")) {
            o->methods_n = parse_method_list(v, o->methods, MAX_LIST);
            if (o->methods_n <= 0) { fprintf(stderr, "bad --method: %s\n", v); return -1; }
        } else if (!strcmp(a, "--layers")) {
            o->layers_n = parse_layer_list(v, o->layers, MAX_LIST);
            if (o->layers_n <= 0) { fprintf(stderr, "bad --layers: %s\n", v); return -1; }
        } else {
            fprintf(stderr, "unknown option: %s\n", a);
            return -1;
        }
    }
    return 0;
}

/* ---------------- helpers ---------------- */

static int join(char *dst, size_t cap, const char *dir, const char *name) {
    const int written = snprintf(dst, cap, "%s/%s", dir, name);
    return (written < 0 || (size_t)written >= cap) ? -1 : 0;
}

/*
 * Held-out indices come from the validation split, so the off-calibration
 * objective is measured on samples the quantiser has never touched. This is a
 * measurement, not a selection: no configuration is chosen by it here, and the
 * test set stays where it belongs.
 */
static int select_holdout(const subset *val_set, int count, unsigned long long seed, int *out) {
    if (count > val_set->count) count = val_set->count;

    int *perm = (int *)malloc((size_t)val_set->count * sizeof(int));
    if (!perm) return -1;
    for (int i = 0; i < val_set->count; i++) perm[i] = i;

    prng_state rng;
    prng_seed_r(&rng, seed, 17ULL);
    for (int i = 0; i < count; i++) {
        const int j = i + (int)prng_below_r(&rng, (u32)(val_set->count - i));
        const int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        out[i] = val_set->index[perm[i]];
    }
    free(perm);
    return count;
}

int main(int argc, char **argv) {
    sweep_options o = {
        DEFAULT_DATA_DIR, DEFAULT_WEIGHTS, DEFAULT_OUT_DIR, "q",
        128, 6000, DEFAULT_SPLIT_SEED, 1ULL,
        512, 2048, 0.01,
        { 8, 6, 4, 3, 2 }, 5,
        { QUANT_RTN, QUANT_GPTQ }, 2,
        { 3 }, 1,
        1
    };

    const int parsed = parse_options(argc, argv, &o);
    if (parsed < 0) return 2;
    if (parsed > 0) { usage(argv[0]); return 0; }

    /* Training files only. There is no code path here that names the t10k pair. */
    char ipath[1024], lpath[1024];
    if (join(ipath, sizeof(ipath), o.data_dir, "train-images.idx3-ubyte") ||
        join(lpath, sizeof(lpath), o.data_dir, "train-labels.idx1-ubyte")) {
        fprintf(stderr, "path too long\n");
        return 1;
    }

    dataset ds;
    if (dataset_load(&ds, ipath, lpath) != 0) {
        fprintf(stderr, "hint: run scripts/get_dataset.sh to fetch and verify the IDX files\n");
        return 1;
    }

    subset train_set, val_set;
    if (subset_split(&ds, o.val_count, o.split_seed, &train_set, &val_set) != 0) {
        dataset_free(&ds);
        return 1;
    }

    network *n = net_create(ds.pixels, o.hidden, 10);
    if (!n || net_load(n, o.weights) != 0) {
        fprintf(stderr, "%s: no usable weights -- train first\n", o.weights);
        net_free(n); subset_free(&train_set); subset_free(&val_set); dataset_free(&ds);
        return 1;
    }

    const size_t n1 = (size_t)n->hidden * (size_t)n->input;
    const size_t n2 = (size_t)n->output * (size_t)n->hidden;

    /* The unquantised weights, kept to restore between configurations and to
       serve as the reference W in ||WX - W_hat X||. */
    real *w1_ref  = (real *)malloc(n1 * sizeof(real));
    real *w2_ref  = (real *)malloc(n2 * sizeof(real));
    real *scratch = (real *)malloc((size_t)n->output * sizeof(real));
    int  *calib   = (int *)malloc((size_t)o.calib_n * sizeof(int));
    int  *hold    = (int *)malloc((size_t)(o.holdout_n > 0 ? o.holdout_n : 1) * sizeof(int));
    if (!w1_ref || !w2_ref || !scratch || !calib || !hold) { fprintf(stderr, "out of memory\n"); return 1; }

    memcpy(w1_ref, n->w1, n1 * sizeof(real));
    memcpy(w2_ref, n->w2, n2 * sizeof(real));

    /* Aborts if the draw is not purely from the training split. */
    if (quant_select_calibration(&train_set, &val_set, o.calib_n, o.calib_seed, calib) != 0) return 1;
    const int hold_n = select_holdout(&val_set, o.holdout_n, o.calib_seed, hold);

    double *x1 = NULL, *x2 = NULL, *hx1 = NULL, *hx2 = NULL;
    if (quant_collect_activations(n, &ds, calib, o.calib_n, &x1, &x2) != 0 ||
        (hold_n > 0 && quant_collect_activations(n, &ds, hold, hold_n, &hx1, &hx2) != 0)) {
        fprintf(stderr, "could not collect activations\n");
        return 1;
    }

    const real base_val = evaluate(n, &val_set, scratch);

    printf("network   %d-%d-%d from %s\n", n->input, n->hidden, n->output, o.weights);
    printf("split     %d train / %d validation (split seed %llu)\n",
           train_set.count, val_set.count, o.split_seed);
    printf("calib     %d samples from the training split (calib seed %llu), damping %g\n",
           o.calib_n, o.calib_seed, o.damping);
    printf("holdout   %d validation samples, for the same objective off-calibration\n", hold_n);
    printf("baseline  validation %.4f%%, %zu bytes of float32 weights\n\n",
           (double)base_val * 100.0, (n1 + n2 + (size_t)(n->hidden + n->output)) * sizeof(float));

    printf("%-5s %-5s %-5s  %11s %11s  %11s %11s  %9s %9s  %8s %9s\n",
           "meth", "bits", "layer",
           "calib_w1", "calib_w2", "hold_w1", "hold_w2",
           "maxd_w1", "maxd_w2", "val", "bytes");

    int status = 0;

    for (int li = 0; li < o.layers_n; li++)
    for (int mi = 0; mi < o.methods_n; mi++)
    for (int bi = 0; bi < o.bits_n; bi++) {
        const int mask = o.layers[li];

        quant_config cfg;
        cfg.method      = o.methods[mi];
        cfg.bits        = o.bits[bi];
        cfg.calib_n     = o.calib_n;
        cfg.damping     = o.damping;
        cfg.quantize_w1 = (mask & 1) != 0;
        cfg.quantize_w2 = (mask & 2) != 0;

        memcpy(n->w1, w1_ref, n1 * sizeof(real));
        memcpy(n->w2, w2_ref, n2 * sizeof(real));

        quant_report rep;
        const double t0 = now_seconds();
        if (quant_apply(n, &cfg, x1, x2, &rep) != 0) { status = 1; continue; }
        const double secs = now_seconds() - t0;

        double hold_w1 = 0.0, hold_w2 = 0.0;
        if (hold_n > 0) {
            if (cfg.quantize_w1)
                hold_w1 = quant_layer_error(w1_ref, n->w1, n->hidden, n->input, hx1, hold_n);
            if (cfg.quantize_w2)
                hold_w2 = quant_layer_error(w2_ref, n->w2, n->output, n->hidden, hx2, hold_n);
        }

        const real   val   = evaluate(n, &val_set, scratch);
        const size_t bytes = quant_packed_bytes(n, &cfg);

        char path[1024] = "";
        if (o.save) {
            char name[256];
            snprintf(name, sizeof(name), "%s.%s.b%d.%s.dat",
                     o.tag, quant_method_name(cfg.method), cfg.bits, layer_name(mask));
            if (join(path, sizeof(path), o.out_dir, name) == 0) {
                if (net_save(n, path) != 0) { fprintf(stderr, "could not write %s\n", path); status = 1; }
            }
        }

        printf("%-5s %-5d %-5s  %11.4e %11.4e  %11.4e %11.4e  %9.2e %9.2e  %7.4f%% %9zu\n",
               quant_method_name(cfg.method), cfg.bits, layer_name(mask),
               rep.layer_err_w1, rep.layer_err_w2, hold_w1, hold_w2,
               rep.max_abs_delta_w1, rep.max_abs_delta_w2,
               (double)val * 100.0, bytes);
        fflush(stdout);

        /* One machine-readable line per configuration, for scripts/quant_seeds.sh.
           Accuracies are fractions, as in ./mnist's RESULT lines; the table
           above is the one in percent. Errors are also given per calibration
           sample so that runs with different N are comparable. */
        printf("QRESULT method=%s bits=%d layers=%s calib=%d holdout=%d damping=%g "
               "err_w1=%.6e err_w2=%.6e err_w1_per=%.6e err_w2_per=%.6e "
               "hold_w1=%.6e hold_w2=%.6e hold_w1_per=%.6e hold_w2_per=%.6e "
               "maxd_w1=%.6e maxd_w2=%.6e val_acc=%.6f base_val_acc=%.6f bytes=%zu secs=%.2f weights=%s\n",
               quant_method_name(cfg.method), cfg.bits, layer_name(mask),
               o.calib_n, hold_n, o.damping,
               rep.layer_err_w1, rep.layer_err_w2,
               rep.layer_err_w1 / o.calib_n, rep.layer_err_w2 / o.calib_n,
               hold_w1, hold_w2,
               hold_n > 0 ? hold_w1 / hold_n : 0.0, hold_n > 0 ? hold_w2 / hold_n : 0.0,
               rep.max_abs_delta_w1, rep.max_abs_delta_w2,
               (double)val, (double)base_val, bytes, secs, path);
        fflush(stdout);
    }

    free(x1); free(x2); free(hx1); free(hx2);
    free(w1_ref); free(w2_ref); free(scratch); free(calib); free(hold);
    net_free(n);
    subset_free(&train_set);
    subset_free(&val_set);
    dataset_free(&ds);
    return status;
}
