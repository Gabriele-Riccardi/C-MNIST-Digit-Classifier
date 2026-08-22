#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "idx.h"
#include "net.h"
#include "train.h"
#include "util.h"
#include "prng.h"

#define COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

/*
 * Two subcommands, deliberately separate:
 *
 *   train   fits on a 54k slice of the training set and reports accuracy on the
 *           6k held out from it. It never opens the test files.
 *   test    loads saved weights and scores them on the 10k test set, once.
 *
 * Keeping them apart is the point. Every hyperparameter below was chosen by
 * looking at validation accuracy; if `train` could print a test number there
 * would be nothing stopping that number from leaking into the choice.
 */

#define DEFAULT_DATA_DIR   "dataset"
#define DEFAULT_WEIGHTS    "network.dat"
#define DEFAULT_VAL_COUNT  6000
#define DEFAULT_SPLIT_SEED 20260822ULL

typedef struct {
    const char        *data_dir;
    const char        *weights;
    int                epochs;
    int                hidden;
    real               lr;
    real               aug_prob;
    int                max_shift;
    int                val_count;
    unsigned long long seed;
    unsigned long long split_seed;
    int                resume;
    int                save;
} options;

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s <command> [options]\n"
        "\n"
        "commands:\n"
        "  train            fit on the training split, score on the validation split\n"
        "  test             load saved weights and score them on the test set (final measurement)\n"
        "\n"
        "options:\n"
        "  --epochs N       training epochs (default 20)\n"
        "  --lr F           learning rate (default 0.01)\n"
        "  --hidden N       hidden units (default 128)\n"
        "  --aug F          probability of translating a training sample (default 0, see README)\n"
        "  --shift N        maximum translation in pixels (default 1)\n"
        "  --val N          validation samples held out of the training set (default 6000)\n"
        "  --seed N         seed for initialisation, shuffling and augmentation (default 1)\n"
        "  --split-seed N   seed for the train/validation split (default %llu)\n"
        "  --data DIR       directory holding the four IDX files (default \"%s\")\n"
        "  --weights FILE   weight file to write, or read in test mode (default \"%s\")\n"
        "  --resume         start training from the existing weight file\n"
        "  --no-save        do not write the weight file\n",
        argv0, DEFAULT_SPLIT_SEED, DEFAULT_DATA_DIR, DEFAULT_WEIGHTS);
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    const long v = strtol(s, &end, 10);
    if (!end || *end != '\0' || v < 0 || v > 1000000000L) return -1;
    *out = (int)v;
    return 0;
}

static int parse_u64(const char *s, unsigned long long *out) {
    char *end = NULL;
    const unsigned long long v = strtoull(s, &end, 10);
    if (!end || *end != '\0') return -1;
    *out = v;
    return 0;
}

static int parse_real(const char *s, real *out) {
    char *end = NULL;
    const double v = strtod(s, &end);
    if (!end || *end != '\0') return -1;
    *out = (real)v;
    return 0;
}

/* name, type, destination. 'i' int, 'r' real, 'u' unsigned long long, 's' string. */
static int parse_options(int argc, char **argv, options *o) {
    const struct { const char *name; char kind; void *dst; } table[] = {
        { "--epochs",     'i', &o->epochs     },
        { "--hidden",     'i', &o->hidden     },
        { "--shift",      'i', &o->max_shift  },
        { "--val",        'i', &o->val_count  },
        { "--lr",         'r', &o->lr         },
        { "--aug",        'r', &o->aug_prob   },
        { "--seed",       'u', &o->seed       },
        { "--split-seed", 'u', &o->split_seed },
        { "--data",       's', &o->data_dir   },
        { "--weights",    's', &o->weights    },
    };

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];

        if (!strcmp(a, "-h") || !strcmp(a, "--help")) return 1;
        if (!strcmp(a, "--resume"))  { o->resume = 1; continue; }
        if (!strcmp(a, "--no-save")) { o->save   = 0; continue; }

        size_t k = 0;
        while (k < COUNT_OF(table) && strcmp(a, table[k].name) != 0) k++;
        if (k == COUNT_OF(table)) {
            fprintf(stderr, "unknown option: %s\n", a);
            return -1;
        }

        if (++i >= argc) {
            fprintf(stderr, "%s needs a value\n", a);
            return -1;
        }

        const char *v  = argv[i];
        int         ok = 0;
        switch (table[k].kind) {
            case 'i': ok = parse_int (v, (int *)table[k].dst)                == 0; break;
            case 'r': ok = parse_real(v, (real *)table[k].dst)               == 0; break;
            case 'u': ok = parse_u64 (v, (unsigned long long *)table[k].dst) == 0; break;
            case 's': *(const char **)table[k].dst = v; ok = 1;                    break;
        }

        if (!ok) {
            fprintf(stderr, "bad value for %s: %s\n", a, v);
            return -1;
        }
    }

    if (o->aug_prob < (real)0.0 || o->aug_prob > (real)1.0) {
        fprintf(stderr, "--aug must be in [0,1]\n");
        return -1;
    }
    if (o->hidden <= 0 || o->epochs < 0) {
        fprintf(stderr, "--hidden must be positive and --epochs non-negative\n");
        return -1;
    }
    return 0;
}

static int join(char *dst, size_t cap, const char *dir, const char *name) {
    const int written = snprintf(dst, cap, "%s/%s", dir, name);
    return (written < 0 || (size_t)written >= cap) ? -1 : 0;
}

static int load_split(const options *o, const char *images, const char *labels, dataset *ds) {
    char ipath[1024], lpath[1024];
    if (join(ipath, sizeof(ipath), o->data_dir, images) ||
        join(lpath, sizeof(lpath), o->data_dir, labels)) {
        fprintf(stderr, "path too long\n");
        return -1;
    }
    return dataset_load(ds, ipath, lpath);
}

static int cmd_train(const options *o) {
    dataset ds;
    if (load_split(o, "train-images.idx3-ubyte", "train-labels.idx1-ubyte", &ds) != 0) {
        fprintf(stderr, "hint: run scripts/get_dataset.sh to fetch and verify the IDX files\n");
        return 1;
    }

    subset train_set, val_set;
    if (subset_split(&ds, o->val_count, o->split_seed, &train_set, &val_set) != 0) {
        dataset_free(&ds);
        return 1;
    }

    printf("data     %d images, %dx%d\n", ds.count, ds.rows, ds.cols);
    printf("split    %d train / %d validation (split seed %llu)\n",
           train_set.count, val_set.count, o->split_seed);
    printf("network  %d-%d-%d, %s\n", ds.pixels, o->hidden, 10, REAL_NAME);
    printf("run      seed %llu, %d epochs, lr %g, aug %g at +/-%d px\n",
           o->seed, o->epochs, (double)o->lr, (double)o->aug_prob, o->max_shift);

    /* Seeded here, after the split: the split has its own stream so that
       changing --seed reshuffles training without moving the hold-out set. */
    prng_seed(o->seed, 1ULL);

    network *n = net_create(ds.pixels, o->hidden, 10);
    if (!n) { subset_free(&train_set); subset_free(&val_set); dataset_free(&ds); return 1; }
    net_init_weights(n);

    if (o->resume) {
        if (net_load(n, o->weights) == 0) printf("resumed from %s\n", o->weights);
        else                              printf("no usable %s, starting from scratch\n", o->weights);
    }

    trainer t;
    real   *scratch = (real *)malloc((size_t)n->output * sizeof(real));
    if (trainer_init(&t, n, &train_set) != 0 || !scratch) {
        fprintf(stderr, "out of memory\n");
        free(scratch); trainer_free(&t); net_free(n);
        subset_free(&train_set); subset_free(&val_set); dataset_free(&ds);
        return 1;
    }

    const train_config cfg = { o->epochs, o->lr, o->aug_prob, o->max_shift };

    real   best_val = (real)0.0;
    int    best_epoch = 0;
    double total_time = 0.0;

    for (int e = 1; e <= o->epochs; e++) {
        const double t0   = now_seconds();
        const real   loss = train_epoch(n, &train_set, &cfg, &t);
        const double secs = now_seconds() - t0;
        total_time += secs;

        if (val_set.count > 0) {
            const real val_acc = evaluate(n, &val_set, scratch);
            if (val_acc > best_val) { best_val = val_acc; best_epoch = e; }
            printf("epoch %2d  loss %.6f  val %.4f%%  %.2f s\n",
                   e, (double)loss, (double)val_acc * 100.0, secs);
        } else {
            printf("epoch %2d  loss %.6f  val n/a      %.2f s\n", e, (double)loss, secs);
        }
        fflush(stdout);
    }

    /* --val 0 trains on the whole set: legitimate as a last step once the
       hyperparameters are settled, but then there is nothing to report. */
    const real final_val = (val_set.count > 0) ? evaluate(n, &val_set, scratch) : (real)-1.0;

    if (o->save) {
        if (net_save(n, o->weights) == 0) printf("weights written to %s\n", o->weights);
        else                              fprintf(stderr, "could not write %s\n", o->weights);
    }

    if (o->epochs > 0) {
        if (val_set.count > 0)
            printf("best validation %.4f%% at epoch %d, %.2f s/epoch\n",
                   (double)best_val * 100.0, best_epoch, total_time / o->epochs);
        else
            printf("no validation split, %.2f s/epoch\n", total_time / o->epochs);
    }

    /* One machine-readable line, for scripts/run_seeds.sh. */
    printf("RESULT seed=%llu split=%d/%d epochs=%d hidden=%d lr=%g aug=%g val_acc=%.6f\n",
           o->seed, train_set.count, val_set.count, o->epochs, o->hidden,
           (double)o->lr, (double)o->aug_prob, (double)final_val);

    free(scratch);
    trainer_free(&t);
    net_free(n);
    subset_free(&train_set);
    subset_free(&val_set);
    dataset_free(&ds);
    return 0;
}

static int cmd_test(const options *o) {
    dataset ds;
    if (load_split(o, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &ds) != 0) {
        fprintf(stderr, "hint: run scripts/get_dataset.sh to fetch and verify the IDX files\n");
        return 1;
    }

    subset all;
    if (subset_all(&ds, &all) != 0) { dataset_free(&ds); return 1; }

    network *n = net_create(ds.pixels, o->hidden, 10);
    if (!n) { subset_free(&all); dataset_free(&ds); return 1; }

    if (net_load(n, o->weights) != 0) {
        fprintf(stderr, "%s: no usable weights -- train first\n", o->weights);
        net_free(n); subset_free(&all); dataset_free(&ds);
        return 1;
    }

    real *scratch = (real *)malloc((size_t)n->output * sizeof(real));
    if (!scratch) { net_free(n); subset_free(&all); dataset_free(&ds); return 1; }

    const real acc = evaluate(n, &all, scratch);
    printf("test set %d images from %s\n", ds.count, o->data_dir);
    printf("RESULT test_acc=%.6f errors=%d/%d\n",
           (double)acc, (int)((1.0 - (double)acc) * ds.count + 0.5), ds.count);

    free(scratch);
    net_free(n);
    subset_free(&all);
    dataset_free(&ds);
    return 0;
}

int main(int argc, char **argv) {
    options o = {
        DEFAULT_DATA_DIR, DEFAULT_WEIGHTS,
        20, 128, (real)0.01, (real)0.0, 1,
        DEFAULT_VAL_COUNT, 1ULL, DEFAULT_SPLIT_SEED,
        0, 1
    };

    if (argc < 2) { usage(argv[0]); return 2; }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) { usage(argv[0]); return 0; }

    const int parsed = parse_options(argc, argv, &o);
    if (parsed < 0) return 2;
    if (parsed > 0) { usage(argv[0]); return 0; }

    if      (!strcmp(argv[1], "train")) return cmd_train(&o);
    else if (!strcmp(argv[1], "test"))  return cmd_test(&o);

    fprintf(stderr, "unknown command: %s\n", argv[1]);
    usage(argv[0]);
    return 2;
}
