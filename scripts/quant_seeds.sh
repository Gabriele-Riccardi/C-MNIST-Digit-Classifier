#!/usr/bin/env bash
#
# The quantisation protocol, over several seeds.
#
# For each seed: train a baseline (or reuse one already in $OUT), score it once
# on the test set, then quantise it under every configuration in the grid and
# score each of those once. Quantisation itself never opens the test files --
# build/quant_sweep loads the training pair and nothing else, and reports
# validation accuracy. The test number comes from ./mnist test afterwards, here,
# in the same place and the same way it does for an unquantised network.
#
# That separation matters more here than in the baseline protocol. The main grid
# alone is ten configurations per seed; with the ablations it is thirty. Choosing
# a bit width by test accuracy would be an easy accident to have, and would turn
# the test set into a validation set with thirty chances to fit it.
#
# Three experiments, selected by $EXPERIMENTS (default: all three):
#
#   main    bits x method, both layers quantised
#   layers  w1 only / w2 only, at 4 and 2 bits -- w2 is 1.3% of the parameters,
#           so whether it dominates the damage is the question
#   calib   calibration sizes 128 / 512 / 2048, at 4 and 3 bits: how many
#           samples X X^T needs before it is an estimate of something
#
#   usage: scripts/quant_seeds.sh [seeds] [-- extra mnist train options]
#          EXPERIMENTS=main scripts/quant_seeds.sh 5
#
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=${1:-5}
shift || true
if [ "${1:-}" = "--" ]; then shift; fi
EXTRA=("$@")

OUT=${OUT_DIR:-runs/quant}
CALIB=${CALIB:-512}
HOLDOUT=${HOLDOUT:-2048}
EXPERIMENTS=${EXPERIMENTS:-main layers calib}
TSV="$OUT/quant.tsv"

mkdir -p "$OUT"
make --no-print-directory mnist build/quant_sweep >/dev/null

echo "quantisation protocol over $SEEDS seeds"
echo "  experiments:  $EXPERIMENTS"
echo "  calibration:  $CALIB samples from the training split; the objective is also"
echo "                scored on $HOLDOUT validation samples the calibration never saw"
echo "  the test set is read once per configuration, by ./mnist test, and never by the quantiser"
echo

printf 'seed\texp\tmethod\tbits\tlayers\tcalib\terr_w1\terr_w2\thold_w1\thold_w2\tmaxd_w1\tmaxd_w2\tval_acc\ttest_acc\tbytes\n' > "$TSV"

# One "key=value key=value ..." line in, one value out.
field() { echo "$1" | tr ' ' '\n' | grep "^$2=" | cut -d= -f2- ; }

pct() { awk -v v="$1" 'BEGIN { printf "%.2f%%", v * 100 }'; }

# $1 experiment name, $2 calibration size, rest: quant_sweep options.
# Runs the sweep, then scores each weight file it wrote on the test set.
sweep_and_score() {
    local exp=$1 calib=$2
    shift 2
    local log="$OUT/sweep.$exp.seed$SEED.log"
    local line path tst

    ./build/quant_sweep \
        --weights "$WEIGHTS" --out "$OUT" --tag "$exp.seed$SEED" \
        --calib "$calib" --calib-seed "$SEED" --holdout "$HOLDOUT" \
        "$@" | tee "$log" | grep -v '^QRESULT' || true

    while IFS= read -r line; do
        path=$(field "$line" weights)
        [ -n "$path" ] || continue

        tst=$(./mnist test --weights "$path" | grep -o 'test_acc=[0-9.]*' | cut -d= -f2)

        printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$SEED" "$exp" \
            "$(field "$line" method)" "$(field "$line" bits)" \
            "$(field "$line" layers)" "$(field "$line" calib)" \
            "$(field "$line" err_w1_per)" "$(field "$line" err_w2_per)" \
            "$(field "$line" hold_w1_per)" "$(field "$line" hold_w2_per)" \
            "$(field "$line" maxd_w1)" "$(field "$line" maxd_w2)" \
            "$(field "$line" val_acc)" "$tst" \
            "$(field "$line" bytes)" >> "$TSV"
    done < <(grep '^QRESULT' "$log")
}

for ((SEED = 1; SEED <= SEEDS; SEED++)); do
    echo "=== seed $SEED ==="
    WEIGHTS="$OUT/network.seed$SEED.dat"

    if [ ! -f "$WEIGHTS" ]; then
        ./mnist train --seed "$SEED" --weights "$WEIGHTS" ${EXTRA[@]+"${EXTRA[@]}"} \
            > "$OUT/train.seed$SEED.log" 2>&1
        echo "trained $WEIGHTS"
    else
        echo "reusing $WEIGHTS"
    fi

    # The unquantised reference for this seed, scored the same way as everything
    # else so the comparison is like for like. Its size is the weight file minus
    # its 16-byte header, which is the float32 parameters and nothing else.
    base_val=$(grep -o 'val_acc=[0-9.]*' "$OUT/train.seed$SEED.log" | tail -1 | cut -d= -f2)
    base_test=$(./mnist test --weights "$WEIGHTS" | grep -o 'test_acc=[0-9.]*' | cut -d= -f2)
    base_bytes=$(( $(wc -c < "$WEIGHTS") - 16 ))

    printf '%d\tbase\tfp32\t32\tnone\t0\t0\t0\t0\t0\t0\t0\t%s\t%s\t%d\n' \
        "$SEED" "$base_val" "$base_test" "$base_bytes" >> "$TSV"
    echo "baseline  val $(pct "$base_val")  test $(pct "$base_test")  $base_bytes bytes"

    case " $EXPERIMENTS " in *" main "*)
        sweep_and_score main "$CALIB" --bits 8,6,4,3,2 --method rtn,gptq --layers both ;;
    esac
    case " $EXPERIMENTS " in *" layers "*)
        sweep_and_score layers "$CALIB" --bits 4,2 --method rtn,gptq --layers w1,w2 ;;
    esac
    case " $EXPERIMENTS " in *" calib "*)
        for cn in 128 512 2048; do
            sweep_and_score "calib$cn" "$cn" --bits 4,3 --method rtn,gptq --layers both
        done ;;
    esac
    echo
done

echo "=== summary over $SEEDS seeds ==="
echo "errors are per calibration sample, so rows with different --calib compare directly"
echo

awk -F'\t' '
    NR == 1 { next }
    {
        key = $2 "|" $3 "|" $4 "|" $5 "|" $6
        if (!(key in cnt)) order[++k] = key
        cnt[key]++
        te[key] += $14; tesq[key] += $14 * $14
        va[key] += $13
        e1[key] += $7;  e2[key] += $8
        h1[key] += $9;  h2[key] += $10
        by[key]  = $15
    }
    END {
        printf "%-8s %-6s %-4s %-6s %-6s  %11s %11s  %11s %11s  %8s  %16s  %8s\n",
               "exp", "method", "bits", "layers", "calib",
               "calib_w1", "calib_w2", "hold_w1", "hold_w2", "val", "test", "bytes"
        for (i = 1; i <= k; i++) {
            key = order[i]
            split(key, f, "|")
            c  = cnt[key]
            m  = te[key] / c
            sd = (c > 1) ? sqrt((tesq[key] - c * m * m) / (c - 1)) : 0
            printf "%-8s %-6s %-4s %-6s %-6s  %11.4e %11.4e  %11.4e %11.4e  %7.2f%%  %7.2f%% +/- %.2f  %8d\n",
                   f[1], f[2], f[3], f[4], f[5],
                   e1[key]/c, e2[key]/c, h1[key]/c, h2[key]/c,
                   100 * va[key]/c, 100 * m, 100 * sd, by[key]
        }
    }
' "$TSV"

echo
echo "per-run rows in $TSV"
