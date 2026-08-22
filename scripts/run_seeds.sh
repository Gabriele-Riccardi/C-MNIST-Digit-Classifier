#!/usr/bin/env bash
#
# Runs the full protocol over several seeds and reports mean +/- standard
# deviation, because a single fixed-seed number says nothing about how stable
# it is: 98.4 +/- 0.05 and 98.4 +/- 0.4 are different claims.
#
# For each seed: train on the 54k training split, score the 6k validation split,
# then score the 10k test set once with the weights that run produced.
#
#   usage: scripts/run_seeds.sh [seeds] [-- extra mnist options]
#          scripts/run_seeds.sh 5
#          scripts/run_seeds.sh 5 -- --epochs 30 --hidden 256
#
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=${1:-5}
shift || true
if [ "${1:-}" = "--" ]; then shift; fi

# bash 3.2 (which is what macOS ships) treats "${EXTRA[@]}" as an unbound
# variable when the array is empty, so every expansion is guarded.
EXTRA=("$@")
extra() { printf '%s' "${EXTRA[*]-}"; }

OUT=${OUT_DIR:-runs}
mkdir -p "$OUT"

make --no-print-directory mnist >/dev/null

echo "protocol: train on the training split, tune on validation, touch the test set once per seed"
echo "seeds:    $SEEDS"
echo "options:  $( [ ${#EXTRA[@]} -gt 0 ] && extra || echo "<defaults>" )"
echo

: > "$OUT/results.tsv"
printf 'seed\tval_acc\ttest_acc\n' >> "$OUT/results.tsv"

for ((s = 1; s <= SEEDS; s++)); do
    weights="$OUT/network.seed$s.dat"
    log="$OUT/train.seed$s.log"

    echo "--- seed $s ---"
    ./mnist train --seed "$s" --weights "$weights" ${EXTRA[@]+"${EXTRA[@]}"} | tee "$log"

    val=$(grep -o 'val_acc=[-0-9.]*' "$log" | tail -1 | cut -d= -f2)
    test_line=$(./mnist test --weights "$weights" ${EXTRA[@]+"${EXTRA[@]}"} | tee -a "$log" | grep '^RESULT')
    tst=$(echo "$test_line" | grep -o 'test_acc=[0-9.]*' | cut -d= -f2)

    printf '%d\t%s\t%s\n' "$s" "$val" "$tst" >> "$OUT/results.tsv"
    echo
done

echo "=== summary over $SEEDS seeds ==="
awk -F'\t' '
    NR > 1 { n++; v[n] = $2; t[n] = $3; sv += $2; st += $3 }
    END {
        if (n < 1) { print "no results"; exit 1 }
        mv = sv / n; mt = st / n
        for (i = 1; i <= n; i++) { dv += (v[i] - mv)^2; dt += (t[i] - mt)^2 }
        # sample standard deviation (n-1): these are a sample of the seed
        # distribution, not the whole of it
        sdv = (n > 1) ? sqrt(dv / (n - 1)) : 0
        sdt = (n > 1) ? sqrt(dt / (n - 1)) : 0
        printf "validation  %.2f%% +/- %.2f  (n=%d)\n", mv * 100, sdv * 100, n
        printf "test        %.2f%% +/- %.2f  (n=%d)\n", mt * 100, sdt * 100, n
        printf "test range  %.2f%% .. ", 100 * min(t, n)
        printf "%.2f%%\n", 100 * max(t, n)
    }
    function min(a, k,   i, m) { m = a[1]; for (i = 2; i <= k; i++) if (a[i] < m) m = a[i]; return m }
    function max(a, k,   i, m) { m = a[1]; for (i = 2; i <= k; i++) if (a[i] > m) m = a[i]; return m }
' "$OUT/results.tsv"

echo
echo "per-seed results in $OUT/results.tsv"
