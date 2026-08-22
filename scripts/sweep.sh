#!/usr/bin/env bash
#
# The hyperparameter search, on the validation split only.
#
# This exists so that "lr 0.01, 128 hidden units, augmentation on half the
# samples" is an answerable question rather than a set of numbers that appeared
# in the source one day. Every run here reads validation accuracy; none of them
# opens the test files. The test set is scored once, at the end, by
# scripts/run_seeds.sh.
#
#   usage: scripts/sweep.sh [epochs]     (default 10 -- long enough to rank,
#                                         short enough to run)
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS=${1:-10}
SEED=${SEED:-1}
OUT=${OUT_DIR:-runs/sweep}
mkdir -p "$OUT"

make --no-print-directory mnist >/dev/null

run() {
    local tag=$1; shift
    local log="$OUT/$tag.log"
    ./mnist train --epochs "$EPOCHS" --seed "$SEED" --no-save "$@" > "$log" 2>&1
    local acc
    acc=$(grep -o 'val_acc=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
    printf '%-22s %-24s %s\n' "$tag" "$*" "$(awk -v a="$acc" 'BEGIN { printf "%.2f%%", a * 100 }')"
    printf '%s\t%s\t%s\n' "$tag" "$*" "$acc" >> "$OUT/sweep.tsv"
}

: > "$OUT/sweep.tsv"

echo "validation sweep: $EPOCHS epochs, seed $SEED, 54000/6000 split"
echo "the test set is not opened by anything in this script"
echo

echo "-- learning rate (128 hidden units, augmentation off) --"
run "lr-0.003"  --lr 0.003
run "lr-0.01"   --lr 0.01
run "lr-0.03"   --lr 0.03
run "lr-0.1"    --lr 0.1

echo
echo "-- hidden units (lr 0.01, augmentation off) --"
run "hidden-64"   --hidden 64
run "hidden-256"  --hidden 256

echo
echo "-- augmentation, +/-1 px translation (128 hidden units, lr 0.01) --"
run "aug-0.0"  --aug 0.0
run "aug-0.5"  --aug 0.5
run "aug-1.0"  --aug 1.0
echo "   (one seed each: scripts/ablation.sh settles this one across seeds)"

echo
echo "results in $OUT/sweep.tsv"
