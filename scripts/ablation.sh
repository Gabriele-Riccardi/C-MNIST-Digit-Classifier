#!/usr/bin/env bash
#
# Paired ablation on the validation split.
#
# A one-seed sweep cannot separate two configurations that differ by less than
# the seed-to-seed spread, and for augmentation that is exactly the situation.
# This runs both arms over the same seeds and reports the paired difference,
# which cancels most of the seed noise.
#
# Validation only. The test set is not opened.
#
#   usage: scripts/ablation.sh [seeds] [-- extra mnist options]
set -euo pipefail

cd "$(dirname "$0")/.."

SEEDS=${1:-5}
shift || true
if [ "${1:-}" = "--" ]; then shift; fi
EXTRA=("$@")

ARM_A=${ARM_A:---aug 0.0}
ARM_B=${ARM_B:---aug 0.5}

OUT=${OUT_DIR:-runs/ablation}
mkdir -p "$OUT"

make --no-print-directory mnist >/dev/null

echo "paired ablation over $SEEDS seeds, validation split only"
echo "  arm A: $ARM_A"
echo "  arm B: $ARM_B"
echo

: > "$OUT/ablation.tsv"
printf 'seed\tA\tB\tA_minus_B\n' >> "$OUT/ablation.tsv"

for ((s = 1; s <= SEEDS; s++)); do
    # shellcheck disable=SC2086
    a=$(./mnist train --seed "$s" --no-save $ARM_A ${EXTRA[@]+"${EXTRA[@]}"} \
        | grep -o 'val_acc=[-0-9.]*' | tail -1 | cut -d= -f2)
    # shellcheck disable=SC2086
    b=$(./mnist train --seed "$s" --no-save $ARM_B ${EXTRA[@]+"${EXTRA[@]}"} \
        | grep -o 'val_acc=[-0-9.]*' | tail -1 | cut -d= -f2)
    awk -v s="$s" -v a="$a" -v b="$b" \
        'BEGIN { printf "seed %d   A %.2f%%   B %.2f%%   A-B %+.2f\n", s, a*100, b*100, (a-b)*100 }'
    printf '%d\t%s\t%s\t%s\n' "$s" "$a" "$b" "$(awk -v a="$a" -v b="$b" 'BEGIN{print a-b}')" >> "$OUT/ablation.tsv"
done

echo
awk -F'\t' '
    NR > 1 { n++; d[n] = $4 * 100; a += $2 * 100; b += $3 * 100; sd += $4 * 100 }
    END {
        if (n < 2) { print "need at least two seeds"; exit 0 }
        md = sd / n
        for (i = 1; i <= n; i++) v += (d[i] - md)^2
        s = sqrt(v / (n - 1)); se = s / sqrt(n)
        printf "arm A mean %.2f%%   arm B mean %.2f%%\n", a / n, b / n
        printf "paired difference %+.2f pp, sd %.2f, standard error %.2f", md, s, se
        if (se > 0) printf ", t = %.2f on %d df", md / se, n - 1
        printf "\n"
    }
' "$OUT/ablation.tsv"
