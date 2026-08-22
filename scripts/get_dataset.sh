#!/usr/bin/env bash
#
# Fetches the four MNIST IDX files and verifies them against known SHA-256
# digests. "Reproducible" is not a property of a README sentence telling you to
# go and find the dataset yourself: the original yann.lecun.com URLs have been
# unreliable for years, and a half-downloaded file is a much worse failure than
# a missing one because it still parses.
#
# The digests below are of the DECOMPRESSED IDX files, so they hold whichever
# mirror served the archive and however it was gzipped.
#
#   usage: scripts/get_dataset.sh [target-dir]     (default: dataset)
#
set -euo pipefail

DEST=${1:-dataset}

MIRRORS=(
    "https://ossci-datasets.s3.amazonaws.com/mnist"
    "https://storage.googleapis.com/cvdf-datasets/mnist"
)

# local name : archive name : sha256 of the decompressed file : expected bytes
FILES=(
    "train-images.idx3-ubyte:train-images-idx3-ubyte.gz:ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db:47040016"
    "train-labels.idx1-ubyte:train-labels-idx1-ubyte.gz:65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5:60008"
    "t10k-images.idx3-ubyte:t10k-images-idx3-ubyte.gz:0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7:7840016"
    "t10k-labels.idx1-ubyte:t10k-labels-idx1-ubyte.gz:ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2:10008"
)

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum   >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
    else echo "need sha256sum or shasum" >&2; exit 1
    fi
}

size_of() {
    if stat -f%z "$1" >/dev/null 2>&1; then stat -f%z "$1"   # BSD / macOS
    else stat -c%s "$1"                                      # GNU
    fi
}

fetch() {
    local url=$1 out=$2
    if   command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then wget -q -O "$out" "$url"
    else echo "need curl or wget" >&2; exit 1
    fi
}

mkdir -p "$DEST"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

ok=0
for entry in "${FILES[@]}"; do
    IFS=: read -r name archive want_sha want_size <<< "$entry"
    target="$DEST/$name"

    if [ -f "$target" ] && [ "$(sha256_of "$target")" = "$want_sha" ]; then
        echo "ok        $name (already present and verified)"
        ok=$((ok + 1))
        continue
    fi

    [ -f "$target" ] && echo "warning   $name is present but does not match its digest -- refetching"

    got=""
    for mirror in "${MIRRORS[@]}"; do
        echo "fetching  $mirror/$archive"
        if fetch "$mirror/$archive" "$tmp/$archive"; then got=$mirror; break; fi
        echo "          mirror failed, trying the next one"
    done

    if [ -z "$got" ]; then
        echo "error     could not download $archive from any mirror" >&2
        exit 1
    fi

    gunzip -c "$tmp/$archive" > "$tmp/$name"

    have_size=$(size_of "$tmp/$name")
    have_sha=$(sha256_of "$tmp/$name")

    if [ "$have_size" != "$want_size" ]; then
        echo "error     $name is $have_size bytes, expected $want_size" >&2
        exit 1
    fi
    if [ "$have_sha" != "$want_sha" ]; then
        echo "error     $name failed its checksum" >&2
        echo "          expected $want_sha" >&2
        echo "          got      $have_sha" >&2
        exit 1
    fi

    mv "$tmp/$name" "$target"
    echo "verified  $name  ($have_size bytes, sha256 ${have_sha:0:16}...)"
    ok=$((ok + 1))
done

echo
echo "$ok/4 files present and verified in $DEST/"
