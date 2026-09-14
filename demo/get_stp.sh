#!/usr/bin/env bash
set -euo pipefail

# Clones the STP source (https://github.com/leihouyeung/STP) into this demo,
# pinned to a specific commit for reproducibility. The upstream repo has had
# no updates since 2024-09-11, so this pins to its current HEAD.
#
# Usage: ./get_stp.sh [destination_dir]

STP_REPO_URL="https://github.com/leihouyeung/STP.git"
STP_COMMIT="4e56f8dcea5a42d9a9089093bfa203d1d193b10c"

DEST_DIR="${1:-third_party/STP}"

if [ -d "$DEST_DIR/.git" ]; then
    current_commit="$(git -C "$DEST_DIR" rev-parse HEAD)"
    if [ "$current_commit" = "$STP_COMMIT" ]; then
        echo "STP already present at $DEST_DIR (commit $STP_COMMIT)"
        exit 0
    else
        echo "Error: $DEST_DIR exists but is at commit $current_commit, expected $STP_COMMIT" >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "$DEST_DIR")"
git clone "$STP_REPO_URL" "$DEST_DIR"
git -C "$DEST_DIR" checkout "$STP_COMMIT"

checked_out_commit="$(git -C "$DEST_DIR" rev-parse HEAD)"
if [ "$checked_out_commit" != "$STP_COMMIT" ]; then
    echo "Error: checked out commit $checked_out_commit does not match expected $STP_COMMIT" >&2
    exit 1
fi

echo "Cloned STP (commit $STP_COMMIT) into $DEST_DIR"
