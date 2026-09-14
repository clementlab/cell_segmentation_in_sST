#!/usr/bin/env bash
set -euo pipefail

# Runs the mouse-brain cell segmentation pipeline end-to-end using only what
# the other demo scripts prepared - no machine-specific (CHPC) paths.
#
# Prerequisites (run once, from this directory):
#   ./get.sh          -> downloads data into DATA_DIR      (default: mouse_brain)
#   ./get_stp.sh       -> clones STP into STP_DIR           (default: third_party/STP)
#   ./build_envs.sh    -> creates the snakemake_env conda env
#
# Usage: ./run_demo_pipeline.sh [cores]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/mouse_brain}"
STP_DIR="${STP_DIR:-$SCRIPT_DIR/third_party/STP}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/run_output}"
CORES="${1:-4}"

if [ ! -f "$DATA_DIR/Visium_HD_Mouse_Brain_tissue_image.tif" ]; then
    echo "Error: data not found in $DATA_DIR. Run ./get.sh first." >&2
    exit 1
fi

if [ ! -d "$STP_DIR" ]; then
    echo "Error: STP source not found at $STP_DIR. Run ./get_stp.sh first." >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found on PATH." >&2
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
# Some conda environments' activation hooks (e.g. libblas_mkl_activate.sh)
# reference variables like MKL_INTERFACE_LAYER without a default, which
# trips `set -u`. Relax nounset just for the activation call.
set +u
conda activate snakemake_env
set -u
echo "Activated conda environment: $(conda info | grep 'active environment')"

CONDA_FRONTEND="conda"
if command -v mamba >/dev/null 2>&1; then
    CONDA_FRONTEND="mamba"
fi

export STP_PATH="$STP_DIR"

# envs/STP_env.yaml mixes packages from conda-forge and defaults that share
# some names as different builds (e.g. pyogrio). Snakemake's --use-conda
# calls `mamba env create` internally to build that env on first use, and
# conda's default "strict" channel_priority (if set in ~/.condarc) silently
# overrides the explicit per-package channel pins in that situation, making
# the solve fail outright. This override is scoped to this script's own
# snakemake subprocess only - it does not touch your global conda config.
export CONDA_CHANNEL_PRIORITY=flexible

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

snakemake -s "$REPO_ROOT/analysis/pipeline.smk" \
    --use-conda \
    --conda-frontend "$CONDA_FRONTEND" \
    --conda-prefix "$SCRIPT_DIR/.snakemake_conda" \
    --cores "$CORES" \
    --config \
    region_x="[14000, 16000]" region_y="[6500, 8500]" \
    image_path="$DATA_DIR/Visium_HD_Mouse_Brain_tissue_image.tif" \
    square_002um="$DATA_DIR/binned_outputs/square_002um" \
    spatial="$DATA_DIR/binned_outputs/square_002um/spatial" \
    matrix_path="$DATA_DIR/binned_outputs/square_002um/filtered_feature_bc_matrix.h5" \
    positions_path="$DATA_DIR/binned_outputs/square_002um/spatial/tissue_positions.parquet" \
    2>&1 | tee run.log

echo "Demo pipeline run complete. Output in $RUN_DIR, log at $RUN_DIR/run.log"
