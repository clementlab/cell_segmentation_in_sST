#!/usr/bin/env bash
set -euo pipefail

# One-command entry point for the cell-segmentation demo.
# Runs, in order:
#   1. get.sh                -> download the Visium HD Mouse Brain dataset
#   2. get_stp.sh             -> clone STP (pinned commit)
#   3. build_envs.sh          -> create the conda environments
#   4. run_demo_pipeline.sh   -> run the pipeline end-to-end
#
# Usage: ./run_demo.sh [--allow-cpu] [cores]
# All steps are idempotent: rerunning this script skips work already done.
#
# This pipeline expects an NVIDIA GPU (torch, tensorflow, cellpose, and
# stardist are all used with CUDA). Without one it will error out or run
# unusably slowly, so run_demo.sh checks for a GPU up front and stops unless
# --allow-cpu is passed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ALLOW_CPU=0
CORES=4
for arg in "$@"; do
    case "$arg" in
        --allow-cpu)
            ALLOW_CPU=1
            ;;
        *)
            CORES="$arg"
            ;;
    esac
done

BOLD="$(tput bold 2>/dev/null || true)"
RESET="$(tput sgr0 2>/dev/null || true)"
GREEN="$(tput setaf 2 2>/dev/null || true)"
RED="$(tput setaf 1 2>/dev/null || true)"

step() {
    echo
    echo "${BOLD}==> $1${RESET}"
}

ok() {
    echo "${GREEN}[OK]${RESET} $1"
}

fail() {
    echo "${RED}[FAILED]${RESET} $1" >&2
    echo "See the error above for details. Fix the issue and rerun ./run_demo.sh -- completed steps will be skipped." >&2
    exit 1
}

on_error() {
    local msg="run_demo.sh stopped early (step: ${CURRENT_STEP:-unknown})"
    local run_log="$SCRIPT_DIR/run_output/run.log"
    if [ "${CURRENT_STEP:-}" = "running the pipeline (run_demo_pipeline.sh)" ] && [ -f "$run_log" ]; then
        msg="$msg
       Snakemake reports errors per-rule, not just pipeline-wide - check $run_log
       for the specific rule name, wildcards, and log path it points to."
    fi
    fail "$msg"
}

trap on_error ERR

echo "${BOLD}Cell segmentation demo${RESET}"
echo "Working directory: $SCRIPT_DIR"
echo "Cores for pipeline run: $CORES"

CURRENT_STEP="checking prerequisites"
step "Checking prerequisites"
for cmd in curl tar git conda; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        fail "'$cmd' is required but not found on PATH. Install it and rerun."
    fi
done
ok "curl, tar, git, and conda are all available"

# snakemake_env.yaml pins pytorch-cuda=12.1, which needs a driver new enough
# to support CUDA 12.1 (NVIDIA's compatibility table lists 530.30.02 as the
# minimum on Linux). This is a warning, not a hard stop: driver/toolkit
# backward compatibility has enough edge cases that an older driver may
# still work, but this flags the most common source of confusing CUDA
# errors deep inside a pipeline run.
MIN_CUDA121_DRIVER="530.30.02"

CURRENT_STEP="checking for a GPU"
step "Checking for a GPU"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    ok "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/       /'

    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    if [ "$(printf '%s\n%s\n' "$MIN_CUDA121_DRIVER" "$DRIVER_VERSION" | sort -V | head -n1)" = "$MIN_CUDA121_DRIVER" ]; then
        ok "Driver version $DRIVER_VERSION supports CUDA 12.1 (min $MIN_CUDA121_DRIVER)"
    else
        echo "${RED}[WARNING]${RESET} Driver version $DRIVER_VERSION is older than $MIN_CUDA121_DRIVER, the minimum"
        echo "          NVIDIA lists for CUDA 12.1 (used by snakemake_env.yaml's pytorch-cuda=12.1)."
        echo "          The pipeline may still work, but CUDA errors from torch/tensorflow are more likely."
        echo "          Consider updating your NVIDIA driver if you hit CUDA initialization errors."
    fi
elif [ "$ALLOW_CPU" -eq 1 ]; then
    echo "${RED}[WARNING]${RESET} No NVIDIA GPU detected (nvidia-smi not found or failed)."
    echo "          Continuing anyway because --allow-cpu was passed."
    echo "          torch/tensorflow/cellpose/stardist steps may be extremely slow or fail outright on CPU."
else
    fail "No NVIDIA GPU detected (nvidia-smi not found or failed).
       This pipeline uses GPU-accelerated torch, tensorflow, cellpose, and stardist and is not expected to
       work correctly on CPU only. If you understand the risk and want to try anyway, rerun with:
           ./run_demo.sh --allow-cpu [cores]"
fi

CURRENT_STEP="downloading data (get.sh)"
step "Step 1/4: Downloading demo dataset"
if [ -f "mouse_brain/Visium_HD_Mouse_Brain_tissue_image.tif" ]; then
    ok "Dataset already present in mouse_brain/, skipping download"
else
    ./get.sh
    ok "Dataset downloaded to mouse_brain/"
fi

CURRENT_STEP="cloning STP (get_stp.sh)"
step "Step 2/4: Fetching STP source"
./get_stp.sh
ok "STP source ready in third_party/STP"

CURRENT_STEP="building conda environments (build_envs.sh)"
step "Step 3/4: Building conda environments"
echo "This step can take several minutes the first time it runs."
./build_envs.sh
ok "Conda environments (snakemake_env, STP_env) are ready"

CURRENT_STEP="running the pipeline (run_demo_pipeline.sh)"
step "Step 4/4: Running the pipeline"
./run_demo_pipeline.sh "$CORES"
ok "Pipeline run complete"

trap - ERR

echo
echo "${GREEN}${BOLD}Demo finished successfully.${RESET}"
echo "Results are in: $SCRIPT_DIR/run_output"
echo "Full log: $SCRIPT_DIR/run_output/run.log"
