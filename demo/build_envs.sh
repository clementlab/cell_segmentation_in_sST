#!/usr/bin/env bash
set -euo pipefail

# Verifies conda/mamba is available and builds the driver conda environment
# this repo's pipeline needs, from the pinned spec in analysis/envs/.
#
#   - snakemake_quick.yaml : driver env used to launch the Snakemake pipeline
#     (built as the "snakemake_env" name so it matches what
#     run_demo_pipeline.sh activates). Every package is pinned to an exact
#     version/build proven to co-install (derived from a known-working env),
#     since leaving packages like csbdeep/cellpose/tensorflow unconstrained
#     causes multi-hour or unsolvable conda searches.
#
# This spec mixes packages from nvidia/pytorch/conda-forge that share some
# package names (e.g. cuda-cudart, opencv) across channels as different
# builds. conda's default "strict" channel_priority setting (if enabled in
# ~/.condarc) silently overrides explicit per-package channel pins in that
# situation, causing solves to fail. CONDA_CHANNEL_PRIORITY=flexible is set
# below for this script's own conda/mamba calls only - it does not touch
# your global conda config.
#
# Note: envs/STP_env.yaml is NOT built here. Snakemake builds it itself
# (as a separate, hash-named env under --conda-prefix) the first time a rule
# with `conda: "envs/STP_env.yaml"` runs, via `snakemake --use-conda`. A
# named env built here would just be redundant extra build time - Snakemake
# would never use it.
#
# Usage: ./build_envs.sh
# Does not modify anything under analysis/envs/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVS_DIR="$SCRIPT_DIR/../analysis/envs"

export CONDA_CHANNEL_PRIORITY=flexible

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda not found on PATH. Install Miniconda/Miniforge first." >&2
    exit 1
fi

CONDA_FRONTEND="conda"
if command -v mamba >/dev/null 2>&1; then
    CONDA_FRONTEND="mamba"
fi
echo "Using $CONDA_FRONTEND to build environments"

env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

build_env() {
    local yaml_path="$1"
    local env_name="$2"

    if [ ! -f "$yaml_path" ]; then
        echo "Error: $yaml_path not found" >&2
        exit 1
    fi

    if env_exists "$env_name"; then
        echo "Environment '$env_name' already exists, updating from $yaml_path"
        "$CONDA_FRONTEND" env update -n "$env_name" -f "$yaml_path" --prune
    else
        echo "Creating environment '$env_name' from $yaml_path"
        "$CONDA_FRONTEND" env create -n "$env_name" -f "$yaml_path"
    fi
}

build_env "$ENVS_DIR/snakemake_quick.yaml" "snakemake_env"

echo "Verifying environment..."
if env_exists "snakemake_env"; then
    echo "  [OK] snakemake_env"
else
    echo "  [MISSING] snakemake_env" >&2
    exit 1
fi

echo "Environment built successfully: snakemake_env"
