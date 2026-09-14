#!/usr/bin/env bash
set -euo pipefail

# Downloads the 10x Genomics Visium HD Mouse Brain demo dataset used by this repo's demo.
# Usage: ./get.sh [destination_dir]

DEST_DIR="${1:-mouse_brain}"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_binned_outputs.tar.gz
curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_tissue_image.tif

tar -xvzf Visium_HD_Mouse_Brain_binned_outputs.tar.gz
rm Visium_HD_Mouse_Brain_binned_outputs.tar.gz

echo "Downloaded Visium HD Mouse Brain demo data into $DEST_DIR/"
