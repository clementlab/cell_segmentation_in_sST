# Cell Segmentation in Sequencing-Based Spatial Transcriptomics

This repository accompanies the study:

**Cell segmentation method governs transcript recovery and biological interpretation in sequencing-based spatial transcriptomics**  
James Lubkowitz, Boyi Guo, Wei Zhang, Beatrice Knudsen, Kendell Clement

The project benchmarks how cell segmentation strategy changes transcript assignment, cell definitions, and downstream biological interpretation in high-resolution sequencing-based spatial transcriptomics, with a focus on 10x Genomics Visium HD data. The code compares eleven segmentation methods spanning nucleus-based, whole-cell, and transcript-aware approaches across multiple tissues.

## What This Repository Contains

The tracked repository is centered on a Snakemake workflow plus manuscript figure notebooks:

- [`analysis/pipeline.smk`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/pipeline.smk) is the main benchmarking pipeline.
- [`analysis/bin2cell_nuclei.py`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/bin2cell_nuclei.py) runs the Bin2Cell-based nucleus and expanded-cell workflow.
- [`analysis/STP.py`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/STP.py) runs STP-based cytoplasmic segmentation.
- [`analysis/envs/`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/envs) contains conda environment definitions used by the workflow.
- [`analysis/figures/`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures) contains notebooks and exported figures used for the manuscript.
- [`data/patholog_rankings/`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data/patholog_rankings) contains expert ranking spreadsheets used in the analysis.

This is an analysis repository, not a packaged Python library. The expected usage model is: prepare a Visium HD dataset, choose a region of interest, run the Snakemake workflow in a scratch/output directory, then use the notebooks to assemble manuscript figures.

## Quick Demo

A one-command demo is provided in [`demo/run_demo.sh`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/demo/run_demo.sh). It downloads the public Visium HD Mouse Brain dataset, fetches the pinned STP source, builds the required conda environments, and runs the Snakemake workflow on a small mouse-brain region of interest.

```bash
cd demo
./run_demo.sh 4
```

The numeric argument is the number of cores to give Snakemake. The demo expects an NVIDIA GPU because the workflow uses GPU-backed segmentation libraries; to bypass the GPU preflight check for testing, run:

```bash
./run_demo.sh --allow-cpu 4
```

Demo results are written to `demo/run_output/`. The demo data, generated conda environments, third-party checkout, and run outputs are intentionally ignored by git.

## Segmentation Methods Compared

The main pipeline produces or combines the following segmentation outputs:

- Nucleus-only outputs: `stardist`, `cellpose`, `bin2cell`
- Cytoplasm / whole-cell outputs: `cellpose`, `bin2cell`, `STP_tuned`, `STP_untuned`, `naive_8um`
- Expansion outputs: `stardist_voronoi`, `stardist_proximity`, `cellpose_voronoi`, `cellpose_proximity`

The manuscript frames the benchmark as **eleven segmentation methods**. In the code, some methods generate both nucleus-level and expanded-cell outputs, so the workflow materializes more than eleven analysis objects even when they derive from the same underlying method family.

## Data Requirements

The workflow expects inputs derived from a **10x Genomics Visium HD** run, specifically:

- Full-resolution tissue image (`.btf` or `.tif`)
- Filtered feature-barcode matrix (`filtered_feature_bc_matrix.h5`)
- Tissue positions parquet (`tissue_positions.parquet`)
- `square_002um` bin directory
- `spatial/` directory containing `scalefactors_json.json`

The repository does **not** ship the Visium HD datasets. The helper script [`data/get.sh`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data/get.sh) shows the original download locations used during manuscript development for several public 10x datasets:

- Human colon cancer
- Human lung cancer
- Mouse brain
- Mouse embryo
- Mouse small intestine
- Mouse kidney

Those paths are examples from the original HPC environment and should be adapted to your system.

## Environment and System Requirements

This workflow was built for a Linux/HPC environment and assumes:

- Conda or Mamba
- Snakemake
- Python 3.10
- NVIDIA GPU access for most segmentation methods
- Enough memory for full-image preprocessing and segmentation

The demo uses quick-solving environment specs for normal setup:

- [`analysis/envs/snakemake_quick.yaml`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/envs/snakemake_quick.yaml) is the driver environment used to launch Snakemake.
- [`analysis/envs/STP_env.yaml`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/envs/STP_env.yaml) is the STP per-rule environment. Snakemake builds this automatically under its `--conda-prefix` when STP rules run.

Full solved environment exports used for provenance are tracked separately:

- [`analysis/envs/explicit_envs/snakemake_env_full.yml`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/envs/explicit_envs/snakemake_env_full.yml)
- [`analysis/envs/explicit_envs/STP_env_full.yml`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/envs/explicit_envs/STP_env_full.yml)

These full exports document the solved environments used during development/publication runs, but the demo scripts intentionally point to the smaller quick-solving YAML files.

### Create the main environment

```bash
conda env create -n snakemake_env -f analysis/envs/snakemake_quick.yaml
conda activate snakemake_env
```

### STP environment

Do not create a named STP environment manually for the demo. The Snakemake run uses `--use-conda` and builds the STP rule environment from `analysis/envs/STP_env.yaml` under the configured conda prefix.

## Important Reproducibility Notes

Before running the full benchmark, there are a few non-obvious constraints in the tracked code:

- `analysis/STP.py` imports `STP_utils` from `analysis/dev_files/stp/STP`, so STP depends on that local vendored path rather than an installed package.


## Workflow Overview
At a high level, the workflow does the following:

1. Preprocess the Visium HD image and bin-level expression matrix.
2. Convert transcript bin coordinates into a GeoDataFrame.
3. Run multiple segmentation methods.
4. Assign bins/transcripts to polygons.
5. Aggregate counts per segmented cell.
6. Normalize, log-transform, run PCA/neighborhood graph construction, and Leiden clustering.
7. Generate per-method summary plots and cross-method benchmarking tables.

The pipeline internally computes:

- Cell counts
- UMI per cell
- Unique genes per cell
- Area per cell
- UMI density per cell area
- ROI transcript capture
- Pairwise adjusted Rand index (ARI) summaries across segmentation methods
- Resource usage summaries for runtime and memory

## Inputs Required by the Snakemake Workflow

The workflow is configured entirely through `--config` arguments. The required keys are:

- `region_x`: ROI x-range as a two-element list
- `region_y`: ROI y-range as a two-element list
- `image_path`: full-resolution tissue image
- `square_002um`: path to the `square_002um` directory
- `spatial`: path to the matching `spatial` directory
- `matrix_path`: path to `filtered_feature_bc_matrix.h5`
- `positions_path`: path to `tissue_positions.parquet`

### Minimal example

```bash
snakemake -s /path/to/repo/analysis/pipeline.smk \
  --use-conda \
  --conda-frontend mamba \
  --cores 8 \
  --resources gpu=1 \
  --config \
  region_x="[9000, 11000]" \
  region_y="[9000, 11000]" \
  image_path="/path/to/Visium_HD_sample_tissue_image.btf" \
  square_002um="/path/to/binned_outputs/square_002um" \
  spatial="/path/to/binned_outputs/square_002um/spatial" \
  matrix_path="/path/to/binned_outputs/square_002um/filtered_feature_bc_matrix.h5" \
  positions_path="/path/to/binned_outputs/square_002um/spatial/tissue_positions.parquet"
```

## Main Outputs

The pipeline produces method-level segmentation objects, clustered cell-by-gene matrices, resource-use summaries, and benchmarking plots.

### Core intermediate outputs

- `normalized_img.btf`
- `gdf_coordinates.gpkg`
- `adata_processed.h5ad`

### Segmentation outputs

- `*_nuclei.gpkg`
- `*_cyto.gpkg`
- `nuc_filtered_*_adata.h5ad`
- `cyto_filtered_*_adata.h5ad`

### Benchmarking outputs

- `resource_usage/*.csv`
- `plots/overview/total_cell_calls.csv`
- `plots/overview/total_UMI_per_cell.csv`
- `plots/overview/total_area_per_cell.csv`
- `plots/overview/total_UMI_per_area_per_cell.csv`
- `plots/overview/roi_transcripts_summary.csv`
- `plots/overview/ARI_segmentation.csv`

### Visualization outputs

Per-method plotting rules generate files such as:

- `plots/nuc_<method>_cluster.png`
- `plots/nuc_<method>_umi.png`
- `plots/nuc_<method>_spatial_visualization.png`
- `plots/nuc_<method>_area.png`
- `plots/cyto_<method>_cluster.png`
- `plots/cyto_<method>_umi.png`
- `plots/cyto_<method>_spatial_visualization.png`
- `plots/cyto_<method>_area.png`

## Figure Generation

The manuscript figures are assembled from notebooks in [`analysis/figures/`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures), including:

- [`analysis/figures/benchmark_fig.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/benchmark_fig.ipynb)
- [`analysis/figures/resource_usage.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/resource_usage.ipynb)
- [`analysis/figures/spatial_visuals.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/spatial_visuals.ipynb)
- [`analysis/figures/ARI_fig.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/ARI_fig.ipynb)
- [`analysis/figures/cell_type_fig.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/cell_type_fig.ipynb)
- [`analysis/figures/pathologist_ranking.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/pathologist_ranking.ipynb)
- [`analysis/figures/goblet_nuc_localization.ipynb`](/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/figures/goblet_nuc_localization.ipynb)

## Known Limitations

- The benchmark is tuned for Visium HD-style bin-level inputs and is not presented as a general segmentation framework for arbitrary spatial platforms.
- Several methods rely upon GPU availability
- STP integration relies on a local source path instead of a separately versioned installable package.

## Contact

For questions about the benchmark, manuscript, or repository, use the authorship information in the manuscript or the repository issue tracker at:

<https://github.com/clementlab/cell_segmentation_in_sST>
