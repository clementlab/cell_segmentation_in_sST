import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables all GPUs as Bin2Cell is CPU tool according to paper
import torch
torch.set_default_device("cpu") 

import tensorflow as tf
from tifffile import imread, imwrite
import numpy as np
from stardist.models import StarDist2D
import geopandas as gpd
import scanpy as sc
from cellpose import models, denoise, io, core, utils
import anndata
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon
from scipy import sparse
from skimage import measure
from csbdeep.utils import normalize
import bin2cell as b2c
import matplotlib.pyplot as plt
import cellpose
from scipy.spatial import Voronoi, cKDTree
from shapely import affinity, wkt, unary_union
from scipy.ndimage import binary_dilation
import sys
import json
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
import itertools
import time
from memory_profiler import memory_usage
import subprocess
import threading
import argparse

parser = argparse.ArgumentParser(description="Run Bin2Cell nuclei & cytoplasmic segmentation and clustering.")

# inputs & Params
parser.add_argument("--gdf_cords", required=True)
parser.add_argument("--img", required=True)
parser.add_argument("--square_002um", required=True)
parser.add_argument("--spatial", required=True)
parser.add_argument("--region_x", type=int, nargs=2, required=True)
parser.add_argument("--region_y", type=int, nargs=2, required=True)

# outputs
parser.add_argument("--nuc_gdf", required=True)
parser.add_argument("--nuc_adata", required=True)
parser.add_argument("--nuc_expanded_adata", required=True)
parser.add_argument("--resource_usage_cyto", required=True)

# Load in everything
args = parser.parse_args()
xmin, xmax = args.region_x
ymin, ymax = args.region_y
gdf_cords = gpd.read_file(args.gdf_cords)
path = args.square_002um
source_image_path = args.img
spatial = args.spatial

def monitor_gpu(tf_device_names, tf_usage_history, running_flag):
    while running_flag[0]:
        total_tf = 0
        for dev_name in tf_device_names:
            try:
                mem_info = tf.config.experimental.get_memory_info(dev_name)
                total_tf += mem_info['current']
            except Exception:
                pass  # ignore errors
        tf_usage_history.append(total_tf / 1e6)  # store in MB
        time.sleep(0.1)

def bin2cell_call(adata):
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.memory.reset_peak_memory_stats()

    torch_gpu_before = torch.cuda.memory_allocated() / 1e6

    # Prepare TF devices
    gpus = tf.config.list_physical_devices('GPU')
    tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
    
    tf_usage_history = []
    running_flag = [True]  # mutable container so thread can see updates

    # Start monitoring
    t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
    t.start()

    start = time.time()

    # rescale     
    mpp = 0.3
    b2c.scaled_he_image(adata, mpp=mpp, save_path="stardist/he.tiff")

    b2c.destripe(adata)

    b2c.stardist(image_path="stardist/he.tiff", 
    labels_npz_path="stardist/he.npz", 
    stardist_model="2D_versatile_he", 
    prob_thresh=0.01)

    b2c.insert_labels(adata, 
        labels_npz_path="stardist/he.npz", 
        basis="spatial", 
        spatial_key="spatial_cropped_150_buffer",
        mpp=mpp, 
        labels_key="labels_he")

    b2c.expand_labels(adata, 
        labels_key='labels_he', 
        expanded_labels_key="labels_he_expanded")
    
    # Make image of total counts
    b2c.grid_image(adata, "n_counts_adjusted", mpp=mpp, sigma=5, save_path="stardist/gex.tiff")
    
    # Predict cells based off total counts image
    b2c.stardist(image_path="stardist/gex.tiff", 
        labels_npz_path="stardist/gex.npz", 
        stardist_model="2D_versatile_fluo"
        #, 
        #prob_thresh=0.05, 
        #nms_thresh=0.5
        )

    # Insert labels into adata
    b2c.insert_labels(adata, 
            labels_npz_path="stardist/gex.npz", 
            basis="array", 
            mpp=mpp, 
            labels_key="labels_gex"
            )
    
    # Salvage secondary labels ie make 1 column with expanded cells and gex cells
    b2c.salvage_secondary_labels(adata, 
                        primary_label="labels_he_expanded", 
                        secondary_label="labels_gex", 
                        labels_key="labels_joint"
                        )

    end = time.time() - start
    torch_gpu_after = torch.cuda.memory_allocated() / 1e6
    torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"Bin2Cell GPU RAM before: {torch_gpu_before:.2f} MB")
    print(f"Bin2Cell GPU RAM peak:   {torch_gpu_peak:.2f} MB")
    print(f"Bin2Cell GPU RAM after:  {torch_gpu_after:.2f} MB")
    print(f"Bin2Cell segmentation time: {end:.2f} seconds")
    print(f"Peak TensorFlow memory: {max(tf_usage_history):.2f} MB")

    # Stop monitoring
    running_flag[0] = False
    t.join()

    stats = {
    "torch_gpu_before": torch_gpu_before,
    "torch_peak": torch_gpu_peak,
    "torch_after": torch_gpu_after,
    "segmentation_time": end,
    "tensorflow_peak": max(tf_usage_history)
    }   
    return adata, stats

os.makedirs("stardist", exist_ok=True) 

adata = b2c.read_visium(path, 
                source_image_path = source_image_path, 
                spaceranger_image_path = spatial
                )

adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_cells=1)        
sc.pp.filter_cells(adata, min_counts=1)

mem_usage, (adata, stats) = memory_usage(
    (bin2cell_call, (adata,), {}),
    retval=True)

stats["cpu_peak"] = max(mem_usage)

print(f"Bin2Cell Cyto Peak memory used: {max(mem_usage):.2f} MB")

# join adata.obs index and gdf_cords index
gdf_cords = gdf_cords.set_index('index')
# Perform a left join: keep all rows from adata.obs, add columns from gdf_cords where index matches
adata.obs = adata.obs.merge(gdf_cords, left_index=True, right_index=True, how='left')

adata_roi = adata[
    adata.obs['geometry'].apply(lambda p: xmin <= p.x <= xmax and ymin <= p.y <= ymax)
].copy()

# 1 - get nuclei gdfs
gdf = gpd.GeoDataFrame(adata_roi.obs, geometry='geometry')
# Drop NaN labels and label 0 (which are not cells)
gdf = gdf.dropna(subset=['labels_he'])
gdf = gdf[gdf['labels_he'] != 0]

# Grouping object
grouped_nuc = gdf.groupby('labels_he')

# Function to make convex hull or buffer
def make_polygon_or_buffer(pts):
    if len(pts) == 1:
        return pts.iloc[0].buffer(1)
    else:
        return MultiPoint(list(pts)).convex_hull
    
nuc_polygons = grouped_nuc.geometry.apply(make_polygon_or_buffer)

# filter out points not in cell
adata_cell_roi = adata_roi[(~adata_roi.obs['labels_joint'].isna()) & (adata_roi.obs['labels_joint'] != 0), :].copy()

# 2 - get total gene expression for each nucleus
# Group the data by unique nucleous IDs
adata_roi_nuc = adata_roi[(~adata_roi.obs['labels_he'].isna()) & (adata_roi.obs['labels_he'] != 0), :].copy()

groupby_object = adata_roi_nuc.obs.groupby(['labels_he'], observed=True)

# Extract the gene expression counts from the AnnData object
counts = adata_roi_nuc.X

# Obtain the number of unique nuclei and the number of genes in the expression data
N_groups = groupby_object.ngroups
N_genes = counts.shape[1]

# Initialize a sparse matrix to store the summed gene counts for each nucleus
summed_counts = sparse.lil_matrix((N_groups, N_genes))

# Create a list for polygon IDs (labels_he)
polygon_id = list(groupby_object.indices.keys())

# Map labels_he values to row indices
label_to_row = {label: i for i, label in enumerate(polygon_id)}

# Sum counts for each group
for label, idx_ in groupby_object.indices.items():
    row = label_to_row[label]
    summed_counts[row] = counts[idx_].sum(0)

# Convert to CSR for efficient slicing
summed_counts = summed_counts.tocsr()

# Make obs DataFrame: one row per polygon
obs_df = pd.DataFrame({'labels_he': polygon_id}, index=polygon_id)

# Attach nuc_polygons geometries by matching index (labels_he)
obs_df['geometry'] = nuc_polygons.loc[polygon_id].values

# Optionally make this a GeoDataFrame if you want to retain spatial features
obs_gdf = gpd.GeoDataFrame(obs_df, geometry='geometry')

# Create new AnnData with summed counts and grouped obs
nuc_grouped_filtered_adata = anndata.AnnData(
    X=summed_counts,
    obs=obs_gdf,
    var=adata_roi_nuc.var)


# Instad of calling norm_log_cluster_adata function from pipeline do manually
sc.pp.calculate_qc_metrics(nuc_grouped_filtered_adata, inplace=True)

# Normalize total counts for each cell in the AnnData object
sc.pp.normalize_total(nuc_grouped_filtered_adata, inplace=True)

# Logarithmize the values in the AnnData object after normalization
sc.pp.log1p(nuc_grouped_filtered_adata)

# Identify highly variable genes in the dataset using the Seurat method
sc.pp.highly_variable_genes(nuc_grouped_filtered_adata, flavor="seurat", n_top_genes=2000)

# Perform Principal Component Analysis (PCA) on the AnnData object
sc.pp.pca(nuc_grouped_filtered_adata)

# Build a neighborhood graph based on PCA components
sc.pp.neighbors(nuc_grouped_filtered_adata)

# Perform Leiden clustering on the neighborhood graph and store the results in 'clusters' column
sc.tl.leiden(nuc_grouped_filtered_adata, resolution=0.35, key_added="clusters", random_state=0)

# rename for clarity
processed_adata = nuc_grouped_filtered_adata.copy()

processed_adata.obs['id'] = processed_adata.obs['labels_he'].astype(str)
clusters_df = processed_adata.obs[['id', 'clusters', 'geometry']].reset_index()
nuclei_gdf = gpd.GeoDataFrame(clusters_df, geometry='geometry')
nuclei_gdf['area'] = nuclei_gdf.geometry.area
processed_adata.obs['geometry'] = processed_adata.obs['geometry'].apply(lambda geom: geom.wkt)
processed_adata.obs['geometry'] = processed_adata.obs['geometry'].astype(str)
processed_adata.obs = pd.DataFrame(processed_adata.obs)

adata_cell_roi.obs['geometry'] = adata_cell_roi.obs['geometry'].apply(lambda geom: geom.wkt)
adata_cell_roi.obs['geometry'] = adata_cell_roi.obs['geometry'].astype(str)
adata_cell_roi.obs = pd.DataFrame(adata_cell_roi.obs)

# Save outputs
nuclei_gdf.to_file(args.nuc_gdf, driver="GPKG")
processed_adata.write(args.nuc_adata)
resource_use_df_cyto = pd.DataFrame([stats], index=["bin2cell_cyto"])
resource_use_df_cyto.to_csv(args.resource_usage_cyto)

# Save the non-filtered adata for cell segmentation
adata_cell_roi.write(args.nuc_expanded_adata)



