import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Ensure TF does not allocate all GPU memory at once

import tensorflow as tf
# Explicitly set memory growth for processes that use GPU
gpus = tf.config.list_physical_devices("GPU")
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from tifffile import imread, imwrite
import numpy as np
from stardist.models import StarDist2D
import geopandas as gpd
import scanpy as sc
from cellpose import models, denoise, io, core, utils
import anndata
import pandas as pd
from shapely.geometry import Polygon, Point, MultiPoint, MultiPolygon, box
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
import torch
import time
from memory_profiler import memory_usage
import subprocess
import threading
import gc

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

def norm_log_cluster_adata(adata):
    """
    Normalize, log-transform, and cluster the AnnData object.
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object to be processed.
    Returns
    ------- 
    adata : anndata.AnnData
        The processed AnnData object with normalized, log-transformed data and clustering results.
    """
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Normalize total counts for each cell in the AnnData object
    sc.pp.normalize_total(adata, inplace=True)

    # Logarithmize the values in the AnnData object after normalization
    sc.pp.log1p(adata)

    # Identify highly variable genes in the dataset using the Seurat method
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

    # Perform Principal Component Analysis (PCA) on the AnnData object
    sc.pp.pca(adata)

    # Build a neighborhood graph based on PCA components
    sc.pp.neighbors(adata)

    # Perform Leiden clustering on the neighborhood graph and store the results in 'clusters' column
    sc.tl.leiden(adata, resolution=0.35, key_added="clusters", random_state=0)

    return adata


NUC_TOOLS = ["stardist", "cellpose", "bin2cell"]
CYTO_TOOLS = ["cellpose", "bin2cell", "STP_tuned", "STP_untuned", "naive_8um"]

NUC_EXPANSION_COMBO = ["stardist", "cellpose"]
CYTO_EXPANSION_COMBO = ["voronoi", "proximity"]
PLOTTING_COMBO = [f"{x}_{y}" for x in NUC_EXPANSION_COMBO for y in CYTO_EXPANSION_COMBO]

RESOURCE_USAGE = [
    "resource_usage/stardist_nuclei.csv",
    "resource_usage/cellpose_nuclei.csv",
    "resource_usage/naive_8um.csv",
    "resource_usage/STP_tuned.csv",
    "resource_usage/STP_untuned.csv",
    "resource_usage/bin2cell_cyto.csv",
    "resource_usage/stardist_proximity.csv",
    "resource_usage/stardist_voronoi.csv",
    "resource_usage/cellpose_proximity.csv",
    "resource_usage/cellpose_voronoi.csv",
]

rule all:
    input:
        # Nuclei Calls
        expand("plots/nuc_{sample}_cluster.png", sample=NUC_TOOLS),
        expand("plots/nuc_{sample}_umi.png", sample=NUC_TOOLS),
        expand("plots/nuc_{sample}_spatial_visualization.png", sample=NUC_TOOLS),
        expand("plots/nuc_{sample}_area.png", sample=NUC_TOOLS),

        # Cytoplasm Calls
        expand("plots/cyto_{sample}_cluster.png", sample=CYTO_TOOLS),
        expand("plots/cyto_{sample}_umi.png", sample=CYTO_TOOLS),
        expand("plots/cyto_{sample}_spatial_visualization.png", sample=CYTO_TOOLS),
        expand("plots/cyto_{sample}_area.png", sample=CYTO_TOOLS),
        
        expand("plots/cyto_{sample}_cluster.png", sample=PLOTTING_COMBO),
        expand("plots/cyto_{sample}_umi.png", sample=PLOTTING_COMBO),
        expand("plots/cyto_{sample}_spatial_visualization.png", sample=PLOTTING_COMBO),
        expand("plots/cyto_{sample}_area.png", sample=PLOTTING_COMBO),


        expand("{sample}", sample=RESOURCE_USAGE),

        #expand("{sample}_voronoi_cyto.gpkg", sample=NUC_EXPANSION_COMBO),
        #expand("cyto_filtered_{sample}_voronoi_adata.h5", sample=NUC_EXPANSION_COMBO),

        expand("plots/overview/total_cell_calls.csv")


rule preprocess_image_spatial_data:
    output:
        norm_img = "normalized_img.btf",
        gdf = "gdf_coordinates.gpkg",
        adata = "adata_processed.h5ad"
    params:
        image=config["image_path"],
        matrix=config["matrix_path"],
        positions=config["positions_path"]
    resources:
        gpu=1
    run:
        img = imread(params.image)
        min_percentile = 5
        max_percentile = 95
        norm_img = normalize(img, min_percentile, max_percentile)
        imwrite(output.norm_img, norm_img)

        adata = sc.read_10x_h5(params.matrix)
        df_pos = pd.read_parquet(params.positions).set_index('barcode')
        df_pos['index'] = df_pos.index
        adata.obs = adata.obs.merge(df_pos, left_index=True, right_index=True)

        geometry = [Point(xy) for xy in zip(df_pos['pxl_col_in_fullres'], df_pos['pxl_row_in_fullres'])]
        gdf = gpd.GeoDataFrame(df_pos, geometry=geometry)
        gdf.to_file(output.gdf, driver="GPKG")
        adata.write(output.adata)

rule naive_8um:
    input:
        gdf_cords="gdf_coordinates.gpkg",
        adata="adata_processed.h5ad"
    params:
        region_x=config["region_x"],
        region_y=config["region_y"],
        spatial=config["spatial"],
    output:
        cell_gdf="naive_8um_cyto.gpkg",
        cell_adata="cyto_filtered_naive_8um_adata.h5ad",
        resource_usage="resource_usage/naive_8um.csv"
    resources:
        gpu=1
    run:
        def run_naive_8um(xmin, xmax, ymin, ymax, pixels_per_8_micron):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.memory.reset_peak_memory_stats()
                        torch_gpu_before = torch.cuda.memory_allocated() / 1e6
                    else:
                        torch_gpu_before = 0
                        
                    # Prepare TF devices
                    gpus = tf.config.list_physical_devices('GPU')
                    tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
                    
                    tf_usage_history = []
                    running_flag = [True]  # mutable container so thread can see updates

                    # Start monitoring
                    t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
                    t.start()
                    start = time.time()
                    
                    # Create grid positions (make sure the last edge hits xmax/ymax)
                    x_coords = list(np.arange(xmin, xmax, pixels_per_8_micron))
                    y_coords = list(np.arange(ymin, ymax, pixels_per_8_micron))

                    if x_coords[-1] != xmax:
                        x_coords.append(xmax)

                    if y_coords[-1] != ymax:
                        y_coords.append(ymax)

                    polygons = []

                    # Loop over rows and columns to create rectangular polygons
                    for i in range(len(x_coords)-1):
                        for j in range(len(y_coords)-1):
                            poly = Polygon([
                                (x_coords[i], y_coords[j]),
                                (x_coords[i+1], y_coords[j]),
                                (x_coords[i+1], y_coords[j+1]),
                                (x_coords[i], y_coords[j+1])
                            ])
                            polygons.append(poly)

                    # Create a GeoDataFrame
                    grid_gdf = gpd.GeoDataFrame({'geometry': polygons})

                    
                    end = time.time() - start
                    torch_gpu_after = torch.cuda.memory_allocated() / 1e6
                    torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
                    print(f"Naive GPU RAM before: {torch_gpu_before:.2f} MB")
                    print(f"Naive GPU RAM peak:   {torch_gpu_peak:.2f} MB")
                    print(f"Naive GPU RAM after:  {torch_gpu_after:.2f} MB")
                    print(f"Naive segmentation time: {end:.2f} seconds")
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
                    return grid_gdf, stats

        # Inputs
        gdf = gpd.read_file(input.gdf_cords)
        adata = sc.read_h5ad(input.adata)
        spatial = params.spatial
        region_x = params.region_x
        region_y = params.region_y
        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])


        with open(os.path.join(spatial, "scalefactors_json.json")) as f:
            scalefactors = json.load(f)
        microns_per_pixel = scalefactors["microns_per_pixel"]



        # crop everything for faster processing 
        gdf_cropped = gdf.cx[xmin:xmax, ymin:ymax]
        adata_cropped = adata[adata.obs.index.isin(gdf_cropped['barcode'])].copy()

        pixels_per_8_micron = int(8 / microns_per_pixel)

        mem_usage, (grid_gdf, stats) = memory_usage((run_naive_8um, (xmin, xmax, ymin, ymax, pixels_per_8_micron), {}), retval=True)
        
        stats["cpu_peak"] = max(mem_usage)
        
        grid_gdf['cell_id'] = range(len(grid_gdf))

        # Join grid_gdf with gdf_cropped to count points in each polygon
        grid_gdf['cell_id'] = range(len(grid_gdf))

        joined = gpd.sjoin(grid_gdf, gdf_cropped, how='left', predicate='contains')
        joined.set_index('index')

        joined = joined[['index','geometry', 'cell_id']]
        joined.rename(columns={'geometry': 'cell_geometry'}, inplace=True)
        joined.set_index('index', inplace=True)

        filtered_adata = adata_cropped.copy()
        # Add the results of the point spatial join to the Anndata object
        filtered_adata.obs = pd.merge(
                filtered_adata.obs,
                joined[['cell_geometry','cell_id']],
                left_on='index',  
                right_on='index',
                how='left'
                ).set_index('index')

        # Group the data by unique cell IDs
        groupby_object = filtered_adata.obs.groupby(['cell_id'], observed=True)

        # Extract the gene expression counts from the AnnData object
        counts = filtered_adata.X

        # Obtain the number of unique nuclei and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each nucleus
        summed_counts = sparse.lil_matrix((N_groups, N_genes))

        # Lists to store the IDs of polygons and the current row index
        polygon_id = []
        row = 0

        # Iterate over each unique polygon to calculate the sum of gene counts.
        for polygons, idx_ in groupby_object.indices.items():
            summed_counts[row] = counts[idx_].sum(0)
            row += 1
            polygon_id.append(polygons)

        # Create and AnnData object from the summed count matrix
        summed_counts = summed_counts.tocsr()
        grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)

        grid_gdf['area'] = grid_gdf['geometry'].area
        cell_gdf = grid_gdf

        processed_adata = norm_log_cluster_adata(grouped_filtered_adata)

        cell_gdf.rename(columns={'cell_id': 'id'}, inplace=True)
        
        clusters_df = processed_adata.obs[['id', 'clusters']].reset_index()
        cell_gdf = cell_gdf.merge(clusters_df, on='id', how='left')

        resource_use_df = pd.DataFrame([stats], index=["naive_8um"])
        resource_use_df.to_csv(output.resource_usage)
        processed_adata.write(output.cell_adata)
        cell_gdf.to_file(output.cell_gdf, driver="GPKG")


rule stardist_nuclei:
    input:
        norm_img="normalized_img.btf",
        gdf="gdf_coordinates.gpkg",
        adata="adata_processed.h5ad"
    params:
        region_x=config["region_x"],
        region_y=config["region_y"]
    output:
        nuc_gdf="stardist_nuclei.gpkg",
        nuc_adata="nuc_filtered_stardist_adata.h5ad",
        resource_usage="resource_usage/stardist_nuclei.csv"
    resources:
        gpu=1
    run:
        def run_stardist(img, blocksize):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.reset_peak_memory_stats()
                torch_gpu_before = torch.cuda.memory_allocated() / 1e6
            else:
                torch_gpu_before = 0
                
            # Prepare TF devices
            gpus = tf.config.list_physical_devices('GPU')
            tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
            
            tf_usage_history = []
            running_flag = [True]  # mutable container so thread can see updates

            # Start monitoring
            t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
            t.start()
            start = time.time()
            
            model = StarDist2D.from_pretrained('2D_versatile_he')
            labels, polys = model.predict_instances_big(img, axes='YXC', block_size=blocksize, prob_thresh=0.01, nms_thresh=0.001, min_overlap=128, context=128, normalizer=None)
            
            end = time.time() - start
            torch_gpu_after = torch.cuda.memory_allocated() / 1e6
            torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"StarDist GPU RAM before: {torch_gpu_before:.2f} MB")
            print(f"StarDist GPU RAM peak:   {torch_gpu_peak:.2f} MB")
            print(f"StarDist GPU RAM after:  {torch_gpu_after:.2f} MB")
            print(f"StarDist segmentation time: {end:.2f} seconds")
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
            return labels, polys, stats

        region_x = params.region_x
        region_y = params.region_y
        gdf_coords = gpd.read_file(input.gdf)
        adata = anndata.read_h5ad(input.adata)
        img = imread(input.norm_img)
        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])

        img = img[ymin:ymax, xmin:xmax]

        width = xmax - xmin
        height = ymax - ymin
        blocksize = min(width, height, 4096)
        
        #model = StarDist2D.from_pretrained('2D_versatile_he')
        #labels, polys = model.predict_instances_big(img, axes='YXC', block_size=blocksize, prob_thresh=0.01, nms_thresh=0.001, min_overlap=128, context=128, normalizer=None)
        
        # Track memory and get results
        mem_usage, (labels, polys, stats) = memory_usage((run_stardist, (img, blocksize), {}), retval=True)
        
        print(f"Stardist Peak memory used: {max(mem_usage):.2f} MB")
        stats["cpu_peak"] = max(mem_usage)

        # Creating a list to store Polygon geometries
        nuclei_geometries = []

        # Iterating through each nuclei in the 'polys' DataFrame
        for nuclei in range(len(polys['coord'])):
            # Extracting coordinates for the current nuclei and converting them to (y, x) format
            coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]
            # Creating a Polygon geometry from the coordinates
            nuclei_geometries.append(Polygon(coords))
        # Creating a GeoDataFrame using the Polygon geometries
        nuclei_gdf = gpd.GeoDataFrame(geometry=nuclei_geometries)

        nuclei_gdf['id'] = [f"ID_{i+1}" for i, _ in enumerate(nuclei_gdf.index)]

        # Translate each polygon
        nuclei_gdf['geometry'] = nuclei_gdf['geometry'].apply(lambda p: affinity.translate(p, xoff=xmin, yoff=ymin))

        # Perform a spatial join to check which coordinates are in a cell nucleus
        result_spatial_join = gpd.sjoin(gdf_coords, nuclei_gdf, how='left', predicate='within')

        # Identify nuclei associated barcodes and find barcodes that are in more than one nucleus
        result_spatial_join['is_within_polygon'] = ~result_spatial_join['index_right'].isna()
        barcodes_in_overlaping_polygons = pd.unique(result_spatial_join[result_spatial_join.duplicated(subset=['index'])]['index'])
        result_spatial_join['is_not_in_an_polygon_overlap'] = ~result_spatial_join['index'].isin(barcodes_in_overlaping_polygons)

        # Remove barcodes in overlapping nuclei
        barcodes_in_one_polygon = result_spatial_join[result_spatial_join['is_within_polygon'] & result_spatial_join['is_not_in_an_polygon_overlap']]

        # The AnnData object is filtered to only contain the barcodes that are in non-overlapping polygon regions
        filtered_obs_mask = adata.obs['index'].isin(barcodes_in_one_polygon['index'])
        filtered_adata = adata[filtered_obs_mask,:]

        # Add the results of the point spatial join to the Anndata object
        filtered_adata.obs = pd.merge(
                filtered_adata.obs,
                barcodes_in_one_polygon[['index','geometry','id','is_within_polygon','is_not_in_an_polygon_overlap']],
                left_on='index',  
                right_on='index',
                how='left'
                ).set_index('index')

        # Group the data by unique nucleous IDs
        groupby_object = filtered_adata.obs.groupby(['id'], observed=True)

        # Extract the gene expression counts from the AnnData object
        counts = filtered_adata.X

        # Obtain the number of unique nuclei and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each nucleus
        summed_counts = sparse.lil_matrix((N_groups, N_genes))

        # Lists to store the IDs of polygons and the current row index
        polygon_id = []
        row = 0

        # Iterate over each unique polygon to calculate the sum of gene counts.
        for polygons, idx_ in groupby_object.indices.items():
            summed_counts[row] = counts[idx_].sum(0)
            row += 1
            polygon_id.append(polygons)

        # Create and AnnData object from the summed count matrix
        summed_counts = summed_counts.tocsr()
        nuc_grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)
        
        nuclei_gdf['area'] = nuclei_gdf['geometry'].area

        processed_adata = norm_log_cluster_adata(nuc_grouped_filtered_adata)

        resource_use_df = pd.DataFrame([stats], index=["stardist_nuclei"])
        resource_use_df.to_csv(output.resource_usage)
        processed_adata.write(output.nuc_adata)
        clusters_df = processed_adata.obs[['id', 'clusters']].reset_index()
        nuclei_gdf = nuclei_gdf.merge(clusters_df, on='id', how='left')
        nuclei_gdf.to_file(output.nuc_gdf, driver="GPKG")

rule cellpose_nuclei:
    input:
        norm_img="normalized_img.btf",
        gdf="gdf_coordinates.gpkg",
        adata="adata_processed.h5ad"
    params:
        region_x=config["region_x"],
        region_y=config["region_y"]
    resources:
        gpu=1
    output:
        nuc_gdf="cellpose_nuclei.gpkg",
        nuc_adata="nuc_filtered_cellpose_adata.h5ad",
        resource_usage="resource_usage/cellpose_nuclei.csv"
    run:
        def cellpose_nuclei_call(img):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.memory.reset_peak_memory_stats()
                torch_gpu_before = torch.cuda.memory_allocated() / 1e6
            else:
                torch_gpu_before = 0

            # Prepare TF devices
            gpus = tf.config.list_physical_devices('GPU')
            tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
            
            tf_usage_history = []
            running_flag = [True]  # mutable container so thread can see updates

            # Start monitoring
            t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
            t.start()
            start = time.time()

            model = models.Cellpose(gpu=True, model_type="nuclei")
            masks, flows, styles, imgs_dn = model.eval(img, diameter=None, channels=[0,0])
            
            end = time.time() - start
            torch_gpu_after = torch.cuda.memory_allocated() / 1e6
            torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"Cellpose Nuclei GPU RAM before: {torch_gpu_before:.2f} MB")
            print(f"Cellpose Nuclei GPU RAM peak:   {torch_gpu_peak:.2f} MB")
            print(f"Cellpose Nuclei GPU RAM after:  {torch_gpu_after:.2f} MB")
            print(f"Cellpose Nuclei segmentation time: {end:.2f} seconds")
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

            return masks, flows, styles, imgs_dn, stats

        gdf_coords = gpd.read_file(input.gdf)
        adata = anndata.read_h5ad(input.adata)
        img = imread(input.norm_img)

        region_x = params.region_x
        region_y = params.region_y
        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])

        img = img[ymin:ymax, xmin:xmax]

        mem_usage, (masks, flows, styles, imgs_dn, stats) = memory_usage(
            (cellpose_nuclei_call, (img,), {}), 
            retval=True)
        
        print(f"Cellpose Peak memory used: {max(mem_usage):.2f} MB")

        stats["cpu_peak"] = max(mem_usage)
        
        outlines = utils.outlines_list(masks, multiprocessing=False)
        print("outlines extracted")
        polygons = []
        for outline in outlines:
            coords = [(x, y) for x, y in outline]
            if len(coords) > 3:
                polygons.append(Polygon(coords))

        nuc_gdf = gpd.GeoDataFrame(geometry=polygons)
        nuc_gdf['id'] = [f"CP_{i}" for i in nuc_gdf.index]

        # Translate each polygon
        nuc_gdf['geometry'] = nuc_gdf['geometry'].apply(lambda p: affinity.translate(p, xoff=xmin, yoff=ymin))

        result_spatial_join = gpd.sjoin(gdf_coords, nuc_gdf, how='left', predicate='within')

        # Identify nuclei associated barcodes and find barcodes that are in more than one nucleus
        result_spatial_join['is_within_polygon'] = ~result_spatial_join['index_right'].isna()
        barcodes_in_overlaping_polygons = pd.unique(result_spatial_join[result_spatial_join.duplicated(subset=['index'])]['index'])
        result_spatial_join['is_not_in_an_polygon_overlap'] = ~result_spatial_join['index'].isin(barcodes_in_overlaping_polygons)

        # Remove barcodes in overlapping nuclei
        barcodes_in_one_polygon = result_spatial_join[result_spatial_join['is_within_polygon'] & result_spatial_join['is_not_in_an_polygon_overlap']]

        # The AnnData object is filtered to only contain the barcodes that are in non-overlapping polygon regions
        filtered_obs_mask = adata.obs['index'].isin(barcodes_in_one_polygon['index'])
        filtered_adata = adata[filtered_obs_mask,:]

        # Add the results of the point spatial join to the Anndata object
        filtered_adata.obs = pd.merge(
        filtered_adata.obs,
        barcodes_in_one_polygon[['index','geometry','id','is_within_polygon','is_not_in_an_polygon_overlap']],
        left_on='index',  
        right_on='index',
        how='left'
        ).set_index('index')

        # Group the data by unique nucleous IDs
        groupby_object = filtered_adata.obs.groupby(['id'], observed=True)

        # Extract the gene expression counts from the AnnData object
        counts = filtered_adata.X

        # Obtain the number of unique nuclei and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each nucleus
        summed_counts = sparse.lil_matrix((N_groups, N_genes))

        # Lists to store the IDs of polygons and the current row index
        polygon_id = []
        row = 0

        # Iterate over each unique polygon to calculate the sum of gene counts.
        for polygons, idx_ in groupby_object.indices.items():
            summed_counts[row] = counts[idx_].sum(0)
            row += 1
            polygon_id.append(polygons)

        # Create and AnnData object from the summed count matrix
        summed_counts = summed_counts.tocsr()
        nuc_grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)
        """Anndata object of nuclei expression data grouped"""

        nuc_gdf['area'] = nuc_gdf['geometry'].area

        processed_adata = norm_log_cluster_adata(nuc_grouped_filtered_adata)

        clusters_df = processed_adata.obs[['id', 'clusters']].reset_index()
        nuclei_gdf = nuc_gdf.merge(clusters_df, on='id', how='left')

        # Save outputs
        resource_use_df = pd.DataFrame([stats], index=["cellpose_nuclei"])
        resource_use_df.to_csv(output.resource_usage)
        nuclei_gdf.to_file(output.nuc_gdf, driver="GPKG")
        processed_adata.write(output.nuc_adata)

rule nuclei_plotting:
    input:
        adata = "nuc_filtered_{sample}_adata.h5ad",
        gdf = "{sample}_nuclei.gpkg",
    output:
        cluster_plot = "plots/nuc_{sample}_cluster.png",
        umi_plot = "plots/nuc_{sample}_umi.png",
        spatial_plot = "plots/nuc_{sample}_spatial_visualization.png",
        area_plot = "plots/nuc_{sample}_area.png"
    resources:
        mem_mb=15000
    params:
        image=config["image_path"],
        label = lambda wildcards: wildcards.sample
    run:
        # Load inputs
        gdf = gpd.read_file(input.gdf)
        adata = anndata.read_h5ad(input.adata)
        img = imread(params.image)

        cluster_plot_path = output.cluster_plot
        umi_plot_path = output.umi_plot
        sample_label = params.label
        spatial_plot_path = output.spatial_plot
        area_plot_path = output.area_plot

    # Plot 1: Cluster Distribution Bar Plot
        adata.obs['clusters'].value_counts().sort_index().plot(kind='bar')
        plt.xlabel('Cluster')
        plt.title(f'{sample_label}: Cluster Distribution')
        plt.ylabel('Number of Nuclei')
        plt.tight_layout()
        plt.savefig(cluster_plot_path)
        plt.close()

    # Plot 2: UMI Counts Histogram
        values = adata.obs['total_counts']
        clusters = adata.obs['clusters']

        # Get sorted unique cluster labels
        unique_clusters = sorted(clusters.unique(), key=lambda x: int(x))
        n_clusters = len(unique_clusters)

        # Define color for each cluster (consistent)
        cmap = plt.cm.tab20
        cluster_colors = {cluster: cmap(i / n_clusters) for i, cluster in enumerate(unique_clusters)}

        # Prepare data for stacked histogram
        cluster_values_list = [values[clusters == cluster] for cluster in unique_clusters]
        colors_list = [cluster_colors[cluster] for cluster in unique_clusters]

        # Set up subplots (1 for combined + n_clusters)
        n_cols = 4
        n_rows = int(np.ceil((n_clusters + 1) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        # Plot combined stacked histogram in first subplot
        ax = axes[0]
        ax.hist(
            cluster_values_list, 
            bins=50, 
            stacked=True, 
            color=colors_list, 
            label=[f'Cluster {cluster}' for cluster in unique_clusters],
            alpha=0.85
        )
        ax.set_title('All Clusters Combined')
        ax.set_xlabel('UMI Counts per Nuclei')
        ax.set_ylabel('Number of Nuclei')
        ax.legend(fontsize='x-small', frameon=False)

        # Plot individual histograms
        for i, cluster in enumerate(unique_clusters, start=1):
            ax = axes[i]
            cluster_values = values[clusters == cluster]
            ax.hist(
                cluster_values, 
                bins=50, 
                color=cluster_colors[cluster], 
                alpha=0.85
            )
            ax.set_title(f'Cluster {cluster}')
            ax.set_xlabel('UMI Counts per Nuclei')
            ax.set_ylabel('Number of Nuclei')

        # Hide any extra empty subplots
        for j in range(n_clusters + 1, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f'{sample_label} UMI Counts per Nuclei', fontsize=16, y=.98)
        plt.tight_layout()
        plt.savefig(umi_plot_path)
        plt.close()

    # Plot 3: Spatial Visualization
        xmin, ymin, xmax, ymax = gdf.total_bounds
        # if any are negative set to 0
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xlim = (xmin,xmax)
        ylim = (ymin,ymax)

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf.plot(ax=ax, column='clusters', cmap='viridis', edgecolor='black', legend=True, alpha=.7)
        cropped_img = img[int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1])]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{sample_label} Segmentation w/ Nuclei Boundries in Black & colored by Cluster')
        ax.imshow(cropped_img, cmap='gray', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')
        plt.tight_layout()
        plt.savefig(spatial_plot_path)
        plt.close()

    # Plot 4: Area of Nuclei
        gdf['area'] = gdf['geometry'].area

        # drop NA
        gdf = gdf.dropna(subset=['clusters'])

        areas = gdf['area']
        clusters = gdf['clusters'] 

        # Get sorted unique cluster labels
        unique_clusters = sorted(clusters.unique(), key=lambda x: int(x))
        n_clusters = len(unique_clusters)

        # Define consistent colors
        cmap = plt.cm.tab20
        cluster_colors = {cluster: cmap(i / n_clusters) for i, cluster in enumerate(unique_clusters)}

        # Prepare data for stacked histogram
        cluster_areas_list = [areas[clusters == cluster] for cluster in unique_clusters]
        colors_list = [cluster_colors[cluster] for cluster in unique_clusters]

        # Set up subplots (1 combined + individual clusters)
        n_cols = 4
        n_rows = int(np.ceil((n_clusters + 1) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        # Combined stacked histogram for all clusters
        ax = axes[0]
        ax.hist(
            cluster_areas_list,
            bins=50,
            stacked=True,
            color=colors_list,
            label=[f'Cluster {cluster}' for cluster in unique_clusters],
            alpha=0.85
        )
        ax.set_title('All Clusters Combined')
        ax.set_xlabel('Area')
        ax.set_ylabel('Number of Nuclei')
        ax.legend(fontsize='x-small', frameon=False)

        # Individual cluster histograms
        for i, cluster in enumerate(unique_clusters, start=1):
            ax = axes[i]
            cluster_areas = areas[clusters == cluster]
            ax.hist(
                cluster_areas,
                bins=50,
                color=cluster_colors[cluster],
                alpha=0.85
            )
            ax.set_title(f'Cluster {cluster}')
            ax.set_xlabel('Area')
            ax.set_ylabel('Number of Nuclei')

        # Hide empty subplots
        for j in range(n_clusters + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'{sample_label} Area of Nuclei', fontsize=16, y=.98)
        plt.tight_layout()
        plt.savefig(area_plot_path)
        plt.close()

rule cellpose_cyto:
    input:
        gdf="gdf_coordinates.gpkg",
        adata="adata_processed.h5ad"
    output:
        cyto_gdf="cellpose_cyto.gpkg",
        cellpose_cyto_adata="cyto_filtered_cellpose_adata.h5ad",
        resource_usage="resource_usage/cellpose_cyto3.csv"
    resources:
        gpu=1
    params:
        region_x=config["region_x"],
        region_y=config["region_y"],
        img=config["image_path"]
    run:
        def cellpose_cyto_call(img):
            if torch.cuda.is_available():   
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()  
                torch_gpu_before = torch.cuda.memory_allocated() / 1e6
            else:
                torch_gpu_before = 0
            # Prepare TF devices
            gpus = tf.config.list_physical_devices('GPU')
            tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
            
            tf_usage_history = []
            running_flag = [True]  # mutable container so thread can see updates

            # Start monitoring
            t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
            t.start()
            start = time.time()

            model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="upsample_cyto3")
            masks, flows, styles, imgs_dn = model.eval(img, diameter=None, channels=[0,0])

            end = time.time() - start
            torch_gpu_after = torch.cuda.memory_allocated() / 1e6
            torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"Cellpose Cyto GPU RAM before: {torch_gpu_before:.2f} MB")
            print(f"Cellpose Cyto GPU RAM peak:   {torch_gpu_peak:.2f} MB")
            print(f"Cellpose Cyto GPU RAM after:  {torch_gpu_after:.2f} MB")
            print(f"Cellpose Cyto segmentation time: {end:.2f} seconds")
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

            return masks, flows, styles, imgs_dn, stats

        img = imread(params.img)

        region_x = params.region_x
        region_y = params.region_y

        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])

        img = img[ymin:ymax, xmin:xmax]

        print(f"Cellpose Version: {cellpose.version}")
        use_GPU = core.use_gpu()
        yn = ['NO', 'YES']
        print(f'>>> GPU activated? {yn[use_GPU]}')

        io.logger_setup()

        mem_usage, (masks, flows, styles, imgs_dn, stats) = memory_usage(
            (cellpose_cyto_call, (img,), {}),
            retval=True)
        
        print(f"Cellpose Cyto3 Peak memory used: {max(mem_usage):.2f} MB")

        stats["cpu_peak"] = max(mem_usage)

        outlines = utils.outlines_list(masks, multiprocessing=False)
        print("outlines extracted")


        gdf_coords = gpd.read_file(input.gdf)
        adata = anndata.read_h5ad(input.adata)

        # Convert mask regions to polygons
        polygons = []
        for outline in outlines:
            coords = [(x, y) for x, y in outline]
            if len(coords) > 3:
                polygons.append(Polygon(coords))

        cell_gdf = gpd.GeoDataFrame(geometry=polygons)
        cell_gdf['id'] = [f"CP_cyto_{i}" for i in cell_gdf.index]

        # Translate each polygon
        cell_gdf['geometry'] = cell_gdf['geometry'].apply(lambda p: affinity.translate(p, xoff=xmin, yoff=ymin))
        print("valid geos:", cell_gdf.is_valid.sum(), "/", len(cell_gdf))
        cell_gdf = cell_gdf[cell_gdf.is_valid].copy()

        result_spatial_join = gpd.sjoin(gdf_coords, cell_gdf, how='left', predicate='within')

        # Identify cell associated barcodes and find barcodes that are in more than one cell
        result_spatial_join['is_within_polygon'] = ~result_spatial_join['index_right'].isna()
        barcodes_in_overlaping_polygons = pd.unique(result_spatial_join[result_spatial_join.duplicated(subset=['index'])]['index'])
        result_spatial_join['is_not_in_an_polygon_overlap'] = ~result_spatial_join['index'].isin(barcodes_in_overlaping_polygons)

        # Remove barcodes in overlapping cells
        barcodes_in_one_polygon = result_spatial_join[result_spatial_join['is_within_polygon'] & result_spatial_join['is_not_in_an_polygon_overlap']]

        # The AnnData object is filtered to only contain the barcodes that are in non-overlapping polygon regions
        filtered_obs_mask = adata.obs['index'].isin(barcodes_in_one_polygon['index'])
        filtered_adata = adata[filtered_obs_mask,:]

        # Add the results of the point spatial join to the Anndata object
        filtered_adata.obs = pd.merge(filtered_adata.obs,
                                        barcodes_in_one_polygon[['index','geometry','id','is_within_polygon','is_not_in_an_polygon_overlap']],
                                        left_on='index',  
                                        right_on='index',
                                        how='left').set_index('index')
        print(filtered_adata.shape)
        sc.pp.filter_genes(filtered_adata, min_cells=1)
        print(f"Post-filtering out non-expressed genes: {filtered_adata.shape}")

        # Group the data by unique cell IDs
        groupby_object = filtered_adata.obs.groupby(['id'], observed=True)

        # Extract the gene expression counts from the AnnData object
        counts = filtered_adata.X

        # Obtain the number of unique cells and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each cell
        summed_counts = sparse.lil_matrix((N_groups, N_genes))

        # Lists to store the IDs of polygons and the current row index
        polygon_id = []
        row = 0

        # Iterate over each unique polygon to calculate the sum of gene counts.
        for polygons, idx_ in groupby_object.indices.items():
            summed_counts[row] = counts[idx_].sum(0)
            row += 1
            polygon_id.append(polygons)

        # Create and AnnData object from the summed count matrix
        summed_counts = summed_counts.tocsr()

        cellpose_cyto_grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)

        cell_gdf['area'] = cell_gdf['geometry'].area
        print(f"shape: {cellpose_cyto_grouped_filtered_adata.shape}")
        sc.pp.filter_genes(cellpose_cyto_grouped_filtered_adata, min_cells=1)
        print(f"shape after filtering: {cellpose_cyto_grouped_filtered_adata.shape}")

        processed_adata = norm_log_cluster_adata(cellpose_cyto_grouped_filtered_adata)

        clusters_df = processed_adata.obs[['id', 'clusters']].reset_index()
        cell_gdf = cell_gdf.merge(clusters_df, on='id', how='left')

        # Save outputs
        resource_use_df = pd.DataFrame([stats], index=["cellpose_cyto3"])
        resource_use_df.to_csv(output.resource_usage)
        cell_gdf.to_file(output.cyto_gdf, driver="GPKG")
        processed_adata.write(output.cellpose_cyto_adata)

rule cyto_plotting:
    input:
        adata = "cyto_filtered_{sample}_adata.h5ad",
        gdf = "{sample}_cyto.gpkg",
    output:
        cluster_plot = "plots/cyto_{sample}_cluster.png",
        umi_plot = "plots/cyto_{sample}_umi.png",
        spatial_plot = "plots/cyto_{sample}_spatial_visualization.png",
        area_plot = "plots/cyto_{sample}_area.png"
    params:
        image=config["image_path"],
        label = lambda wildcards: wildcards.sample
    resources:
        mem_mb=15000
    run:
    # Load inputs
        gdf = gpd.read_file(input.gdf)
        adata = anndata.read_h5ad(input.adata)
        img = imread(params.image)

        print(f"Image shape: {img.shape}")

        cluster_plot_path = output.cluster_plot
        umi_plot_path = output.umi_plot
        sample_label = params.label
        spatial_plot_path = output.spatial_plot
        area_plot_path = output.area_plot

    # Plot 1: Cluster Distribution Bar Plot
        adata.obs['clusters'].value_counts().sort_index().plot(kind='bar')
        plt.xlabel('Cluster')
        plt.title(f'{sample_label}: Cluster Distribution')
        plt.ylabel('Number of Cells')
        plt.tight_layout()
        plt.savefig(cluster_plot_path)
        plt.close()

    # Plot 2: UMI Counts Histogram
        values = adata.obs['total_counts']
        clusters = adata.obs['clusters']

        # Get sorted unique cluster labels
        unique_clusters = sorted(clusters.unique(), key=lambda x: int(x))
        n_clusters = len(unique_clusters)

        # Define color for each cluster (consistent)
        cmap = plt.cm.tab20
        cluster_colors = {cluster: cmap(i / n_clusters) for i, cluster in enumerate(unique_clusters)}

        # Prepare data for stacked histogram
        cluster_values_list = [values[clusters == cluster] for cluster in unique_clusters]
        colors_list = [cluster_colors[cluster] for cluster in unique_clusters]

        # Set up subplots (1 for combined + n_clusters)
        n_cols = 4
        n_rows = int(np.ceil((n_clusters + 1) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        # Plot combined stacked histogram in first subplot
        ax = axes[0]
        ax.hist(
            cluster_values_list, 
            bins=50, 
            stacked=True, 
            color=colors_list, 
            label=[f'Cluster {cluster}' for cluster in unique_clusters],
            alpha=0.85
        )
        ax.set_title('All Clusters Combined')
        ax.set_xlabel('UMI Counts per Cell')
        ax.set_ylabel('Number of Cells')
        ax.legend(fontsize='x-small', frameon=False)

        # Plot individual histograms
        for i, cluster in enumerate(unique_clusters, start=1):
            ax = axes[i]
            cluster_values = values[clusters == cluster]
            ax.hist(
                cluster_values, 
                bins=50, 
                color=cluster_colors[cluster], 
                alpha=0.85
            )
            ax.set_title(f'Cluster {cluster}')
            ax.set_xlabel('UMI Counts per Cell')
            ax.set_ylabel('Number of Cells')

        # Hide any extra empty subplots
        for j in range(n_clusters + 1, len(axes)):
            axes[j].axis('off')
        fig.suptitle(f'{sample_label} UMI Counts per Cell', fontsize=16, y=.98)
        plt.tight_layout()
        plt.savefig(umi_plot_path)
        plt.close()

    # Plot 3: Spatial Visualization
        xmin, ymin, xmax, ymax = gdf.total_bounds
        # if any are negative set to 0
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xlim = (xmin,xmax)
        ylim = (ymin,ymax)

        fig, ax = plt.subplots(figsize=(10, 10))
        gdf = gdf.set_crs(None, allow_override=True)
        gdf.plot(ax=ax, column='clusters', cmap='viridis', edgecolor='black', legend=True, alpha=0.7)
        cropped_img = img[int(ylim[0]):int(ylim[1]), int(xlim[0]):int(xlim[1])]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{sample_label} Segmentation w/ Cell Boundries in Black & colored by Cluster')
        ax.imshow(cropped_img, cmap='gray', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin='lower')
        plt.tight_layout()
        plt.savefig(spatial_plot_path)
        plt.close()

    # Plot 4: Area of Cell
        gdf['area'] = gdf['geometry'].area

        # drop NA
        gdf = gdf.dropna(subset=['clusters'])

        areas = gdf['area']
        clusters = gdf['clusters'] 

        # Get sorted unique cluster labels
        unique_clusters = sorted(clusters.unique(), key=lambda x: int(x))
        n_clusters = len(unique_clusters)

        # Define consistent colors
        cmap = plt.cm.tab20
        cluster_colors = {cluster: cmap(i / n_clusters) for i, cluster in enumerate(unique_clusters)}

        # Prepare data for stacked histogram
        cluster_areas_list = [areas[clusters == cluster] for cluster in unique_clusters]
        colors_list = [cluster_colors[cluster] for cluster in unique_clusters]

        # Set up subplots (1 combined + individual clusters)
        n_cols = 4
        n_rows = int(np.ceil((n_clusters + 1) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        # Combined stacked histogram for all clusters
        ax = axes[0]
        ax.hist(
            cluster_areas_list,
            bins=50,
            stacked=True,
            color=colors_list,
            label=[f'Cluster {cluster}' for cluster in unique_clusters],
            alpha=0.85
        )
        ax.set_title('All Clusters Combined')
        ax.set_xlabel('Area')
        ax.set_ylabel('Number of Cells')
        ax.legend(fontsize='x-small', frameon=False)

        # Individual cluster histograms
        for i, cluster in enumerate(unique_clusters, start=1):
            ax = axes[i]
            cluster_areas = areas[clusters == cluster]
            ax.hist(
                cluster_areas,
                bins=50,
                color=cluster_colors[cluster],
                alpha=0.85
            )
            ax.set_title(f'Cluster {cluster}')
            ax.set_xlabel('Area')
            ax.set_ylabel('Number of Cells')

        # Hide empty subplots
        for j in range(n_clusters + 1, len(axes)):
            axes[j].axis('off')

        fig.suptitle(f'{sample_label} Area of Cell', fontsize=16, y=.98)
        plt.tight_layout()
        plt.savefig(area_plot_path)
        plt.close()

rule bin2cell_nuclei:
    input:
        gdf_cords="gdf_coordinates.gpkg"
    output:
        nuc_gdf="bin2cell_nuclei.gpkg",
        nuc_adata="nuc_filtered_bin2cell_adata.h5ad",
        nuc_expanded_adata=temp("nuc_nonfiltered_bin2cell_adata.h5ad"), #temp output for cyto segmentation 
        resource_usage_cyto="resource_usage/bin2cell_cyto.csv"
    resources:
        gpu=1
    params:
        image=config["image_path"],
        square_002um=config["square_002um"],
        spatial=config["spatial"],
        region_x=config["region_x"],
        region_y=config["region_y"]
    shell:
        # Passing in default parameters
        """
        python {workflow.basedir}/bin2cell_nuclei.py \
            --gdf_cords {input.gdf_cords} \
            --img {params.image} \
            --square_002um {params.square_002um} \
            --spatial {params.spatial} \
            --region_x {params.region_x} \
            --region_y {params.region_y} \
            --nuc_gdf {output.nuc_gdf} \
            --nuc_adata {output.nuc_adata} \
            --nuc_expanded_adata {output.nuc_expanded_adata} \
            --resource_usage_cyto {output.resource_usage_cyto}
        """
    
rule bin2cell_cyto:
    input:
        nuc_gdf="bin2cell_nuclei.gpkg",
        nuc_adata="nuc_nonfiltered_bin2cell_adata.h5ad",
    output:
        cell_gdf="bin2cell_cyto.gpkg",
        cell_adata="cyto_filtered_bin2cell_adata.h5ad"
    params:
        region_x=config["region_x"],
        region_y=config["region_y"]
    run:
        nuc_gdf = gpd.read_file(input.nuc_gdf)
        nuc_adata = anndata.read_h5ad(input.nuc_adata)
        region_x = params.region_x
        region_y = params.region_y
        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])

        #convert wkt to shapely geometry
        nuc_adata.obs['geometry'] = nuc_adata.obs['geometry'].apply(wkt.loads)

        # get expanded nuclei gdf ie cells
        gdf = gpd.GeoDataFrame(nuc_adata.obs, geometry='geometry')
        # Drop NaN labels and label 0 (which are not cells)
        gdf = gdf.dropna(subset=['labels_joint'])
        gdf = gdf[gdf['labels_joint'] != 0]

        # Grouping object
        grouped_cell = gdf.groupby('labels_joint')

        # Function to make convex hull or buffer
        def make_polygon_or_buffer(pts):
            if len(pts) == 1:
                return pts.iloc[0].buffer(1)
            else:
                return MultiPoint(list(pts)).convex_hull
            
        cell_polygons = grouped_cell.geometry.apply(make_polygon_or_buffer)

        cell_gdf = gpd.GeoDataFrame(cell_polygons, geometry=cell_polygons)
        cell_gdf = cell_gdf.rename(columns={0: 'geometry'}).set_geometry('geometry')
        cell_gdf = cell_gdf.reset_index()  

        # get total gene expression for each cell (ie expanded nuclei)
        sc.pp.filter_genes(nuc_adata, min_cells=1)

        groupby_object = nuc_adata.obs.groupby(['labels_joint'], observed=True)

        # Extract the gene expression counts from the AnnData object
        counts = nuc_adata.X

        # Obtain the number of unique cells and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each cell
        summed_counts = sparse.lil_matrix((N_groups, N_genes))

        # Create a list for polygon IDs (labels_joint)
        polygon_id = list(groupby_object.indices.keys())

        # Map labels_joint values to row indices
        label_to_row = {label: i for i, label in enumerate(polygon_id)}

        # Sum counts for each group
        for label, idx_ in groupby_object.indices.items():
            row = label_to_row[label]
            summed_counts[row] = counts[idx_].sum(0)

        # Convert to CSR for efficient slicing
        summed_counts = summed_counts.tocsr()

        # Make obs DataFrame: one row per polygon
        obs_df = pd.DataFrame({'labels_joint': polygon_id}, index=polygon_id)

        # join obs_df with cell_gdf on index
        cell_gdf = cell_gdf.set_index('labels_joint')
        # Perform a left join: keep all rows from obs_df, add columns from cell_gdf where index matches
        obs_df = obs_df.merge(cell_gdf, left_index=True, right_index=True, how='left')

        # Optionally make this a GeoDataFrame if you want to retain spatial features
        obs_gdf = gpd.GeoDataFrame(obs_df, geometry='geometry')

        # Create new AnnData with summed counts and grouped obs
        cell_grouped_filtered_adata = anndata.AnnData(
            X=summed_counts,
            obs=obs_gdf,
            var=nuc_adata.var)

        processed_adata = norm_log_cluster_adata(cell_grouped_filtered_adata)

        processed_adata.obs['id'] = processed_adata.obs['labels_joint'].astype(str)
        clusters_df = processed_adata.obs[['id', 'clusters', 'geometry']].reset_index()
        cell_gdf = gpd.GeoDataFrame(clusters_df, geometry='geometry')
        cell_gdf['area'] = cell_gdf.geometry.area
        processed_adata.obs['geometry'] = processed_adata.obs['geometry'].apply(lambda geom: geom.wkt)
        processed_adata.obs['geometry'] = processed_adata.obs['geometry'].astype(str)
        processed_adata.obs = pd.DataFrame(processed_adata.obs)

        # Save outputs
        cell_gdf.to_file(output.cell_gdf, driver="GPKG")
        processed_adata.write(output.cell_adata)

rule voronoi_expansion_white_mask_filter:
    input:
        adata="adata_processed.h5ad",
        nuclei_gdf="{sample}_nuclei.gpkg",
        gdf_coords="gdf_coordinates.gpkg"
    output:
        cyto_gdf = "{sample}_voronoi_cyto.gpkg",
        cyto_adata = "cyto_filtered_{sample}_voronoi_adata.h5ad",
        resource_usage="resource_usage/{sample}_voronoi.csv"
    params:
        image=config["image_path"],
        label = lambda wildcards: wildcards.sample,
        region_x=config["region_x"],
        region_y=config["region_y"]
    resources:
        gpu=1
    run:
        def voronoi_call(gdf):
            if torch.cuda.is_available():   
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()  
                torch_gpu_before = torch.cuda.memory_allocated() / 1e6
            else:
                torch_gpu_before = 0
                
            # Prepare TF devices
            gpus = tf.config.list_physical_devices('GPU')
            tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
            
            tf_usage_history = []
            running_flag = [True]  # mutable container so thread can see updates

            # Start monitoring
            t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
            t.start()
            start = time.time()
            
            centers = []
            for polygonal in gdf.geometry:
                center = polygonal.centroid
                centers.append([center.x, center.y])
            vor = Voronoi(centers)

            end = time.time() - start
            torch_gpu_after = torch.cuda.memory_allocated() / 1e6
            torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
            tf_peak = max(tf_usage_history) if len(tf_usage_history) > 0 else 0.0

            print(f"Voronoi GPU RAM before: {torch_gpu_before:.2f} MB")
            print(f"Voronoi GPU RAM peak:   {torch_gpu_peak:.2f} MB")
            print(f"Voronoi GPU RAM after:  {torch_gpu_after:.2f} MB")
            print(f"Voronoi segmentation time: {end:.2f} seconds")
            print(f"Peak TensorFlow memory: {(tf_peak):.2f} MB")

            # Stop monitoring
            running_flag[0] = False
            t.join()
            stats = {
            "torch_gpu_before": torch_gpu_before,
            "torch_peak": torch_gpu_peak,
            "torch_after": torch_gpu_after,
            "segmentation_time": end,
            "tensorflow_peak": tf_peak
            }
            
            return vor, stats

        def make_cytoplasm_polygon(points, buffer):
            if len(points) == 0:
                return None
            if len(points) == 1:
                return points[0].buffer(buffer)
            else:
                return MultiPoint(list(points)).convex_hull.buffer(buffer)

        def voronoi_finite_polygons_2d(vor, radius):
            if vor.points.shape[1] != 2:
                raise ValueError("Requires 2D Voronoi input")

            new_regions = []
            new_vertices = vor.vertices.tolist()
            center = vor.points.mean(axis=0)
            all_ridges = defaultdict(list)

            for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
                all_ridges[p1].append((p2, v1, v2))
                all_ridges[p2].append((p1, v1, v2))

            for point_index, region_index in enumerate(vor.point_region):
                region = vor.regions[region_index]
                if -1 not in region:
                    new_regions.append(region)
                    continue

                new_region = [vertex for vertex in region if vertex >= 0]
                for neighbor_index, v1, v2 in all_ridges[point_index]:
                    if v2 < 0:
                        v1, v2 = v2, v1
                    if v1 >= 0:
                        continue

                    tangent = vor.points[neighbor_index] - vor.points[point_index]
                    tangent_norm = np.linalg.norm(tangent)
                    if tangent_norm == 0:
                        continue
                    tangent /= tangent_norm
                    normal = np.array([-tangent[1], tangent[0]])
                    midpoint = vor.points[[point_index, neighbor_index]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, normal)) * normal
                    far_point = vor.vertices[v2] + direction * radius

                    new_region.append(len(new_vertices))
                    new_vertices.append(far_point.tolist())

                region_vertices = np.asarray([new_vertices[vertex] for vertex in new_region])
                region_center = region_vertices.mean(axis=0)
                angles = np.arctan2(
                    region_vertices[:, 1] - region_center[1],
                    region_vertices[:, 0] - region_center[0],
                )
                new_region = [vertex for _, vertex in sorted(zip(angles, new_region))]
                new_regions.append(new_region)

            return new_regions, np.asarray(new_vertices)

        nuclei_gdf = gpd.read_file(input.nuclei_gdf)
        
        mem_usage, (vor, stats) = memory_usage(
            (voronoi_call, (nuclei_gdf,), {}),
            retval=True)
        
        stats["cpu_peak"] = max(mem_usage)
        print(f"Voronoi Peak memory used: {max(mem_usage):.2f} MB")
        region_x = params.region_x
        region_y = params.region_y
        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])
        
        bbox=box(xmin, ymin, xmax, ymax)
        
        

        vor_regions, vor_vertices = voronoi_finite_polygons_2d(
            vor,
            radius=max(xmax - xmin, ymax - ymin) * 2,
        )

        polygons = []
        for region in vor_regions:
            polygonal = Polygon(vor_vertices[region]).intersection(bbox)
            if not polygonal.is_empty and not polygonal.is_valid:
                polygonal = polygonal.buffer(0)
            polygons.append(polygonal)

        nuclei_gdf = nuclei_gdf.copy()
        nuclei_gdf['nuc_centroid'] = nuclei_gdf.geometry.centroid
        nuclei_gdf['nuc_polygon'] = nuclei_gdf.geometry
        vor_gdf = gpd.GeoDataFrame(
            {
                'cell_id': [f"cell_vor_{i+1}" for i in range(len(polygons))],
                'nuclei_id': nuclei_gdf['id'].to_numpy(),
                'nuclei_cluster': nuclei_gdf['clusters'].to_numpy(),
                'nuc_centroid': nuclei_gdf['nuc_centroid'].to_numpy(),
                'nuc_polygon': nuclei_gdf['nuc_polygon'].to_numpy(),
            },
            geometry=polygons,
        )
        vor_gdf = vor_gdf[vor_gdf.geometry.notna() & ~vor_gdf.geometry.is_empty].copy()
        
        # calc cell area
        vor_gdf['area'] = vor_gdf.geometry.area
        vor_gdf['cell_polygon'] = vor_gdf.geometry
        
        gdf_coords = gpd.read_file(input.gdf_coords)
        
        # Perform a spatial join to check which coordinates are in a cell
        result_spatial_join = gpd.sjoin(gdf_coords, vor_gdf, how='left', predicate='within')

        # Identify cell associated barcodes
        result_spatial_join['is_within_cell'] = ~result_spatial_join['index_right'].isna()
        barcodes_in_cytoplasm = result_spatial_join[result_spatial_join['is_within_cell']]


        adata = anndata.read_h5ad(input.adata)

        # Filter the anndata to only contain points in cytoplasmic regions
        cyto_obs_mask = adata.obs_names.isin(barcodes_in_cytoplasm['barcode'])
        cyto_adata = adata[cyto_obs_mask,:]
        result_spatial_join.set_index('barcode', inplace=True)
        cyto_adata.obs = pd.merge(
            cyto_adata.obs,
            result_spatial_join[
                ['cell_id', 'cell_polygon', 'nuc_polygon', 'is_within_cell', 'nuc_centroid', 'nuclei_cluster', 'area', 'nuclei_id', 'geometry']
            ],
            left_index=True,
            right_index=True,
        )
        
        # filter adata to region of interest for better compute time
        mask = (
            (cyto_adata.obs['geometry'].apply(lambda p: p.x) >= xmin) &
            (cyto_adata.obs['geometry'].apply(lambda p: p.x) <= xmax) &
            (cyto_adata.obs['geometry'].apply(lambda p: p.y) >= ymin) &
            (cyto_adata.obs['geometry'].apply(lambda p: p.y) < ymax)
        )

        roi_cyto_adata = cyto_adata[mask]
        
        img = imread(params.image)
        ## White Masking
        mean_rgb = img.mean(axis=2, dtype=np.float32)
        std_rgb = img.std(axis=2, dtype=np.float32)

        mask = (mean_rgb < 220) & (std_rgb > 10)

        # Expand mask by 1 pixel in all directions
        expanded_mask = binary_dilation(mask, iterations=1)

        height, width = mask.shape

        def is_in_tissue_mask(point, mask):
            x, y = int(point.x), int(point.y)
            # Check bounds first to avoid index errors
            if 0 <= x < width and 0 <= y < height:
                return mask[y, x]  # Note: numpy arrays are indexed as [row, col] == [y, x]
            else:
                return False

        roi_cyto_adata.obs['is_in_tissue'] = roi_cyto_adata.obs['geometry'].apply(lambda pt: is_in_tissue_mask(pt, expanded_mask))
        roi_cyto_adata = roi_cyto_adata[roi_cyto_adata.obs['is_in_tissue']].copy()
        
        # Ensure GeoDataFrame
        gdf = gpd.GeoDataFrame(roi_cyto_adata.obs, geometry='geometry')

        # Define buffer size (adjust based on your coordinate units)
        buffer_size = 2 

        # Group by 'cell_id', compute convex hull, and buffer it
        polygons = (
            gdf.groupby('cell_id', observed=True)['geometry']
            .apply(lambda points: make_cytoplasm_polygon(points, buffer_size))
            .reset_index(name='cytoplasm_polygon'))

        # Drop empty results if needed
        rounded_polygons = polygons.dropna(subset=['cytoplasm_polygon'])

        # join rounded_polygons geometry (named polished_cell_polygon to roi_cyto_adata on 'cell_id
        roi_cyto_adata.obs = roi_cyto_adata.obs.merge(rounded_polygons, on='cell_id', how='left')
        
        # eliminate non-expressed genes just to decrease sparsity of the matrix via 0 expressed rows
        sc.pp.filter_genes(roi_cyto_adata, min_cells=1)

        # Group the data by unique cell IDs
        groupby_object = roi_cyto_adata.obs.groupby(['cell_id'], observed=True)

        # Extract the gene expression counts from the AnnData object
        counts = roi_cyto_adata.X

        # Obtain the number of unique cells and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each cell
        summed_counts = sparse.lil_matrix((N_groups, N_genes))
        # Lists to store the IDs of polygons and the current row index
        polygon_id = []
        row = 0

        # Iterate over each unique polygon to calculate the sum of gene counts.
        for polygons, idx_ in groupby_object.indices.items():
            summed_counts[row] = counts[idx_].sum(0)
            row += 1
            polygon_id.append(polygons)
        print("done")
        # Create and AnnData object from the summed count matrix
        summed_counts = summed_counts.tocsr()
        print("done")

        voronoi_cyto_grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=roi_cyto_adata.var)
        cell_gdf = rounded_polygons
        print(f"shape: {voronoi_cyto_grouped_filtered_adata.shape}")
        sc.pp.filter_genes(voronoi_cyto_grouped_filtered_adata, min_cells=1)
        print(f"shape after filtering: {voronoi_cyto_grouped_filtered_adata.shape}")

        processed_adata = norm_log_cluster_adata(voronoi_cyto_grouped_filtered_adata)

        clusters_df = processed_adata.obs[['id', 'clusters']].reset_index()
        cell_gdf = cell_gdf.merge(clusters_df, left_on='cell_id', right_on='id', how='left').drop(columns='id')
        cell_gdf = cell_gdf.rename(columns={'cytoplasm_polygon': 'geometry'})
        cell_gdf['area'] = cell_gdf['geometry'].area

        # Save outputs
        resource_usage_df = pd.DataFrame([stats], index=[f"{wildcards.sample}_voronoi"])
        resource_usage_df.to_csv(output.resource_usage)
        cell_gdf.to_file(output.cyto_gdf, driver="GPKG")
        processed_adata.write(output.cyto_adata)


rule proximity_cyto:
    input:
        adata="adata_processed.h5ad",
        nuclei_gdf="{sample}_nuclei.gpkg",
        gdf_coords="gdf_coordinates.gpkg"
    output:
        cyto_gdf = "{sample}_proximity_cyto.gpkg",
        cyto_adata = "cyto_filtered_{sample}_proximity_adata.h5ad",
        resource_usage = "resource_usage/{sample}_proximity.csv"
    params:
        label = lambda wildcards: wildcards.sample,
        region_x=config["region_x"],
        region_y=config["region_y"]
    resources:
        gpu=1
    run:
        nuc_gdf = gpd.read_file(input.nuclei_gdf)
        adata = anndata.read_h5ad(input.adata)
        gdf_cords = gpd.read_file(input.gdf_coords)
        region_x = params.region_x
        region_y = params.region_y
        xmin, xmax = int(region_x[0]), int(region_x[1])
        ymin, ymax = int(region_y[0]), int(region_y[1])

        def proximity_call(nuc_gdf, filtered_adata, xmin, xmax, ymin, ymax):
            if torch.cuda.is_available():   
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()  
                torch_gpu_before = torch.cuda.memory_allocated() / 1e6
            else:
                torch_gpu_before = 0

            # Prepare TF devices
            gpus = tf.config.list_physical_devices('GPU')
            tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]
            
            tf_usage_history = []
            running_flag = [True]  # mutable container so thread can see updates

            # Start monitoring
            t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
            t.start()
            start = time.time()
            
            # filter to ROI w/ some cushioining for cell expansion for faster compute time 
            filtered_adata = filtered_adata[
                (filtered_adata.obs['pxl_row_in_fullres'] > ymin-50) & 
                (filtered_adata.obs['pxl_row_in_fullres'] < ymax+50) &
                (filtered_adata.obs['pxl_col_in_fullres'] > xmin-50) & 
                (filtered_adata.obs['pxl_col_in_fullres'] < xmax+50)
            ]

            # make geometry column out of cords
            filtered_adata.obs['geometry'] = filtered_adata.obs.apply(lambda row: Point(round(row['pxl_col_in_fullres'],3), round(row['pxl_row_in_fullres'], 3)), axis=1)
            
            # make centers
            nuc_gdf['center'] = nuc_gdf['geometry'].centroid

            # make gdf for calculating distances
            bin_cords = filtered_adata.obs
            bin_gdf = gpd.GeoDataFrame(bin_cords, geometry='geometry')   

            # Create a KDTree for fast nearest neighbor search
            points_coords = bin_gdf.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
            tree = cKDTree(points_coords)

            # Add empty columns to bin_gdf to fill in
            bin_gdf['nucleus_id'] = np.nan # associated nuclei cell ID
            bin_gdf['nucleus_cluster'] = np.nan # carry over clustering
            bin_gdf['min_distance'] = np.inf # each bin can ONLY be denoted to the nearest nucleus

            # Loop through each nucleus, get 10 nearest points, and label them
            for _, nuc in nuc_gdf.iterrows():
                # Find indices and distances of nearest points
                dists, idxs = tree.query((nuc['center'].x, nuc['center'].y), k=min(10, len(bin_gdf)))

                # Ensure idxs and dists are arrays
                if np.isscalar(idxs):
                    idxs = [idxs]
                    dists = [dists]

                for i, idx in enumerate(idxs):
                    # if new cell nuc center is closer to bin then previous overwrite old cell ID
                    if dists[i] < bin_gdf.iloc[idx, bin_gdf.columns.get_loc('min_distance')]:
                        bin_gdf.iloc[idx, bin_gdf.columns.get_loc('min_distance')] = dists[i]
                        bin_gdf.iloc[idx, bin_gdf.columns.get_loc('nucleus_id')] = nuc['id']
                        bin_gdf.iloc[idx, bin_gdf.columns.get_loc('nucleus_cluster')] = nuc['clusters']

            # rename
            inside_nuclei.rename(columns={'index_right': 'nucleus_id'}, inplace=True)
            inside_nuclei.rename(columns={'index_left': 'index'}, inplace=True)
            inside_nuclei.rename(columns={'clusters': 'nucleus_cluster'}, inplace=True)

            # combine cyto and nuclei gdfs for whole cell groups
            cyto_nuc_combined = pd.concat([bin_gdf[['index', 'nucleus_id', 'nucleus_cluster', 'geometry']], inside_nuclei[['index', 'nucleus_id', 'nucleus_cluster', 'geometry']]], axis=0)

            # In some cases bins can be denoted to multiple nuclei - resolve here by assigning duplicated bins to closest nuc center
            dups = cyto_nuc_combined['index'].duplicated(keep=False)

            duplicate_rows = cyto_nuc_combined[dups]

            if len(duplicate_rows) != 0:
                print("Duplicate indices found in cyto_nuc_combined:")
                print(duplicate_rows)

                best_matches = []

                for idx, group in duplicate_rows.groupby(level=0):  # group by index
                    bin_geom = group.iloc[0].geometry
                    # if isinstance(bin_geom, gpd.GeoSeries):
                    #    bin_geom = bin_geom.iloc[0]  # in case of multiple geometries (due to duplicates)

                    min_dist = float('inf')
                    best_row = None

                    for _, row in group.iterrows():
                        nucleus_id = row['nucleus_id']
                        if pd.isna(nucleus_id):
                            continue
                        try:
                            nucleus_geom = nuc_gdf.loc[nucleus_id].geometry
                            nuc_center = nucleus_geom.centroid
                            dist = bin_geom.centroid.distance(nuc_center)

                            if dist < min_dist:
                                min_dist = dist
                                best_row = row
                        except KeyError:
                            continue  # skip if nucleus_id not found in nuc_gdf

                    if best_row is not None:
                        best_matches.append(best_row)

                # cleaned DataFrame with no duplicates
                cleaned = cyto_nuc_combined[~dups].copy()
                cleaned = pd.concat([cleaned, gpd.GeoDataFrame(best_matches, crs=cyto_nuc_combined.crs)])
            else:
                # no duplicates, so cleaned is just the combined gdf
                cleaned = cyto_nuc_combined.copy()

            end = time.time() - start
            torch_gpu_after = torch.cuda.memory_allocated() / 1e6
            torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
            print(f"Proximity GPU RAM before: {torch_gpu_before:.2f} MB")
            print(f"Proximity GPU RAM peak:   {torch_gpu_peak:.2f} MB")
            print(f"Proximity GPU RAM after:  {torch_gpu_after:.2f} MB")
            print(f"Proximity segmentation time: {end:.2f} seconds")
            print(f"Proximity Peak TensorFlow memory: {max(tf_usage_history):.2f} MB")
            
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
            
            return cleaned, stats

        # Identify bins in nuclei
        result_spatial_join = gpd.sjoin(gdf_cords, nuc_gdf, how='left', predicate='within')

        # Label nuclei rows
        result_spatial_join['is_within_nuclei'] = ~result_spatial_join['index_right'].isna()
        
        # Remove barcodes in nuclei for cytoplasm expansions
        barcodes_in_cyto = result_spatial_join[~result_spatial_join['is_within_nuclei']]
        filtered_obs_mask = adata.obs['index'].isin(barcodes_in_cyto['barcode'])
        filtered_adata = adata[filtered_obs_mask,:]
        
        # Points inside nuclei for full cell downstream
        inside_nuclei = result_spatial_join[result_spatial_join['is_within_nuclei']].copy()
        inside_nuclei['assignment_type'] = 'inside_nucleus'


        mem_usage, (cleaned, stats) = memory_usage(
            (proximity_call, (nuc_gdf, filtered_adata, xmin, xmax, ymin, ymax), {}),
            retval=True)

        stats["cpu_peak"] = max(mem_usage)
        print(f"Proximity Peak memory used: {max(mem_usage):.2f} MB")

        cleaned.set_index('index', inplace=True)
        joined_df = pd.merge(adata.obs, cleaned, how='left', on='index')
        adata.obs = joined_df
        cell_adata = adata[adata.obs['nucleus_id'].notna()]

        # Eliminate non-expressed genes just for less sparsity
        sc.pp.filter_genes(cell_adata, min_cells=1)

        cell_adata = cell_adata[cell_adata.obs['nucleus_id'].notna(), :]
        # Group the data by unique nucleous IDs
        groupby_object = cell_adata.obs.groupby(['nucleus_id'], observed=True)

        convex_hulls = groupby_object['geometry'].apply(
            lambda geoms: unary_union(geoms).convex_hull)

        # Extract the gene expression counts from the AnnData object
        counts = cell_adata.X

        # Obtain the number of unique nuclei and the number of genes in the expression data
        N_groups = groupby_object.ngroups
        N_genes = counts.shape[1]

        # Initialize a sparse matrix to store the summed gene counts for each nucleus
        summed_counts = sparse.lil_matrix((N_groups, N_genes))

        # Lists to store the IDs of polygons and the current row index
        nuclei_id = []
        row = 0

        # Iterate over each unique polygon to calculate the sum of gene counts.
        for nuc_id, idx_ in groupby_object.indices.items():
            summed_counts[row] = counts[idx_].sum(0)
            row += 1
            nuclei_id.append(nuc_id)
        # Create and AnnData object from the summed count matrix
        summed_counts = summed_counts.tocsr()

        # Pull nucleus_cluster info: assume cluster is same for all cells in group, so take first
        nucleus_clusters = groupby_object['nucleus_cluster'].first()

        # Build obs DataFrame with id, cluster, and geometry (convex hull)
        obs_df = pd.DataFrame({
            'id': nuclei_id,
            'nucleus_cluster': nucleus_clusters.reindex(nuclei_id).values,
            'geometry': convex_hulls.reindex(nuclei_id).values
        }, index=nuclei_id)

        # Create new AnnData object
        cell_grouped_filtered_adata = anndata.AnnData(
            X=summed_counts,
            obs=obs_df,
            var=cell_adata.var
        )

        processed = norm_log_cluster_adata(cell_grouped_filtered_adata)

        df = processed.obs
        cell_gdf = gpd.GeoDataFrame(df, geometry='geometry')
        cell_gdf['area'] = cell_gdf['geometry'].area
        processed.obs['geometry'] = processed.obs['geometry'].apply(lambda g: g.wkt)

        # Save outputs
        cell_gdf.to_file(output.cyto_gdf, driver="GPKG")
        processed.write(output.cyto_adata)
        resource_usage_df = pd.DataFrame([stats], index=[f"{wildcards.sample}_proximity"])
        resource_usage_df.to_csv(output.resource_usage)


rule STP_tuned_cyto:
    input:
        gdf="gdf_coordinates.gpkg",
        adata="adata_processed.h5ad"
    params:
        region_x=config["region_x"],
        region_y=config["region_y"],
        img=config['image_path']
    resources:
        gpu=1
    conda:
        "envs/STP_env.yaml"
    output:
        cyto_gdf="STP_tuned_cyto.gpkg",
        cyto_adata="cyto_filtered_STP_tuned_adata.h5ad",
        resource_usage = "resource_usage/STP_tuned.csv"
    shell:
        """
        python {workflow.basedir}/STP.py \
            --gdf {input.gdf} \
            --adata {input.adata} \
            --img {params.img} \
            --region_x {params.region_x} \
            --region_y {params.region_y} \
            --cyto_gdf {output.cyto_gdf} \
            --cyto_adata {output.cyto_adata} \
            --resource_usage {output.resource_usage} \
            --tuned True \
            --thres 1.63 \
            --T 200 \
            --T_min 5 \
            --reduction_rate 0.85 \
            --neighbor_num 15 \
            --alpha 0.5 \
            --beta 0.05
        """

rule STP_untuned_cyto:
    input:
        gdf="gdf_coordinates.gpkg",
        adata="adata_processed.h5ad"
    params:
        region_x=config["region_x"],
        region_y=config["region_y"],
        img=config['image_path']
    resources:
        gpu=1
    conda:
        "envs/STP_env.yaml"
    output:
        cyto_gdf="STP_untuned_cyto.gpkg",
        cyto_adata="cyto_filtered_STP_untuned_adata.h5ad",
        resource_usage = "resource_usage/STP_untuned.csv"
    shell:
        # Passing in default parameters
        """
        python {workflow.basedir}/STP.py \
            --gdf {input.gdf} \
            --adata {input.adata} \
            --img {params.img} \
            --region_x {params.region_x} \
            --region_y {params.region_y} \
            --cyto_gdf {output.cyto_gdf} \
            --cyto_adata {output.cyto_adata} \
            --resource_usage {output.resource_usage} \
            --tuned False \
            --thres .8 \
            --T 100 \
            --T_min 10 \
            --reduction_rate 0.5 \
            --neighbor_num 3 \
            --alpha 0.5 \
            --beta 0.1
        """
rule cyto_plot_gen:
    input:
        combo_adata = expand("cyto_filtered_{combo}_adata.h5ad", combo=PLOTTING_COMBO),
        combo_gdf   = expand("{combo}_cyto.gpkg", combo=PLOTTING_COMBO),
        cyto_adata  = expand("cyto_filtered_{cyto}_adata.h5ad", cyto=CYTO_TOOLS),
        cyto_gdf    = expand("{cyto}_cyto.gpkg", cyto=CYTO_TOOLS),
        nuc_adata   = expand("nuc_filtered_{nuc}_adata.h5ad", nuc=NUC_TOOLS),
        nuc_gdf     = expand("{nuc}_nuclei.gpkg", nuc=NUC_TOOLS),
        adata_processed = "adata_processed.h5ad",
        gdf_coords = "gdf_coordinates.gpkg"
    output:
        total_cells_csv = "plots/overview/total_cell_calls.csv",
        total_umi_csv = "plots/overview/total_UMI_per_cell.csv",
        area_csv = "plots/overview/total_area_per_cell.csv",
        umi_per_area_csv = "plots/overview/total_UMI_per_area_per_cell.csv",
        roi_transcripts_csv = "plots/overview/roi_transcripts_summary.csv",
        ARI_csv = "plots/overview/ARI_segmentation.csv"
    resources:
        gpu=1
    params:
        spatial = config['spatial'],
        region_x = config["region_x"],
        region_y = config["region_y"],
    run:
        xmin, xmax = int(params.region_x[0]), int(params.region_x[1])
        ymin, ymax = int(params.region_y[0]), int(params.region_y[1])

        # Load scalefactors
        with open(os.path.join(params.spatial, "scalefactors_json.json")) as f:
            scalefactors = json.load(f)
        microns_per_pixel = scalefactors["microns_per_pixel"]
        pixel_area_to_micron2 = microns_per_pixel ** 2

        # Calculate area of image 
        image_width = xmax - xmin
        image_height = ymax - ymin
        image_area_micron2 = image_width * image_height * pixel_area_to_micron2
        print(f"Image area in microns^2: {image_area_micron2}")

        # Load processed anndata and coordinates
        adata_processed = sc.read(input.adata_processed)
        gdf_coords = gpd.read_file(input.gdf_coords)

        roi_gdf_coords = gdf_coords.cx[xmin:xmax, ymin:ymax]
        roi_adata_processed = adata_processed[adata_processed.obs.index.isin(roi_gdf_coords['barcode'])]
        roi_total_transcripts = roi_adata_processed.X.sum()

        # Prepare output containers
        cell_counts_data, umi_data, area_data, umi_per_area_data, roi_summary_data = ([] for _ in range(5))
        
        # Make dicts for ARI calculation: By cell and by cluster
        # Container: {sample_label: {barcode: cell_id}}
        cell_segmentation_assignments = defaultdict(dict)
        cluster_segmentation_assignments = defaultdict(dict)


        def process_pair(adata_file, gdf_file, label):
            adata = sc.read(adata_file)
            gdf = gpd.read_file(gdf_file)

            cell_counts_data.append({"Sample": label, "Cell_Count": adata.n_obs})

            if 'total_counts' not in adata.obs.columns:
                raise ValueError(f"'total_counts' missing in {adata_file}")
            umi_series = adata.obs['total_counts']
            unique_genes_series = adata.obs['n_genes_by_counts']
            umi_data.append(pd.DataFrame({'UMI_per_cell': umi_series.values, 'Sample': label, 'Unique_Genes': unique_genes_series.values}))

            if 'area' not in gdf.columns:
                raise ValueError(f"'area' missing in {gdf_file}")
            area_micron2 = gdf['area'] * pixel_area_to_micron2
            area_data.append(pd.DataFrame({'Area_per_cell': area_micron2.values, 'Sample': label}))

            # If there's a mismatch, pad missing UMIs with 0 to match number of polygons
            if len(umi_series) != len(area_micron2):
                print(f"\n[DEBUG] {adata_file} has {len(umi_series)} cells; {gdf_file} has {len(area_micron2)} polygons")

                # If fewer UMIs than polygons, pad with zeros
                if len(umi_series) < len(area_micron2):
                    pad_length = len(area_micron2) - len(umi_series)
                    umi_values = np.concatenate([umi_series.values, np.zeros(pad_length)])
                # If more UMIs than polygons then issue...
                elif len(umi_series) > len(area_micron2):
                    raise ValueError(f"More UMIs than polygons in {adata_file} vs {gdf_file}")

            else:
                umi_values = umi_series.values

            umi_per_area = umi_values / area_micron2.values
            umi_per_area_data.append(pd.DataFrame({'UMI_per_area': umi_per_area, 'Sample': label}))


            result_spatial_join = gpd.sjoin(roi_gdf_coords, gdf, how='left', predicate='within')
            result_spatial_join['is_within_polygon'] = ~result_spatial_join['index_right'].isna()
            barcodes_in_one_polygon = result_spatial_join[result_spatial_join['is_within_polygon']]
            filtered_obs_mask = roi_adata_processed.obs.index.isin(barcodes_in_one_polygon['index_left'])
            filtered_adata = roi_adata_processed[filtered_obs_mask, :]
            roi_segmented_transcripts = filtered_adata.X.sum()
            percent_trans_segmented = roi_segmented_transcripts / roi_total_transcripts * 100

            # area segmented
            roi_area = gdf['area'].sum() * pixel_area_to_micron2
            percent_area_segmented = roi_area / image_area_micron2 * 100
            
            # Assign cell ID per barcode for ARI calculation
            for idx, row in result_spatial_join.iterrows():
                barcode = row['index_left']
                cell_id = row['index_right'] if not pd.isna(row['index_right']) else '-1'  # use -1 for not in cell
                cell_segmentation_assignments[label][barcode] = cell_id
                if 'clusters' in row and not pd.isna(row['clusters']):
                    try:
                        cluster_segmentation_assignments[label][barcode] = int(row['clusters'])
                    except ValueError:
                        cluster_segmentation_assignments[label][barcode] = -1
                else:
                    cluster_segmentation_assignments[label][barcode] = -1
            roi_summary_data.append({
                "Sample": label,
                "ROI_Total_Transcripts": roi_total_transcripts,
                "ROI_Segmented_Transcripts": roi_segmented_transcripts,
                "Percent_Transcripts_Segmented": percent_trans_segmented,
                "Percent_Area_Segmented": percent_area_segmented
            })

        # Process PLOTTING_COMBO
        for adata_file, gdf_file in zip(input.combo_adata, input.combo_gdf):
            label = os.path.basename(adata_file).replace("_adata.h5ad", "")
            process_pair(adata_file, gdf_file, label)

        # Process CYTO_TOOLS
        for adata_file, gdf_file in zip(input.cyto_adata, input.cyto_gdf):
            label = os.path.basename(adata_file).replace("_adata.h5ad", "")
            process_pair(adata_file, gdf_file, label)

        # Process NUC_TOOLS
        for adata_file, gdf_file in zip(input.nuc_adata, input.nuc_gdf):
            label = os.path.basename(adata_file).replace("_adata.h5ad", "")
            process_pair(adata_file, gdf_file, label)

        # Save aggregated results
        pd.DataFrame(cell_counts_data).to_csv(output.total_cells_csv, index=False)
        pd.concat(umi_data, ignore_index=True).to_csv(output.total_umi_csv, index=False)
        pd.concat(area_data, ignore_index=True).to_csv(output.area_csv, index=False)
        pd.concat(umi_per_area_data, ignore_index=True).to_csv(output.umi_per_area_csv, index=False)
        pd.DataFrame(roi_summary_data).to_csv(output.roi_transcripts_csv, index=False)

        # Get unique method labels
        method_labels = list(cell_segmentation_assignments.keys())

        # Get list of barcodes present in region of interest
        barcodes = roi_gdf_coords['barcode'].values

        # Build DataFrame for cell segmentation labels
        cell_segmentation_df = pd.DataFrame(index=barcodes)
        for method in method_labels:
            labels = [cell_segmentation_assignments[method].get(bc, -1) for bc in barcodes]
            cell_segmentation_df[method] = labels

        # Build DataFrame for cluster segmentation labels
        cluster_segmentation_df = pd.DataFrame(index=barcodes)
        for method in method_labels:
            labels = [cluster_segmentation_assignments[method].get(bc, -1) for bc in barcodes]
            cluster_segmentation_df[method] = labels

        cluster_segmentation_df.fillna(-1, inplace=True)

        ari_results = []
        first = True
        for m1, m2 in itertools.combinations(method_labels, 2):
            # write out cell_segmentation_df[m1] and [m2] to file for testing
            if first:
                output_df = cell_segmentation_df[[m1, m2]]
                #output_df.to_csv(f"cell_segmentation_{m1}_vs_{m2}.csv", index=False)
                first = False
            cell_ari = adjusted_rand_score(cell_segmentation_df[m1], cell_segmentation_df[m2])
            cluster_ari = adjusted_rand_score(cluster_segmentation_df[m1], cluster_segmentation_df[m2])

            print(f"Cell ARI between {m1} and {m2}: {cell_ari:.3f}")
            print(f"Cluster ARI between {m1} and {m2}: {cluster_ari:.3f}")

            ari_results.append({
                "Method_1": m1,
                "Method_2": m2,
                "Cell_ARI": cell_ari,
                "Cluster_ARI": cluster_ari
            })
        
        # Save ARI results
        pd.DataFrame(ari_results).to_csv(output.ARI_csv, index=False)
