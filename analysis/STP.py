import sys
import geopandas as gpd
import anndata
import argparse
import scanpy as sc
import os
import numpy as np
from skimage.measure import regionprops
from shapely.geometry import Polygon, LineString, Point
from scipy.spatial import ConvexHull
from shapely import affinity
import tifffile
import pandas as pd
import ast
from memory_profiler import memory_usage
import time
import torch
import threading
#import tensorflow as tf

sys.path.append("/uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/analysis/dev_files/stp/STP")

from STP_utils import STP

from scipy import sparse

parser = argparse.ArgumentParser(description="Run STP cytoplasmic segmentation and clustering.")
parser.add_argument("--tuned", type=ast.literal_eval, required=True)
parser.add_argument("--resource_usage", required=True)
parser.add_argument("--gdf", required=True)
parser.add_argument("--adata", required=True)
parser.add_argument("--img", required=True)
parser.add_argument("--region_x", type=int, nargs=2, required=True)
parser.add_argument("--region_y", type=int, nargs=2, required=True)
parser.add_argument("--cyto_gdf", required=True)
parser.add_argument("--cyto_adata", required=True)
parser.add_argument("--T", required=True, type=int)
parser.add_argument("--thres", required=True, type=float)
parser.add_argument("--T_min", required=True, type=int)
parser.add_argument("--reduction_rate", required=True, type=float)
parser.add_argument("--neighbor_num", required=True, type=int)
parser.add_argument("--alpha", required=True, type=float)
parser.add_argument("--beta", required=True, type=float)

args = parser.parse_args()
xmin, xmax = args.region_x
ymin, ymax = args.region_y

gdf_coords = gpd.read_file(args.gdf)
adata = anndata.read_h5ad(args.adata)

img = tifffile.imread(args.img)

# Crop the image
cropped_img = img[ymin:ymax, xmin:xmax]

# copy for stp and post process 
stp_adata = adata.copy()

# Mask stp_adata to region of interest
mask = (
    (stp_adata.obs['pxl_row_in_fullres'] >= ymin) &
    (stp_adata.obs['pxl_row_in_fullres'] < ymax) &
    (stp_adata.obs['pxl_col_in_fullres'] >= xmin) &
    (stp_adata.obs['pxl_col_in_fullres'] < xmax))

adata_trimmed = stp_adata[mask].copy()
adata_trimmed.obs['sum_expression'] = adata_trimmed.X.sum(axis=1).A1

# Assign spatial coordinates (ensure correct order)
adata_trimmed.obsm['spatial'] = adata_trimmed.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].to_numpy().astype(int)

# Shift coordinates to crop origin
adata_trimmed.obsm['spatial'][:, 0] -= ymin  # row/y
adata_trimmed.obsm['spatial'][:, 1] -= xmin  # col/x

#save_path = './stp_temp/'
# os.makedirs(save_path,exist_ok=True)
# STP_utils concatenates filenames directly onto save_path, so keep a trailing slash.
save_path = "./stp_temp_tuned/" if args.tuned else "./stp_temp_untuned/"
os.makedirs(save_path, exist_ok=True)
def monitor_gpu(tf_device_names, tf_usage_history, running_flag):
    while running_flag[0]:
        total_tf = 0
        for dev_name in tf_device_names:
            try:
                #mem_info = tf.config.experimental.get_memory_info(dev_name)
                mem_info = 0  # Placeholder since tf is not imported
                total_tf += mem_info['current']
            except Exception:
                pass  # ignore errors
        tf_usage_history.append(total_tf / 1e6)  # store in MB
        time.sleep(0.1)

def STP_call(adata, img, save_path, T, thres, T_min, reduction_rate, neighbor_num, alpha, beta):
    if torch.cuda.is_available():   
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  
        torch_gpu_before = torch.cuda.memory_allocated() / 1e6
    else:
        torch_gpu_before = 0
    # Prepare TF devices
    #gpus = tf.config.list_physical_devices('GPU')
    #tf_device_names = [f"GPU:{i}" for i in range(len(gpus))]

    #tf_usage_history = []
    tf_usage_history = [0]  # Initialize with 0 to avoid max() error if thread doesn't run
    running_flag = [True]  # mutable container so thread can see updates

    # Start monitoring
    #t = threading.Thread(target=monitor_gpu, args=(tf_device_names, tf_usage_history, running_flag))
    #t.start()
    start = time.time()

    STP(adata, img, save_path, T=T, T_min=T_min, reduction_rate=reduction_rate, neighbor_num=neighbor_num, alpha=alpha, beta=beta, thres=thres)
    
    
    end = time.time() - start
    torch_gpu_after = torch.cuda.memory_allocated() / 1e6
    torch_gpu_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"STP GPU RAM before: {torch_gpu_before:.2f} MB")
    print(f"STP GPU RAM peak:   {torch_gpu_peak:.2f} MB")
    print(f"STP GPU RAM after:  {torch_gpu_after:.2f} MB")
    print(f"STP segmentation time: {end:.2f} seconds")
    print(f"STP Peak TensorFlow memory: {max(tf_usage_history):.2f} MB")
    # Stop monitoring
    running_flag[0] = False
    #t.join()
    stats = {
    "torch_gpu_before": torch_gpu_before,
    "torch_peak": torch_gpu_peak,
    "torch_after": torch_gpu_after,
    "segmentation_time": end,
    "tensorflow_peak": max(tf_usage_history)
    }
    return stats

# Run STP
#STP(adata_trimmed, preprocessed_img_3ch, save_path, T=args.T, T_min=args.T_min, reduction_rate=args.reduction_rate, neighbor_num=args.neighbor_num, alpha=args.alpha, beta=args.beta)
mem_usage, (stats) = memory_usage(
            (STP_call, (adata_trimmed, cropped_img, save_path, args.T, args.thres, args.T_min, args.reduction_rate, args.neighbor_num, args.alpha, args.beta,), {}),
            retval=True)
        
print(f"Cellpose Peak memory used: {max(mem_usage):.2f} MB")

stats["cpu_peak"] = max(mem_usage)

# Load the mask file
path = os.path.join(save_path, 'connected_components.npy')
mask = np.load(path)

# Check shape and contents
print("Mask shape:", mask.shape)
print("Unique labels:", np.unique(mask))

# Generate region properties
props = regionprops(mask)

# Convert each region to a Shapely Polygon
polygons = []
labels = []

for prop in props:
    coords = prop.coords  # (row, col) pairs for the region
    if len(coords) >= 3:
        # Compute convex hull
        hull = ConvexHull(coords)
        hull_coords = coords[hull.vertices]
        # Create polygon
        polygon = Polygon([(col, row) for row, col in hull_coords])
    elif len(coords) == 2:
        # make a LineString if 2 points 
        polygon = LineString([(col, row) for row, col in coords])
    elif len(coords) == 1:
        # Single point as buffer tiny polygon
        polygon = Point(coords[0][1], coords[0][0]).buffer(0.5)
    else:
        continue  # skip empty regions

    polygons.append(polygon)
    labels.append(prop.label)

# Build a GeoDataFrame
cell_gdf = gpd.GeoDataFrame({'id': labels, 'geometry': polygons})

# Translate each polygon
cell_gdf['geometry'] = cell_gdf['geometry'].apply(lambda p: affinity.translate(p, xoff=xmin, yoff=ymin))

# Perform a spatial join to check which coordinates are in a cell
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
filtered_adata.obs = pd.merge(
        filtered_adata.obs,
        barcodes_in_one_polygon[['index','geometry','id','is_within_polygon','is_not_in_an_polygon_overlap']],
        left_on='index',  
        right_on='index',
        how='left'
        ).set_index('index')

# Group the data by unique cell IDs
groupby_object = filtered_adata.obs.groupby(['id'], observed=True)

# Extract the gene expression counts from the AnnData object
counts = filtered_adata.X

# Obtain the number of unique cell and the number of genes in the expression data
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
cell_grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)

cell_gdf['area'] = cell_gdf['geometry'].area

sc.pp.calculate_qc_metrics(cell_grouped_filtered_adata, inplace=True)
# Normalize total counts for each cell in the AnnData object
sc.pp.normalize_total(cell_grouped_filtered_adata, inplace=True)

# Logarithmize the values in the AnnData object after normalization
sc.pp.log1p(cell_grouped_filtered_adata)

# Identify highly variable genes in the dataset using the Seurat method
sc.pp.highly_variable_genes(cell_grouped_filtered_adata, flavor="seurat", n_top_genes=2000)

# Perform Principal Component Analysis (PCA) on the AnnData object
sc.pp.pca(cell_grouped_filtered_adata)

# Build a neighborhood graph based on PCA components
sc.pp.neighbors(cell_grouped_filtered_adata)

# Perform Leiden clustering on the neighborhood graph and store the results in 'clusters' column
sc.tl.leiden(cell_grouped_filtered_adata, resolution=0.35, key_added="clusters", random_state=0)

clusters_df = cell_grouped_filtered_adata.obs[['id', 'clusters']].reset_index()
cell_gdf = cell_gdf.merge(clusters_df, on='id', how='left')

if args.tuned:
    resource_usage_df = pd.DataFrame([stats], index=["STP_tuned"])
else:
    resource_usage_df = pd.DataFrame([stats], index=["STP_untuned"])

# Save outputs
resource_usage_df.to_csv(args.resource_usage)
cell_gdf.to_file(args.cyto_gdf, driver="GPKG")
cell_grouped_filtered_adata.write(args.cyto_adata)
