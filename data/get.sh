#!/bin/tcsh 
#SBATCH --partition=notchpeak-dtn
#SBATCH --account=dtn
#SBATCH --time=5:00:00
#SBATCH -o slurmjob-%j.out-%N
#SBATCH -e slurmjob-%j.err-%N
#SBATCH --job-name=data_download_cell_seg
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jax.lubkowitz@utah.edu

# Human Pancreas 
#mkdir -p /scratch/general/nfs1/u01531817/human_pancreas_proccessed
#ln -s /scratch/general/nfs1/u01531817/human_pancreas /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data/human_pancreas
#cd /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data/human_pancreas
#curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Pancreas/Visium_HD_Human_Pancreas_binned_outputs.tar.gz
#curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Pancreas/Visium_HD_Human_Pancreas_tissue_image.btf
#tar -xvzf Visium_HD_Human_Pancreas_binned_outputs.tar.gz
#rm Visium_HD_Human_Pancreas_binned_outputs.tar.gz



# Mouse Brain 

#mkdir mouse_brain
##cd mouse_brain
#curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_binned_outputs.tar.gz
#curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_tissue_image.tif
#tar -xvzf Visium_HD_Mouse_Brain_binned_outputs.tar.gz
#rm Visium_HD_Mouse_Brain_binned_outputs.tar.gz


# # # Mouse Small Intestine
# cd /scratch/general/nfs1/u01531817/
# mkdir -p mouse_intestine
# ln -s /scratch/general/nfs1/u01531817/mouse_intestine /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data
# cd mouse_intestine
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_tissue_image.btf
# tar -xvzf Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz
# rm Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz

# # # Mouse Embryo
# cd /scratch/general/nfs1/u01531817/
# mkdir -p mouse_embryo
# ln -s /scratch/general/nfs1/u01531817/mouse_embryo /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data
# cd mouse_embryo
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Mouse_Embryo/Visium_HD_Mouse_Embryo_binned_outputs.tar.gz
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/4.0.1/Visium_HD_Mouse_Embryo/Visium_HD_Mouse_Embryo_tissue_image.btf
# tar -xvzf Visium_HD_Mouse_Embryo_binned_outputs.tar.gz
# rm Visium_HD_Mouse_Embryo_binned_outputs.tar.gz

# # # Mouse Kidney
# cd /scratch/general/nfs1/u01531817/
# mkdir -p mouse_kidney
# ln -s /scratch/general/nfs1/u01531817/mouse_kidney /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data
# cd mouse_kidney
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Kidney/Visium_HD_Mouse_Kidney_binned_outputs.tar.gz
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Kidney/Visium_HD_Mouse_Kidney_tissue_image.btf
# tar -xvzf Visium_HD_Mouse_Kidney_binned_outputs.tar.gz
# rm Visium_HD_Mouse_Kidney_binned_outputs.tar.gz

# Human Colon Cancer
cd /scratch/general/nfs1/u01531817/
mkdir human_colon_new
ln -s /scratch/general/nfs1/u01531817/human_colon_new /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data/human_colon3
cd human_colon_new
curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz
curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Colon_Cancer/Visium_HD_Human_Colon_Cancer_tissue_image.btf
ar -xvzf Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz
rm Visium_HD_Human_Colon_Cancer_binned_outputs.tar.gz

# Human Lung Cancer
#  cd /scratch/general/nfs1/u01531817/
#  mkdir human_lung
#  ln -s /scratch/general/nfs1/u01531817/human_lung /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data
#  cd human_lung
#  curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_binned_outputs.tar.gz
#  curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1/Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_tissue_image.btf
#  tar -xvzf Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_binned_outputs.tar.gz
#  rm Visium_HD_Human_Lung_Cancer_HD_Only_Experiment1_binned_outputs.tar.gz

# Mouse Brain 
# cd /scratch/general/nfs1/u01531817/
# mkdir -p mouse_brain
# ln -s /scratch/general/nfs1/u01531817/mouse_brain /uufs/chpc.utah.edu/common/home/u1531817/20250523_cell_seg_eval/data
# cd mouse_brain
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_binned_outputs.tar.gz
# curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_tissue_image.tif
# tar -xvzf Visium_HD_Mouse_Brain_binned_outputs.tar.gz
# rm Visium_HD_Mouse_Brain_binned_outputs.tar.gz
