export nnUNet_raw="data/nnUNet_raw"
export nnUNet_preprocessed="data/nnUNet_preprocessed"
export nnUNet_results="data/nnUNet_results"

export nnUNet_n_proc_DA=8          # number of processes for data augmentation
export CUDA_VISIBLE_DEVICES=0      # which GPU to use

DATASET_ID=501                     # must match source/data/nnunet_dataset.py:DATASET_ID

# Plan + preprocess (run once per dataset)
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity


