"""
# Single fold (most common starting point)
nnUNetv2_train 101 3d_fullres 0 --npz

# Train on all data (no cross-validation) 
nnUNetv2_train 101 3d_fullres all --npz

# Resume interrupted training
nnUNetv2_train 101 3d_fullres 0 --c --npz

# Only re-run validation + save .npz 
nnUNetv2_train 101 3d_fullres 0 --val --npz

# Custom number of epochs
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainer_250epochs --npz
"""


#!/bin/bash
set -e

# ====================== CONFIG ======================
DATASET_ID=501
CONFIGURATION="3d_fullres"
FOLDS=(0 1 2 3 4)
TRAINER="nnUNetTrainer"

PLANS="nnUNetPlans"
USE_NPZ=true # finding best hyperparameters
CONTINUE=true # continue training if interrupted
NUM_GPUS=1
DEVICE="cuda"

ARGS=""
if [ "$USE_NPZ" = true ]; then
    ARGS="$ARGS --npz"
fi
if [ "$CONTINUE" = true ]; then
    ARGS="$ARGS --c"
fi

echo "=============================================="
echo "Dataset:        $DATASET_ID"
echo "Configuration:  $CONFIGURATION"
echo "Trainer:        $TRAINER"
echo "Plans:          $PLANS"
echo "Folds:          ${FOLDS[*]}"
echo "=============================================="

for FOLD in "${FOLDS[@]}"; do
    echo ""
    echo ">>> Training fold $FOLD ..."
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD \
            -tr $TRAINER \
            -p $PLANS \
            -device $DEVICE \
            -num_gpus $NUM_GPUS \
            $ARGS
    else
        nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD \
            -tr $TRAINER \
            -p $PLANS \
            -device $DEVICE \
            $ARGS
    fi
done

echo ""
echo "All requested folds finished!"