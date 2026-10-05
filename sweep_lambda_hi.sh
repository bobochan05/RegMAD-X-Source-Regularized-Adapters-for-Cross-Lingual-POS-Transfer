#!/bin/bash

# Define range of lambda values to test
LAMBDAS=(0.001 0.01 0.05 0.1 0.5)
SEED=42

for L in "${LAMBDAS[@]}"
do
    echo "------------------------------------------------"
    echo "🧪 Testing Lambda: $L for English -> Hindi"
    echo "------------------------------------------------"
    
    # Step 1: Train Regularized Hindi Adapter (MLM)
    # We use GPU 6 for training
    export CUDA_VISIBLE_DEVICES=7
    python run_mlm_reg.py \
        --model_name_or_path bert-base-multilingual-cased \
        --train_file data/hi_hdtb-ud-train.txt \
        --validation_file data/hi_hdtb-ud-dev.txt \
        --do_train \
        --do_eval \
        --num_train_epochs 5 \
        --learning_rate 1e-4 \
        --train_adapter \
        --adapter_config "pfeiffer+inv" \
        --load_adapter ./output/adapter_en/mlm \
        --reg_lambda $L \
        --seed $SEED \
        --output_dir ./output/sweep_hi_L${L} \
        --overwrite_output_dir \
        --dataloader_num_workers 0

    # Step 2: Run Zero-Shot POS Evaluation
    # We pass the newly trained adapter path to the eval script
    python eval_hi_sweep.py \
        --target_adapter_path ./output/sweep_hi_L${L}/mlm \
        --lambda_val $L \
        --seed $SEED
done