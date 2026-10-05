#!/bin/bash

# --- Function to clean up "zombie" processes ---
cleanup() {
    echo "🧹 Cleaning up GPU 5 and 6..."
    # 'fuser -k' kills processes using the specified GPU device
    fuser -k /dev/nvidia5
    fuser -k /dev/nvidia6
    sleep 2
}

# --- STAGE 1: Hindi & French (Baseline vs Reg) ---
#cleanup
#echo "🚀 [STAGE 1] Launching English -> Hindi/French Tasks..."

# Run Baseline on GPU 5
#export CUDA_VISIBLE_DEVICES=5
#export NCCL_P2P_DISABLE=1
#nohup python train_task_eval.py > logs_hi_fr_baseline.txt 2>&1 &

#Run Regularized on GPU 6
#export CUDA_VISIBLE_DEVICES=6
#export NCCL_P2P_DISABLE=1
#nohup python train_task_eval_reg.py > logs_hi_fr_reg.txt 2>&1 &

#echo "⏳ Waiting for Stage 1 to complete..."
#wait # Waits for background processes in this stage to finish

 #--- STAGE 2: Arabic & German (Baseline vs Reg) ---
#cleanup
echo "🚀 [STAGE 2] Launching Arabic/German Tasks..."

# Run Baseline on GPU 5
export CUDA_VISIBLE_DEVICES=5
export NCCL_P2P_DISABLE=1
nohup python eval_ar_de_baseline.py > logs_ar_de_baseline.txt 2>&1 &

# Run Regularized on GPU 6
export CUDA_VISIBLE_DEVICES=6
export NCCL_P2P_DISABLE=1
nohup python eval_ar_de_reg.py > logs_eval_ar_de_reg.txt 2>&1 &

echo "⏳ Waiting for Stage 2 to complete..."
wait

echo "✅ ALL TASKS COMPLETE. Check your logs for Accuracy and F1 results!"