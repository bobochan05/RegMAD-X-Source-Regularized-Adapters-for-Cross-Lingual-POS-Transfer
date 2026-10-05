#!/bin/bash

# --- 1. CLEAN UP ZOMBIE PROCESSES ---
# This ensures GPUs are fresh before starting Stage 2
echo "🧹 Cleaning up existing processes on GPU 5 and 6..."
fuser -k /dev/nvidia5
fuser -k /dev/nvidia6
sleep 2

# --- 2. LAUNCH ARABIC/GERMAN EXPERIMENTS ---
echo "🚀 Launching Arabic/German Tasks..."

# Run Baseline on GPU 5
# Unique log: logs_eval_ar_de_baseline.txt
export CUDA_VISIBLE_DEVICES=5
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
nohup python eval_ar_de_baseline.py > logs_eval_ar_de_baseline.txt 2>&1 &

# Run Regularized on GPU 6
# Unique log: logs_eval_ar_de_reg.txt
export CUDA_VISIBLE_DEVICES=6
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
nohup python eval_ar_de_reg.py > logs_eval_ar_de_reg.txt 2>&1 &

echo "⏳ Waiting for Arabic and German evaluations to complete..."
wait

echo "✅ ALL TASKS COMPLETE. Results saved in logs_eval_ar_de_baseline.txt and logs_eval_ar_de_reg.txt"