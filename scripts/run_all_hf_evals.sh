#!/bin/bash
# Run evaluations using HuggingFace directly (no vLLM needed)
# This script runs evaluations sequentially to avoid GPU memory issues
set -e

cd /weka/scratch/cxiao13/nanxi_2/VLM-Safety-Eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "[$(date)] Starting evaluations..."

# Run SafeQwen2.5-VL-7B on MM-SafetyBench
echo "[$(date)] Running SafeQwen2.5-VL-7B on MM-SafetyBench..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_eval.py \
    --model-path ./models/SafeQwen2.5-VL-7B \
    --model-name SafeQwen2.5-VL-7B \
    --dataset MM \
    --max-workers 2 \
    > logs/eval_SafeQwen2.5-VL-7B_MM.log 2>&1

echo "[$(date)] SafeQwen2.5-VL-7B MM-SafetyBench completed!"

# Run SafeLLaVA-7B on MM-SafetyBench
echo "[$(date)] Running SafeLLaVA-7B on MM-SafetyBench..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_eval.py \
    --model-path ./models/SafeLLaVA-7B \
    --model-name SafeLLaVA-7B \
    --dataset MM \
    --max-workers 2 \
    > logs/eval_SafeLLaVA-7B_MM.log 2>&1

echo "[$(date)] SafeLLaVA-7B MM-SafetyBench completed!"

# Run SafeQwen2.5-VL-7B on MIS
echo "[$(date)] Running SafeQwen2.5-VL-7B on MIS..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_eval.py \
    --model-path ./models/SafeQwen2.5-VL-7B \
    --model-name SafeQwen2.5-VL-7B \
    --dataset MIS \
    --max-workers 2 \
    > logs/eval_SafeQwen2.5-VL-7B_MIS.log 2>&1

echo "[$(date)] SafeQwen2.5-VL-7B MIS completed!"

# Run SafeLLaVA-7B on MIS
echo "[$(date)] Running SafeLLaVA-7B on MIS..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_eval.py \
    --model-path ./models/SafeLLaVA-7B \
    --model-name SafeLLaVA-7B \
    --dataset MIS \
    --max-workers 2 \
    > logs/eval_SafeLLaVA-7B_MIS.log 2>&1

echo "[$(date)] SafeLLaVA-7B MIS completed!"

echo "[$(date)] All evaluations completed!"
