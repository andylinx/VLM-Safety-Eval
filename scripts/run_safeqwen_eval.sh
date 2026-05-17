#!/bin/bash
# Run SafeQwen2.5-VL-7B evaluations on GPU 0
set -e

cd /weka/scratch/cxiao13/nanxi_2/VLM-Safety-Eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "[$(date)] Starting SafeQwen2.5-VL-7B evaluations on GPU 0..."

# Run MM-SafetyBench
echo "[$(date)] Running MM-SafetyBench..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_eval.py \
    --model-path ./models/SafeQwen2.5-VL-7B \
    --model-name SafeQwen2.5-VL-7B \
    --dataset MM \
    --max-workers 2 \
    > logs/eval_SafeQwen_final.log 2>&1

echo "[$(date)] MM-SafetyBench completed!"

# Run MIS
echo "[$(date)] Running MIS..."
CUDA_VISIBLE_DEVICES=0 python scripts/run_hf_eval.py \
    --model-path ./models/SafeQwen2.5-VL-7B \
    --model-name SafeQwen2.5-VL-7B \
    --dataset MIS \
    --max-workers 2 \
    > logs/eval_SafeQwen_MIS_final.log 2>&1

echo "[$(date)] MIS completed!"
echo "[$(date)] All SafeQwen2.5-VL-7B evaluations completed!"
