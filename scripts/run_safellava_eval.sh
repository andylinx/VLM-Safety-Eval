#!/bin/bash
# Run SafeLLaVA evaluations on GPU 1
set -e

cd /weka/scratch/cxiao13/nanxi_2/VLM-Safety-Eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "[$(date)] Starting SafeLLaVA-7B evaluations on GPU 1..."

# Run MM-SafetyBench
echo "[$(date)] Running MM-SafetyBench..."
CUDA_VISIBLE_DEVICES=1 python scripts/run_hf_eval.py \
    --model-path ./models/SafeLLaVA-7B \
    --model-name SafeLLaVA-7B \
    --dataset MM \
    --max-workers 2 \
    > logs/eval_SafeLLaVA_MM_retry.log 2>&1

echo "[$(date)] MM-SafetyBench completed!"

# Run MIS
echo "[$(date)] Running MIS..."
CUDA_VISIBLE_DEVICES=1 python scripts/run_hf_eval.py \
    --model-path ./models/SafeLLaVA-7B \
    --model-name SafeLLaVA-7B \
    --dataset MIS \
    --max-workers 2 \
    > logs/eval_SafeLLaVA_MIS_retry.log 2>&1

echo "[$(date)] MIS completed!"
echo "[$(date)] All SafeLLaVA-7B evaluations completed!"
