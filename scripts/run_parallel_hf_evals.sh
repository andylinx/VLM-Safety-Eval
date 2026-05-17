#!/bin/bash
# Run evaluations in parallel on both GPUs
# GPU 0: SafeQwen2.5-VL-7B evaluations
# GPU 1: SafeLLaVA-7B evaluations
set -e

cd /weka/scratch/cxiao13/nanxi_2/VLM-Safety-Eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "[$(date)] Starting parallel evaluations on 2 GPUs..."

# Function to run all evaluations for a model
run_model_evals() {
    local GPU=$1
    local MODEL_PATH=$2
    local MODEL_NAME=$3
    local LOG_PREFIX=$4
    
    echo "[$(date)] Starting $MODEL_NAME evaluations on GPU $GPU..."
    
    # Run MM-SafetyBench
    echo "[$(date)] $MODEL_NAME: Running MM-SafetyBench..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/run_hf_eval.py \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --dataset MM \
        --max-workers 2 \
        > logs/${LOG_PREFIX}_MM.log 2>&1
    
    echo "[$(date)] $MODEL_NAME: MM-SafetyBench completed!"
    
    # Run MIS
    echo "[$(date)] $MODEL_NAME: Running MIS..."
    CUDA_VISIBLE_DEVICES=$GPU python scripts/run_hf_eval.py \
        --model-path "$MODEL_PATH" \
        --model-name "$MODEL_NAME" \
        --dataset MIS \
        --max-workers 2 \
        > logs/${LOG_PREFIX}_MIS.log 2>&1
    
    echo "[$(date)] $MODEL_NAME: MIS completed!"
    echo "[$(date)] $MODEL_NAME: All evaluations completed!"
}

# Run SafeQwen2.5-VL-7B on GPU 0 in background
run_model_evals 0 "./models/SafeQwen2.5-VL-7B" "SafeQwen2.5-VL-7B" "eval_SafeQwen2" &
PID1=$!
echo "[$(date)] SafeQwen2.5-VL-7B evaluations started (PID: $PID1)"

# Run SafeLLaVA-7B on GPU 1 in background
run_model_evals 1 "./models/SafeLLaVA-7B" "SafeLLaVA-7B" "eval_SafeLLaVA" &
PID2=$!
echo "[$(date)] SafeLLaVA-7B evaluations started (PID: $PID2)"

# Wait for both to complete
echo "[$(date)] Waiting for evaluations to complete..."
wait $PID1
EXIT1=$?
echo "[$(date)] SafeQwen2.5-VL-7B finished with exit code: $EXIT1"

wait $PID2
EXIT2=$?
echo "[$(date)] SafeLLaVA-7B finished with exit code: $EXIT2"

if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ]; then
    echo "[$(date)] All evaluations completed successfully!"
else
    echo "[$(date)] Some evaluations failed (Qwen: $EXIT1, LLaVA: $EXIT2)"
    exit 1
fi
