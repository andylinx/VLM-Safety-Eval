#!/bin/bash
# Run MIS evaluation for SafeLLaVA-7B on GPU 1
set -e

cd /weka/scratch/cxiao13/nanxi_2/VLM-Safety-Eval
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

MODEL_PATH="./models/SafeLLaVA-7B"
MODEL_NAME="SafeLLaVA-7B"
DATASET="MIS"
PORT=8101

echo "[$(date)] Starting vLLM server for $MODEL_NAME on GPU 1 (port $PORT)..."

# Start vLLM server in background
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code \
    > logs/vllm_${MODEL_NAME}_MIS.log 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM server PID: $VLLM_PID"

# Wait for server to be ready
echo "[$(date)] Waiting for vLLM server to start..."
for i in $(seq 1 120); do
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then
        echo "[$(date)] Server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM server process died!"
        exit 1
    fi
    sleep 5
done

# Run evaluation
echo "[$(date)] Starting MIS evaluation for $MODEL_NAME..."
python src/evaluation/get_response_MIS.py \
    --model_name "$MODEL_NAME" \
    --vllm-url http://localhost:$PORT \
    --num-threads 4 \
    > logs/eval_${MODEL_NAME}_MIS.log 2>&1

EVAL_EXIT=$?

# Shutdown vLLM server
echo "[$(date)] Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

if [ $EVAL_EXIT -eq 0 ]; then
    echo "[$(date)] MIS evaluation for $MODEL_NAME completed successfully!"
else
    echo "[$(date)] MIS evaluation for $MODEL_NAME failed with exit code $EVAL_EXIT"
fi

exit $EVAL_EXIT
