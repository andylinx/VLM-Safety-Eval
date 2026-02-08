#!/bin/bash
#SBATCH --job-name=vlm_eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -e

# Default values
MODEL_PATH=""
DATASET="MM"
MODEL_NAME="model"
TP=1
MAX_MODEL_LEN=4096
NUM_THREADS=4

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --num-threads)
      NUM_THREADS="$2"
      shift 2
      ;;
    *)
      # Pass other arguments to run_eval.py
      OTHER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model-path is required"
  exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting evaluation for dataset: $DATASET"
echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"

# Run the wrapper script
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python scripts/run_eval.py \
    --model-path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --model-name "$MODEL_NAME" \
    --tp "$TP" \
    --max-model-len "$MAX_MODEL_LEN" \
    --num-threads "$NUM_THREADS" \
    "${OTHER_ARGS[@]}"

echo "Evaluation job completed."
