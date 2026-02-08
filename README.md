# VLM-Safety-Eval

A framework for evaluating Vision-Language Models (VLMs) on **safety**, **hallucination**, and **benign capabilities**.

## Supported Benchmarks

| Category | Benchmark | Response Script | Judge Script |
|----------|-----------|-----------------|--------------|
| 🛡️ Safety | MM-SafetyBench | `get_response_MM.py` | `judge_MMSafety.py` |
| 🛡️ Safety | MIS | `get_response_MIS.py` | `judge_MIS.py` |
| 🔍 Hallucination | HallusionBench | `get_response_Hallucination.py` | `judge_Hallucination.py` |
| 🔍 Hallucination | MMVP | `get_response_MVP.py` | `judge_MVP.py` |
| ✅ Benign | MMMU-PRO | `get_response_MMMU_Pro.py` | `judge_MMMU_Pro.py` |
| ✅ Benign | MathVision | `get_response_MathVision.py` | `judge_mathvision.py` |

## Quick Start

### 1. Download Datasets

You can download the collection of datasets from Hugging Face:

```bash
# Download from andyc03/VLM-Eval
huggingface-cli download andyc03/VLM-Eval --local-dir ./data
cd ./data
# Prepare data by unzipping folders
bash unzip_folders.sh
```

### 2. Setup Environment

Create a `.env` file with the following configuration:

```properties
# Azure OpenAI Configuration (for judging with GPT-4o)
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=latest

# Data and Output Paths
DATA_BASE_ROOT_PATH=/path/to/your/Dataset
OUTPUT_DIR=./result
```

### 3. Start vLLM Server

```bash
vllm serve /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code
```

### 4. Model-Specific Inference Parameters

> ⚠️ **Important**: Different models have different preferences for inference hyperparameters. Reasoning models are more sensitive to these settings. Using incorrect parameters may significantly affect output quality.

| Model | temperature | top_p | top_k | repetition_penalty | presence_penalty |
|-------|-------------|-------|-------|-------------------|------------------|
| **Qwen3VL-32B-Thinking** | 1.0 | 0.95 | 20 | 1.0 | 0.0 |


Example for Qwen3VL-32B-Thinking:
```bash
export top_p=0.95
export top_k=20
export repetition_penalty=1.0
export presence_penalty=0.0
export temperature=1.0
```

### 5. Generate Responses

```bash
# Set PYTHONPATH to include src
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# MM-SafetyBench
python src/evaluation/get_response_MM.py --model_name "my-model" --vllm-url http://localhost:8000

# HallusionBench
python src/evaluation/get_response_Hallucination.py --model_name "my-model" --vllm-url http://localhost:8000

# MMVP
python src/evaluation/get_response_MVP.py --model_name "my-model" --vllm-url http://localhost:8000

# MIS
python src/evaluation/get_response_MIS.py --model_name "my-model" --vllm-url http://localhost:8000

# MMLU-PRO
python src/evaluation/get_response_MMMU_Pro.py --model_name "my-model" --vllm-url http://localhost:8000
```

Common arguments:
- `--model_name`: Model identifier for output files
- `--vllm-url`: vLLM server URL (default: `http://localhost:8000`)
- `--max-tokens`: Maximum tokens to generate
- `--skip-thinking`: Skip reasoning tokens for thinking models
- `--num-threads`: Parallel processing threads

### 6. Judge Responses

```bash
# MM-SafetyBench (uses Llama Guard 4)
python src/judging/judge_MMSafety.py --input ./result/MM_Safety/MLLM_Result_my-model --model my-model

# MIS (uses Azure OpenAI GPT-4o)
python src/judging/judge_MIS.py --input ./result/MIS/my-model_mis_hard_responses.json

# HallusionBench
python src/judging/judge_Hallucination.py --input ./result/Hallusion/my-model_HallusionBench_result.json

# MMVP
python src/judging/judge_MVP.py --input ./result/MMVP/my-model_MMVP_result.json
```

## Slurm Evaluation

For evaluating on a Slurm cluster, you can use the automated submission script which handles vLLM deployment and response generation.

```bash
sbatch scripts/vllm_slurm_eval.sh \
    --model-path /path/to/your/model \
    --dataset MM \
    --model-name my-model \
    --tp 1 \
    --num-threads 8
```

Supported datasets for `--dataset`: `MM`, `MIS`, `Hallucination`, `MVP`, `MMMU_Pro`, `MathVision`.

The script automates the following steps:
1. Allocates GPU resources via Slurm.
2. Finds an available port and starts the vLLM server.
3. Polls the server until it is ready for requests.
4. Executes the appropriate `get_response_<dataset>.py` script.
5. Gracefully shuts down the vLLM server upon completion.

## Output

Results are saved to `./result/<benchmark>/` with:
- Response JSON files: `{model_name}_{benchmark}_result.json`
- Judged files: `*_judged.json`
- Summary: `judging_summary.json` with ASR/accuracy metrics
