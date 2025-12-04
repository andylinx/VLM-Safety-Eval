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

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

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

### 2. Start vLLM Server

```bash
vllm serve /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --trust-remote-code
```

### 3. Generate Responses

```bash
# MM-SafetyBench
python get_response_MM.py --model_name "my-model" --vllm-url http://localhost:8000

# HallusionBench
python get_response_Hallucination.py --model_name "my-model" --vllm-url http://localhost:8000

# MMVP
python get_response_MVP.py --model_name "my-model" --vllm-url http://localhost:8000

# MIS
python get_response_MIS.py --model_name "my-model" --vllm-url http://localhost:8000

# MMLU-PRO
python get_response_MMMU_Pro.py --model_name "my-model" --vllm-url http://localhost:8000
```

Common arguments:
- `--model_name`: Model identifier for output files
- `--vllm-url`: vLLM server URL (default: `http://localhost:8000`)
- `--max-tokens`: Maximum tokens to generate
- `--skip-thinking`: Skip reasoning tokens for thinking models
- `--num-threads`: Parallel processing threads

### 4. Judge Responses

```bash
# MM-SafetyBench (uses Llama Guard 4)
python judge_MMSafety.py --input ./result/MM_Safety/MLLM_Result_my-model --model my-model

# MIS (uses Azure OpenAI GPT-4o)
python judge_MIS.py --input ./result/MIS/my-model_mis_hard_responses.json

# HallusionBench
python judge_Hallucination.py --input ./result/Hallusion/my-model_HallusionBench_result.json

# MMVP
python judge_MVP.py --input ./result/MMVP/my-model_MMVP_result.json
```

## Output

Results are saved to `./result/<benchmark>/` with:
- Response JSON files: `{model_name}_{benchmark}_result.json`
- Judged files: `*_judged.json`
- Summary: `judging_summary.json` with ASR/accuracy metrics

## License

See LICENSE file.
