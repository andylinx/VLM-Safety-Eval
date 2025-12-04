model_name="Qwen3VL-4B"

nohup python -u src/evaluate.py \
    --model_name $model_name > eval.log 2>&1 &

