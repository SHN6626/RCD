#!/bin/bash
project_root_path="../../"
cli_path="${project_root_path}/src/benchmark_evaluation/knowledge.py"
MODEL_PATH="/root/autodl-tmp/llama2-7b-chat-hf"  
AMATEUR_MODEL_PATH="/root/autodl-tmp/ep1_qa_small"
OUTPUT_DIR="./results"                  
DATASET="natural_questions"               
DEBUG=false                   
DEVICE="cuda"
MODE="contrastive-decoding"               

mkdir -p $OUTPUT_DIR

python ${cli_path} \
    --model-name $MODEL_PATH \
    --amateur-model-name $AMATEUR_MODEL_PATH \
    --device $DEVICE \
    --output-path "$OUTPUT_DIR/${DATASET}_result.json" \
    --dataset_name $DATASET \
    --mode $MODE \
    --debug $DEBUG \
    --num-gpus "1" \
    --max_gpu_memory 27 \
    --early_exit_layers "-1"
