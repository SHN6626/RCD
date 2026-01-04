#!/bin/bash
# 设置文件路径和模型名称
datas=("news" "wiki" "expert")
for data in "${datas[@]}"
do
	amateur_models=("ep1_qa_small")
	for amateur_model in "${amateur_models[@]}"
	do
		DATA_FILE="/root/autodl-tmp/ICD-main-1/data/factor/${data}_factor.csv"
		MODEL_NAME="/root/autodl-tmp/llama2-7b-chat-hf"
		AMATEUR_MODEL_NAME="/root/autodl-tmp/$amateur_model"
		MODE="contrastive-decoding"
		LAMS=("0.1" "0.2" "0.3" "0.4" "0.5" )
		for LAM in "${LAMS[@]}"
		do
		# 运行 Python 脚本
			OUTPUT_FOLDER="/root/autodl-tmp/ICD-main-1/exp_results/factor/${amateur_model}_${data}_${LAM}_${MODE}.json"
			python /root/autodl-tmp/ICD-main-1/src/benchmark_evaluation/factor_eval.py \
			  --data-path $DATA_FILE \
			  --output-path $OUTPUT_FOLDER \
			  --model-name $MODEL_NAME \
			  --amateur-model-name $AMATEUR_MODEL_NAME \
			  --mode $MODE \
			  --savepath ${amateur_model}/${data} \
			  --lam $LAM 
		done
	done
done