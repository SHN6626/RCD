TS=$(date "+%Y%0m%0d_%T")
project_root_path="../../"
cli_path="${project_root_path}/src/benchmark_evaluation/truthfulqa_eval.py"
data_path="${project_root_path}/data/truthfulqa"
amateur_models=("ep0.01_sum_small")
for amateur_model in "${amateur_models[@]}"
do
    for lam in $(seq 1 0.2 2); do 
        ### Exps with Llama2-7B
        model_name="/root/autodl-tmp/llama2-7b-chat-hf"
        amateur_model_name="/root/autodl-tmp/$amateur_model"
        
        # ### For experiments using Baichuan2
        # model_name="/app/data/baichuan2"
        # amateur_model_name="app/data/baichuan2-f"
        
        # ### For experiments using Mistral
        # model_name="mistralai/Mistral-7B-Instruct-v0.1"
        # amateur_model_name="HillZhang/untruthful_mistral_7b"
        
        ### Baseline
        # output_path="${project_root_path}/exp_results/truthfulqa/${TS}/Greedy_llama2_7b_chat"
        # mkdir -p $output_path
        # cp $0 "$(dirname "$output_path")"
        
        generation_args="--relative_top 0.0"
        
        # echo "Greedy Decoding"
        # for i in 0; do
        #     echo "devices: ${i}"
        #     CMD="CUDA_VISIBLE_DEVICES=$i nohup python ${cli_path} \
        #         --model-name ${model_name} \
        #         --num-gpus 1 \
        #         --data-path ${data_path} \
        #         --output-path ${output_path}/result \
        #         --is-chat \
        #         --mode greedy \
        #         --parallel \
        #         --total-shard 8 \
        #         --shard-id $i \
        #         ${generation_args} \
        #         >${output_path}/shard_${i}.log 2>&1 &"
        #     echo $CMD
        #     eval $CMD
        # done
        # wait
        
        ### Our method
        output_path="${project_root_path}/exp_results/truthfulqa/${amateur_model}_${lam}"
        mkdir -p $output_path
        cp $0 "$(dirname "$output_path")"
        
        echo "ICD"
        for i in 1; do
            echo "devices: ${i}"
            CMD="CUDA_VISIBLE_DEVICES=0 nohup python ${cli_path} \
                --model-name ${model_name} \
                --amateur-model-name ${amateur_model_name} \
                --num-gpus 1 \
                --amateur-model-nums-gpus  1\
                --data-path ${data_path} \
                --output-path ${output_path} \
                --max_gpu_memory 80 \
                --is-chat \
                --mode lol_entropy_fusion \
                --mature_layer 32 \
                --candidate_premature_layers "24,26,28,30" \
                --total-shard 1 \
                --shard-id $i \
                --lam $lam \
                --savepath ${amateur_model} \
                ${generation_args} \
                >${output_path}/shard_${i}.log 2>&1 &"
            echo $CMD
            eval $CMD
        done
        wait
    done
done

