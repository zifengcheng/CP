export CUDA_VISIBLE_DEVICES=0,1,2,3

# # 定义模型路径数组
models=(
    "llms/Llama-2-7b-hf"
    # "llms/Meta-Llama-3-8B"
)

run_evaluation() {
    local model_path=$1
    local ol=($2)
    local l_values=($3)
    local c_values=($4)
    local task_set=$5
    # local muti_coeff_act_layer=$6
    for output_layer in "${ol[@]}"
    do
        for c in "${c_values[@]}"
        do
            for l in "${l_values[@]}"
            do
                python evaluate_intervention.py \
                            --model_name_or_path "$model_path" \
                            --mode test \
                            --task_set "$task_set" \
                            --prompt_method prompteol \
                            --output_layer $output_layer \
                            --batch_size 16 \
                            --use_which_plan intervention \
                            --intervention_plan scaled \
                            --intervention_location att_head \
                            --coeff $c \
                            --act_layer $l
            done
        done
    done
}

for model_path in "${models[@]}"
do    
    if [ "$model_path" == "/webdav/MyData/llms/Meta-Llama-3-8B" ]; then
        run_evaluation "$model_path" "-2" "6" "0.25" "stsb"
    elif [ "$model_path" == "/root/wzh/llms/Llama-2-7b-hf" ]; then
        run_evaluation "$model_path" "27" "4" "0.5" "stsb"
    else
        echo "Unknown model path: $model_path"
        continue
    fi
done

