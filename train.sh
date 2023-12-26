#!/bin/bash

# 设置环境变量
export output_dir="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/"
export path_test="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_test.json"
export path_train="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_train.json"
export model_name="/home/rnd/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
# export lora='true' # 或者 "true"
export lora_r="32"
export epoch="10"
export optim="adamw_torch"
export lr="0.00005"
export micro_batch_size="80"
export save_total_limit="10"
export neval="80"

# 运行 Python 脚本
# python /home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/main_pj/ft_copy.py
# echo "Lora Python 脚本成功执行"


# 设置环境变量
export output_dir="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/"
export path_test="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_test.json"
export path_train="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_train.json"
export model_name="/home/rnd/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8"
# export lora=False # 或者 "true"
export lora_r="32"
export epoch="5"
export optim="sgd"
export lr="0.00005"
export micro_batch_size="20"
export save_total_limit="10"
export neval="80"

# 运行 Python 脚本
python /home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/main_pj/ft_copy.py
echo "sft Python 脚本成功执行"
