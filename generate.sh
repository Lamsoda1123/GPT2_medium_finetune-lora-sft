#!/bin/bash

# # Python 脚本的路径
# # 设置环境变量
# export model_name="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/2023_12_25_01_04_09test_Lora_FullFT_lora_weightFalse_optim_adamw_torch_epoch5_lr0.0001/checkpoint-21120"
# export data_path='/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/outcome/prompt.json'
# export lora="" # 或者 "false"
# export bs="20"
# export top_k="4"
# export top_p="0.3"
# export num_beams="5"
# export temperature="0.3"

PYTHON_SCRIPT="./generate.py"

# # 循环直到 Python 脚本无错误执行
# while true; do
#     python $PYTHON_SCRIPT
#     EXIT_STATUS=$?

#     # 检查 Python 脚本是否成功执行
#     if [ $EXIT_STATUS -eq 0 ]; then
#         echo "Python 脚本成功执行。"
#         break
#     else
#         echo "Python 脚本执行失败，正在重试..."
#         # 可选：在重试之前暂停一会儿
#         sleep 1
#     fi
# done

# # Python 脚本的路径
# # 设置环境变量
# export model_name='/home/rnd/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
# export data_path='/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/outcome/prompt.json'
# export lora="/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/2023_12_24_16_15_14test_Lora_FullFT_lora_weightTrue_optim_('adamw_torch',)_epoch5_lr5e-06/checkpoint-5250" # 或者 "false"
# export bs="20"
# export top_k="4"
# export top_p="0.3"
# export num_beams="5"
# export temperature="0.3"

# # 循环直到 Python 脚本无错误执行
# while true; do
#     python $PYTHON_SCRIPT
#     EXIT_STATUS=$?

#     # 检查 Python 脚本是否成功执行
#     if [ $EXIT_STATUS -eq 0 ]; then
#         echo "Python 脚本成功执行。"
#         break
#     else
#         echo "Python 脚本执行失败，正在重试..."
#         # 可选：在重试之前暂停一会儿
#         sleep 1
#     fi
# done

# Python 脚本的路径
# 设置环境变量
export model_name='/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/2023_12_25_23_36_02_lora_weightFalse_loraR32_optim_sgd_epoch5_lr5e-05/checkpoint-21120'
export data_path='/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/outcome/prompt.json'
export lora="" # 或者 "false"
export bs="20"
export top_k="8"
export top_p="0.5"
export num_beams="6"
export temperature="0.8"

# 循环直到 Python 脚本无错误执行
while true; do
    python $PYTHON_SCRIPT
    EXIT_STATUS=$?

    # 检查 Python 脚本是否成功执行
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "Python 脚本成功执行。"
        break
    else
        echo "Python 脚本执行失败，正在重试..."
        # 可选：在重试之前暂停一会儿
        sleep 1
    fi
done
