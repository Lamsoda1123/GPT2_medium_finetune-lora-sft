import os
import transformers
import torch
import torch.nn as nn
# import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,  prepare_model_for_int8_training 
from utils import *
# utils中有基础模型位置 需要手动改

import os

def get_parameters():

    time = get_current_datetime_string()
    # 模型输出位置(checkpoint)
    output_dir = os.environ.get('output_dir',f'/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/')
    # 测试集位置 只取前500
    path_test = os.environ.get('path_test', '/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_test.json')
    # 训练集位置 只取前500
    path_train = os.environ.get('path_train', '/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_train.json') 
    # 模型位置
    model_name = os.environ.get('model_name','/home/rnd/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8')
    # 是否lora训练
    lora = bool(os.environ.get('lora', False))
    # lora rank
    lora_r = int(os.environ.get('lora_r', 32))
    num_epochs = int(os.environ.get('epoch', 5))
    # 优化器
    optim = os.environ.get('optim', "adamw_torch")
    learning_rate = float(os.environ.get('lr', 1e-5))
    # bs
    micro_batch_size = int(os.environ.get('micro_batch_size', 20))
    # 存多少个checkpoint
    save_total_limit = int(os.environ.get('save_total_limit', 10))
    # 训练全过程中eval多少次
    neval = int(os.environ.get('neval', 80))
    # wandb任务名
    wandb_run_name = f'{time}_lora_weight{str(bool(lora))}_loraR{lora_r}_optim_{optim}_epoch{num_epochs}_lr{learning_rate}'
    output_dir = output_dir + wandb_run_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return path_train, path_test, num_epochs, optim,learning_rate,lora,model_name,wandb_run_name,time,output_dir,micro_batch_size,save_total_limit,neval,lora_r

path_train, path_test, num_epochs, optim,learning_rate,lora,model_name,wandb_run_name,\
                                time,output_dir,micro_batch_size,save_total_limit,neval,lora_r = get_parameters()
# 限制显卡使用时设置
# os.environ["CUDA_VISIBLE_DEVICES"]="0" 
# 去除一个harmless warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 硬性规定 gradient_accumulation_steps
# gradient_accumulation_steps = 1

# 导入
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=False, 
    device_map='auto')
# 包装
if lora:
    config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
# 数据集
data,data_list = getdata(tokenizer,path_train)
data_test,data_test_list = getdata(tokenizer,path_test,500)#前500个

group_by_length = True
ddp = True
use_wandb = True

# 使save_steps是eval_steps的整数倍 同时也设定了steps数目
nstep = num_epochs*len(data)/micro_batch_size
eval_steps = int(nstep/neval)
save_steps = (int(nstep/save_total_limit)//eval_steps)*eval_steps

# 常规参数以及lr的cosine规划 不使用wandb在此处改
args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        # gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=0,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type='linear',
        fp16=True,
        logging_steps=20,
        optim=optim,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        auto_find_batch_size=True,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name = wandb_run_name if use_wandb else None,
    )


trainer = transformers.Trainer(
    model=model, 
    train_dataset=data,
    eval_dataset=data_test,
    args=args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
# 不保存最后模型 只存前面checkpoint而已