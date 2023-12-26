import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,  prepare_model_for_int8_training 
import transformers
from datasets import load_dataset
import json
from peft import PeftModel
from datasets import load_dataset,Dataset
from datasets import load_dataset,Dataset
from transformers.generation.utils import GenerationConfig
from utils import *
from functools import reduce
import numpy as np
from tqdm import tqdm
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os

def get_parameters_generate():
    model_name = os.environ.get('model_name', '/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/2023_12_25_01_04_09test_Lora_FullFT_lora_weightFalse_optim_adamw_torch_epoch5_lr0.0001/checkpoint-21120')
    data_path = os.environ.get('data_path', '/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_test.json')
    lora = os.environ.get('lora', '')
    bs = int(os.environ.get('bs', 20))
    top_k = int(os.environ.get('top_k', 4))
    top_p = float(os.environ.get('top_p', 0.3))
    num_beams = int(os.environ.get('num_beams', 5))
    temperature = float(os.environ.get('temperature', 0.3))
    if lora:
        save_path = f"/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/outcome/Lora_{data_path.split('/')[-1].rstrip('.json')}_{bool(lora)}_{model_name.split('/')[-1]}.npy"
    else:
        save_path = f"/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/outcome/sgd_FullFT_{data_path.split('/')[-1].rstrip('.json')}_{model_name.split('/')[-1]}.npy"
    return model_name, lora, save_path, bs, top_k, top_p, num_beams, temperature,data_path

model_name, lora, save_path, bs, top_k, top_p, num_beams, temperature,data_path = get_parameters_generate()

device_map = "auto"
generation_config = GenerationConfig(temperature=0.1,
                                        top_p=0.1,
                                        top_k=1,
                                        num_beams=1,
                                        eos_token_id=tokenizer.eos_token_id,
                                        do_sample=True,
                                        pad_token_id=tokenizer.pad_token_id)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
if lora:
    model = PeftModel.from_pretrained(model,lora,torch_dtype=torch.float32)
model = model.eval()

if os.path.exists(save_path):
    already = np.load(save_path)
else:
    already = np.array([])

data_test,data_list = getdata_generate(tokenizer,data_path,len(already))

if data_test.num_rows != 0:
    num = int(len(data_test['input_ids'])/bs)
    out = []
    for i in tqdm(range(num)):
        if i == num-1:
            id = data_test['input_ids'][bs*i:]
            attm = data_test['attention_mask'][bs*i:]
        id = data_test['input_ids'][bs*i:bs*(i+1)]
        attm = data_test['attention_mask'][bs*i:bs*(i+1)]
        out.append(predict(model,id,attm,generation_config))

        out_ = list(reduce(lambda x,y:x+y,map(lambda o:tokenizer.batch_decode(o.sequences,skip_special_tokens=True),out)))
        out_ = np.hstack([already,np.array(out_)])
        np.save(save_path ,out_)


        