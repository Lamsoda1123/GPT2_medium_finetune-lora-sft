import os
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer
from datasets import Dataset

def segment_text_pairs_with_label(text):
    segments = text.split("Assistant: ")
    assistant_count = len(segments) - 1
    pairs = [(''.join(segments[:i+1])+"Assistant: ", segments[i + 1].split('\n\nHuman:')[0]) for i in range(assistant_count)]
    return dict(zip(['instruction','output'],pairs[-1]))

def get_current_datetime_string():
    now = datetime.now()
    # Format the date and time as a string
    datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    return datetime_string

def getdata(tokenizer,path = "/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_test.json",k=-1):
    with open(path) as fp:
        data = json.load(fp)
    data_list = list(map(lambda x: {'ins':x.replace("\n\nHuman:","<|endoftext|><|endoftext|>\n\nHuman:",).lstrip('<|endoftext|><|endoftext|>')+ "<|endoftext|><|endoftext|>"},data))
    data = Dataset.from_list(data_list[:k])
    data = data.map(lambda x: tokenizer(x['ins'],
                            truncation=True,
                            max_length=300,
                            padding=True,
                            return_tensors='pt',), batched=True)
    return data,data_list

def predict(model,id,attm,generation_config):
    id = torch.tensor(id).to(model.device)
    attm = torch.tensor(attm).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=id,attention_mask = attm,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=128,
                )
    id = torch.zeros(1).to(model.device)
    attm = torch.zeros(1).to(model.device)
    del id,attm
    torch.cuda.empty_cache()
    return out

def getdata_generate(tokenizer,path = "/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/src/sft_train.json",k=-1):
    with open(path) as fp:
        data = json.load(fp)

    from datasets import Dataset
    data_list = list(map(lambda x: x.replace("\n\nHuman:","<|endoftext|>\n\nHuman:",).lstrip('<|endoftext|>')+ "<|endoftext|>",data))
    data_list = list(map(lambda x: {'ins':'\n\nAssistant:'.join(x.split('\n\nAssistant:')[:-1])} if '\n\nAssistant:' in x else {'ins':x},data))

    data = Dataset.from_list(data_list[k:])
    data = data.map(lambda x: tokenizer(x['ins'],
                            truncation=True,
                            max_length=128,
                            padding=True,
                            return_tensors='pt'), batched=True)
    return data,data_list

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
        save_path = f"/home/rnd/Job2Vec/LINSIDA/test_scripts/MLproj/MDS5210-23fall/outcome/FullFT_{data_path.split('/')[-1].rstrip('.json')}_{model_name.split('/')[-1]}.npy"
    return model_name, lora, save_path, bs, top_k, top_p, num_beams, temperature,data_path

model_name = '/home/rnd/.cache/huggingface/hub/models--gpt2/snapshots/e7da7f221d5bf496a48136c0cd264e630fe9fcc8'
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')
pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"