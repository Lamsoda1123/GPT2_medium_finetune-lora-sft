# author: SidaLIN
# date  : 20231225
# python

This code provide Lora finetuning and full parameters finetuning on GPT2-medium.
    (and it's corelated batched inference)

To use these functions you should follow following steps:
1 download a GPT2-medium first (and your data)
2 change model path manually in utils.py, which provide a tokenizer
3 pip install -r requirements.txt
4 chmod +x train.sh
5 chmod +x generate.sh
6 check out train.sh and generate.sh, changing all you should change. 
7 run train.sh
8 run generate.sh

problem:
1 There's a version problem in my transformer or torch that result in OOM problem
  I didn't found any solution, unfortunately. However, I mangage to make it runable 
  by iterative run it until it exit without error. 
2 Even after sft, the model still have severe illusion. No matter the sft method is Lora or what. 
3 when using Lora, load_in_8bit seems not quite appropriate. Too much bugs keep me away from trying. 
4 This code may able to train different LLM since it is a simple one, but be causion with 
  the version of transfomers and torch, sometimes even cuda. LLM is fragile, as well as the code and packages.



