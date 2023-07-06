import torch
import json

from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import CLIPImageProcessor
from llava import LlavaLlamaForCausalLM
from PIL import Image
from tqdm import trange

device='cuda'
llm = './output'

tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=False)

model = LlavaLlamaForCausalLM.from_pretrained(llm).half().to(device)

with open('test_5.json', 'r') as f:
    lines = json.load(f)

for i in trange(len(lines)):
    text = lines[i]['question']
    inputs = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)[:, -1024:]
    output = model.generate(inputs, max_new_tokens=512)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    lines[i]['output'] = text[0].split('<eval_token_5>')[-1]
    print(lines[i]['output'])
    lines[i].pop('question')


with open('result_5.json', 'a') as f:
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=False) + '\n') 