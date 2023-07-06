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
model.model.vision_tower[0].half().to(device)
model_vision_dict = model.get_model().initialize_vision_modules(
    vision_tower='openai/clip-vit-large-patch14',
    mm_vision_select_layer=-1,
    pretrain_mm_mlp_adapter=None
)
dtype = torch.float16
model.get_model().vision_tower[0].to(dtype=dtype, device=device)
vision_config = model_vision_dict['vision_config']

model.initialize_vision_tokenizer(mm_use_im_start_end=False,
                                  tokenizer=tokenizer, device=device,
                                  tune_mm_mlp_adapter=False,
                                  pretrain_mm_mlp_adapter=None)

image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')

with open('test_2.json', 'r') as f:
    lines = json.load(f)
image_folder = "2023_NLG/02"
image_tokens = "".join(["<im_patch>" for _ in range(256)]);

model.model.vision_tower[0].config.im_patch_token = tokenizer.encode("<im_patch>", add_special_tokens=False)[0]
model.model.vision_tower[0].config.use_im_start_end=False

eos_token_id = tokenizer.eos_token_id
import copy
def search(line, max_new_length=128, temperature=1.0,):
    image_path = f"{line['input'][:3]}/{line['input']}"
    image_file = f"{image_folder}/{image_path}.jpg"
    image = Image.open(image_file).convert('RGB')
    image = image_processor(image, return_tensors='pt')['pixel_values'][0].half().to(device)

    text = "<image><eval_token_2>"
    text = text.replace('<image>', image_tokens)
    inputs = tokenizer.encode(text, add_special_tokens=False, return_tensors='pt').to(device)

    pkv = model(input_ids=inputs[:, :-1], images=image.unsqueeze(0), use_cache=True, eval_type=[2])['past_key_values']
    counter = 0
    while True:
        input_tokens = inputs[:, -1:]
        outputs = model(input_ids=input_tokens, past_key_values=pkv, use_cache=True, eval_type=[2])
        pkv = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        if temperature < 1e-4:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)[0]

        # update generated ids, model inputs, and length for next step
        inputs = torch.cat([inputs, next_tokens[:, None]], dim=-1)
        counter += 1
        if next_tokens[0].item()==eos_token_id  or counter >= max_new_length:
            break
    return inputs

for i in trange(len(lines)):
    output = search(lines[i], temperature=0.0001)
    text = tokenizer.batch_decode(output[:, 257:], skip_special_tokens=True)
    lines[i]['output'] = text[0]
    lines[i].pop('question')


with open('result_2.json', 'a') as f:
    for line in lines:
        f.write(json.dumps(line, ensure_ascii=False) + '\n')  # This will append the 'banana' item
