# Korean Corpus Text Generation Tasks 2023
Repository for the text generation competition in [모두의 말뭉치](https://corpus.korean.go.kr/taskOrdtm/useTaskOrdtmList.do), 2023
There are two text-generation competitions
- [그림(사진) 기반 문장 생성 - image captioning](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=58&clCd=ING_TASK&subMenuId=sub01)
- [표 기반 문장 생성 - table summarization](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=41&clCd=ING_TASK&subMenuId=sub01)

## Purpose
This repository aims to share the training code for these competitions.
This code trains [llava](https://github.com/haotian-liu/LLaVA), which is the image-text generation model based on the [llama](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), using [lora](https://github.com/microsoft/LoRA).
As the training data from 모두의 말뭉치 has strict license constraints, the model is not shared.

## Results
Pretrained [KoLLaVA](https://huggingface.co/tabtoyou/KoLLaVA-KoVicuna-7b) has been used, which gives the following results.
- Image Captioning - 46.4804642 (ROGUE 1)
![image](https://github.com/LearnItAnyway/corpus_ko_generation_2023/assets/76693336/93ddfe98-ed8c-45ae-8393-5ae9f5ac35c9)
- Table Summarization - 37.5248569 (ROGUE 1)
![image](https://github.com/LearnItAnyway/corpus_ko_generation_2023/assets/76693336/02d1bcd3-d5a6-4de4-b299-9d35112eacdb)

## How to use
You can download the dataset from [모두의 말뭉치](https://corpus.korean.go.kr/taskOrdtm/useTaskOrdtmList.do)
Using `2023_nlg_format.ipynb`, you can make json file for the training and evaluation.
After making the json files, run `run_train_llava.sh` for the training of the model.

#
The scores are not good, but it is working. 
I hope this code helps the participants of these tasks
Feel free to ask any questions. 
e-mail : (learnitanyway@gmail.com)
