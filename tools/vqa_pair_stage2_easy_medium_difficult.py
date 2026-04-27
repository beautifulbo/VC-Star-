import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from prompt_template import THINGKING_PROMPT, CONTRASTING_PROMPT, RETHINKING_PROMPT
data_path="../data/iconqa_data/vqa_pairs_fill_in_blank_test.json"
Qwen_path="../../Qwen3-VL-8B-Instruct"
new_data_path="../data/iconqa_data/difficulty_1_vqa_pairs_fill_in_blank_test.json"

from PIL import Image
from openai import OpenAI
import os

data=json.load(open(data_path))

model = Qwen3VLForConditionalGeneration.from_pretrained(Qwen_path, torch_dtype=torch.float16).to("cuda:0")
model.eval()

processor = AutoProcessor.from_pretrained(Qwen_path)


easy_messages_list=[]
thinking_messages_list=[]
contrasting_messages_list=[]
rethinking_messages_list=[]


for item in list(data.keys()):
    path1=data[item]["img_url"].replace('\\','/')
    path2=data[item]["img_url_rel"].replace('\\','/')

    firstimage=Image.open(path1).convert("RGB")
    secondimage=Image.open(path2).convert("RGB")
    easy_messages_list.append(
        [
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                        "image":firstimage,
                    },
                    {
                        "type":"text",
                        "text":data[item]["question"],
                    }
                ],
            }
        ]
    )

    thinking_messages_list.append(
        [
            {
                "role":"system",
                "content":[
                    {
                        "type":"text",
                        "text":THINGKING_PROMPT,
                    }
                ],
            },
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                        "image":firstimage,
                    },
                    {
                        "type":"text",
                        "text":data[item]["question"],
                    },
                    {
                        "type":"text",
                        "text":data[item]["answer"],
                    }
                ],
            }
        ]
    )

    contrasting_messages_list.append(
        [
            {
                "role":"system",
                "content":[
                    {
                        "type":"text",
                        "text":CONTRASTING_PROMPT,
                    }
                ],
            },
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                        "image":firstimage,
                    },
                    {
                        "type":"image",
                        "image":secondimage,
                    },
                    {
                        "type":"text",
                        "text":data[item]["question"],
                    },
                    {
                        "type":"text",
                        "text":data[item]["question_rel"],
                    },
                    {
                        "type":"text",
                        "text":data[item]["answer"],
                    }
                ],
            }
        ]
    )


judger = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

import tqdm
    

easy_count=0
medium_or_difficult_count=0
length=len(easy_messages_list)
navi_item=list(data.keys())
item=0

for message in tqdm.tqdm(easy_messages_list):
    item+=1
    # Preparation for inference
    inputs = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]


    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    response = judger.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role":"system","content":"You are a helpful assistant to judege if FirstAnswer is equal to SecondAnswer. If they are equal, return 'Yes'. If they are not equal, return 'No'."},
            {
                "role":"user","content":
                [
                    {
                        "type":"text",
                        "text":output_text[0],
                    },
                    {
                        "type":"text",
                        "text":data[navi_item[item-1]]["answer"],
                    }
                ]
            }
        ]
    )

    if response.choices[0].message.content.strip().lower()=="yes":
        data[navi_item[item-1]]["difficulty"]="easy"
        easy_count+=1
    else:
        data[navi_item[item-1]]["difficulty"]="medium_or_difficult"
        medium_or_difficult_count+=1
    if item%200==0:
        with open(new_data_path,"w") as f:
            json.dump(data,f)
        print(f"easy count: {easy_count}/{length}, medium or difficult count: {medium_or_difficult_count}/{length}")
print(f"easy count: {easy_count}/{length}, medium or difficult count: {medium_or_difficult_count}/{length}")

print("Easy Stage finished!")
