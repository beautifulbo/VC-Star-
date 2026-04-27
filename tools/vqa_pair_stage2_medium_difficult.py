import torch
import json
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from prompt_template import THINGKING_PROMPT, CONTRASTING_PROMPT, RETHINKING_PROMPT,FINAL_PROMPT
data_path="../data/iconqa_data/difficulty_1_vqa_pairs_fill_in_blank_test.json"
Qwen_path="../../Qwen3-VL-8B-Instruct"
new_data_path="../data/iconqa_data/difficulty_2_vqa_pairs_fill_in_blank_test.json"
from PIL import Image
from openai import OpenAI
import os

oral_data=json.load(open(data_path))
data={}
for item in list(oral_data.keys()):
    if oral_data[item]["difficulty"]!="easy":
        data[item]=oral_data[item]

model = Qwen3VLForConditionalGeneration.from_pretrained(Qwen_path, torch_dtype=torch.float16).to("cuda:0")
model.eval()

processor = AutoProcessor.from_pretrained(Qwen_path)


thinking_messages_list=[]
contrasting_messages_list=[]
rethinking_messages_list=[]

for item in list(data.keys()):
    path1=data[item]["img_url"].replace('\\','/')
    path2=data[item]["img_url_rel"].replace('\\','/')

    firstimage=Image.open(path1).convert("RGB")
    secondimage=Image.open(path2).convert("RGB")

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
                        "text":data[item]["answer"].concat("||").concat(data[item]["answer_rel"]),
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
    

medium_count=0
difficult_count=0
length=len(data)

for item in tqdm.tqdm(list(data.keys())):
    message = thinking_messages_list[item]

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
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]


    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    coarse_rationale=output_text[0].strip()
    print("Thinking finished!")

    contrasting_message=contrasting_messages_list[item]

        # Preparation for inference
    inputs = processor.apply_chat_template(
        contrasting_message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]


    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    contrastive_analysis=output_text[0].strip()
    print("Contrasting analysis finished!")

    rethinking_message=[
        {
            "role":"system",
            "content":[
                {
                    "type":"text",
                    "text":RETHINKING_PROMPT,
                }
            ],
        },
        {
            "role":"user",
            "content":[
                {
                    "type":"text",
                    "text":data[item]["question"],
                },
                {
                    "type":"text",
                    "text":data[item]["answer"],
                },
                {
                    "type":"text",
                    "text":coarse_rationale,
                },
                {
                    "type":"text",
                    "text":contrastive_analysis,
                }
            ]
        }
    ]
    print("Rethinking finished!")

    response = judger.chat.completions.create(
        model = "deepseek-chat",
        messages = rethinking_message
    )
    fined_grained_rationale=response.choices[0].message.content.strip()
    data[item]["fined_grained_rationale"]=fined_grained_rationale

    path3=data[item]["img_url"].replace('\\','/')
    image=Image.open(path3).convert("RGB")
    final_message=[
        {
            "role":"system",
            "content":[
                {
                    "type":"text",
                    "text": FINAL_PROMPT,
                }
            ],
        },
        {
            "role":"user",
            "content":[
                {
                    "type":"image",
                    "image":image,
                },
                {
                    "type":"text",
                    "text":data[item]["question"],
                },
                {
                    "type":"text",
                    "text":fined_grained_rationale,
                }
            ]
        }
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        final_message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]


    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Final answering finished!")





    response = judger.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role":"system","content":"You are a helpful assistant to judege if FirstAnswer is equal to SecondAnswer. If they are equal, return 'Yes'. If they are not equal, return 'No'."},
            {
                "role":"user","content":
                [
                    {
                        "type":"text",
                        "text":output_text[0].strip(),
                    },
                    {
                        "type":"text",
                        "text":data[item]["answer"],
                    }
                ]
            }
        ]
    )

    if response.choices[0].message.content.strip().lower()=="yes":
        data[item]["difficulty"]="medium"
        medium_count+=1
    else:
        data[item]["difficulty"]="difficult"
        difficult_count+=1
    if item%200==0:
        with open(new_data_path,"w") as f:
            json.dump(data,f)
        print(f"medium count: {medium_count}/{length}, difficult count: {difficult_count}/{length}")
print(f"medium count: {medium_count}/{length}, difficult count: {difficult_count}/{length}")

print("Medium Stage finished!")
    