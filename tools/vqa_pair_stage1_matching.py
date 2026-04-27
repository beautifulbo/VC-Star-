import os
import torch
import torch.nn.functional as F

task="fill_in_blank"
split="test"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_path="../mlcd-vit-large-patch14-336_14"
img1=torch.load(os.path.join(img_path,"mlcd-vit-large-patch14-336_14_chunk_0.pth"),
                map_location='cpu', 
                mmap=True
                )
img2=torch.load(os.path.join(img_path,"mlcd-vit-large-patch14-336_14.pth"),
                map_location='cpu', 
                mmap=True
                )

def get_img(pid):
    pid = int(pid)
    if pid in img1:
        return img1[pid]
    return img2[pid]

text_path="../gte-modernbert-base_14"
text1=torch.load(os.path.join(text_path,"gte-modernbert-base_14_chunk_0.pth"),
                map_location='cpu', 
                mmap=True
                )
text2=torch.load(os.path.join(text_path,"gte-modernbert-base_14.pth"),
                map_location='cpu', 
                mmap=True
                )

def get_text(pid):
    pid = int(pid)
    if pid in text1:
        return text1[pid]
    return text2[pid]

import json
data=json.load(open("../data/iconqa_data/pid_splits.json"))["%s_%s" % (task, split)]
problems=json.load(open("../data/iconqa_data/problems.json"))

# 这里测试一下每个问题的文本和图像特征的形状
'''
pid_test=int(data[0])
print("Testing PID:", pid_test)
print("Question embedding shape:", get_text(pid_test).shape)  # torch.Size([7,768])
print("Image embedding shape:", get_img(pid_test).shape)  # torch.Size([577,1024])
'''



img_urls="../data/iconqa_data/iconqa/test/fill_in_blank"

paired={}

passed=0
failed=0
import tqdm
import logging

# 1. 创建 logger 对象
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 2. 创建写入文件的 Handler
# Windows 下务必加 encoding='utf-8'，否则遇到中文字符串可能报错
file_handler = logging.FileHandler(
    filename='vqa_stage1.txt',  # 你的txt文件路径
    mode='w',                      # 'a' 为追加模式，不会覆盖之前的内容；'w' 为覆盖模式
    encoding='utf-8'               
)
file_handler.setLevel(logging.INFO)

# 3. 创建控制台输出的 Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 4. 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 5. 将 Handlers 添加到 logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

for pid in tqdm.tqdm(data):
    problem=problems[pid]
    answer=problem['answer']
    ques=problem['question']
    img_url=os.path.join(img_urls, "%s" % pid,"image.png")
    ques_embedding=get_text(pid)[0:1,:].to(device)  # 取第一个token的embedding作为问题的整体表示，形状为 (1, 768)
    img_embedding=get_img(pid).to(device)
    text_theta=0.85
    img_theta=0.7
    for rel_pid in data:
        if rel_pid==pid:
            continue
        rel_ques_embedding=get_text(rel_pid)[0:1,:].to(device)
        rel_img_embedding=get_img(rel_pid).to(device)
        ques_sim=F.cosine_similarity(ques_embedding, rel_ques_embedding, dim=1).mean().item()
        img_sim=F.cosine_similarity(img_embedding, rel_img_embedding, dim=1).mean().item()
        if ques_sim>text_theta and img_sim>img_theta:
            logger.info("PID %s and PID %s are similar. Question similarity: %.4f, Image similarity: %.4f", pid, rel_pid, ques_sim, img_sim)
            rel_problem=problems[rel_pid]
            rel_answer=rel_problem['answer']
            rel_ques=rel_problem['question']
            rel_img_url=os.path.join(img_urls, "%s" % rel_pid,"image.png")
            paired[pid]={
                    "pid": pid,
                    "question": ques,
                    "answer": answer,
                    "img_url": img_url,
                    "pid_rel": rel_pid,
                    "question_rel": rel_ques,
                    "answer_rel": rel_answer,
                    "img_url_rel": rel_img_url,
                    "ques_sim": ques_sim,
                    "img_sim": img_sim
                }
            passed+=1
            break
    if pid not in paired:
        print("PID %s has no similar vqa pair found." % pid)
        failed+=1


paired_path="../data/iconqa_data/vqa_pairs_%s_%s.json" % (task, split)

logger.info("Total problems: %d, Paired: %d, Failed to pair: %d", len(data), passed, failed)

with open(paired_path, 'w') as f:
    json.dump(paired, f, indent=4)

logger.info("Paired problems saved to %s", paired_path)

