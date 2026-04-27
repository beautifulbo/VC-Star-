# IconQA `tools` 目录说明

这份文档说明 `./tools` 目录下各个脚本的用途、输入输出、运行方式和已知注意事项。

除两个 `.sh` 下载脚本外，下文默认你的当前工作目录是 `./tools`，因此所有目录都使用 Linux 风格的相对路径。

补充说明：

- `download_data_and_models.sh` 和 `download_img_feats.sh` 需要从 `../` 目录执行
- 其他 Python 脚本默认都按在 `./tools` 目录执行来说明

## 1. 目录内代码概览

`./tools` 中的代码可以分成 5 类：

- 数据与模型下载：`download_data_and_models.sh`、`download_img_feats.sh`
- 基础预处理：`create_dictionary.py`、`create_ans_label.py`
- 视觉特征生成：`generate_img_patch_feature.py`、`generate_img_choice_feature.py`、`image_models.py`
- 新增的文本/视觉 embedding 与 VQA 配对流水线：`build_text_embedding.py`、`build_vision_embedding.py`、`vqa_pair_stage1_matching.py`、`vqa_pair_stage2_easy_medium_difficult.py`、`vqa_pair_stage2_medium_difficult.py`、`prompt_template.py`
- 通用工具与评测：`utils.py`、`sub_acc.py`

说明：

- `vqa_stage1.txt`、`vqa_stage1_log.txt` 是运行日志，不是源码入口。
- `__pycache__/` 是 Python 缓存目录，不需要手动使用。

## 2. 运行前准备

### 2.1 Python 依赖

如果当前目录在 `./tools`，直接执行：

```bash
pip install -r ../requirements.txt
```

当前 `./tools` 下代码实际会用到的主要依赖包括：

- `torch`
- `torchvision`
- `transformers`
- `Pillow`
- `tqdm`
- `numpy`
- `h5py`
- `openai`

如果你要运行基于 API 的难度分级脚本，还需要：

```bash
export DEEPSEEK_API_KEY=your_key_here
```

### 2.2 目录约定

这些脚本默认依赖以下相对目录：

- `../data/iconqa_data`
- `../saved_models`
- `../../Qwen3-VL-8B-Instruct`
- `../../mlcd-vit-large-patch14-336`
- `../../gte-modernbert-base`

其中：

- `../data/iconqa_data` 是 IconQA 数据集目录
- `../saved_models` 是原始项目给出的预训练模型目录
- `../../Qwen3-VL-8B-Instruct` 是本地 Qwen3-VL 模型目录
- `../../mlcd-vit-large-patch14-336` 是本地图像 embedding 模型目录
- `../../gte-modernbert-base` 是本地文本 embedding 模型目录

## 3. 典型使用流程

### 3.1 原始 IconQA 预处理流程

```bash
cd ..
bash ./tools/download_data_and_models.sh
cd ./tools
python ./create_dictionary.py
python ./create_ans_label.py
python ./generate_img_patch_feature.py --icon_pretrained True --patch_split 79
python ./generate_img_choice_feature.py --icon_pretrained True
```

如果你只想直接下载官方提供的图像特征：

```bash
cd ..
bash ./tools/download_img_feats.sh
cd ./tools
```

### 3.2 原始 IconQA 结果统计

```bash
python ./sub_acc.py \
  --fill_in_blank_result exp_patch_transformer_ques_bert.json \
  --choose_txt_result exp_patch_transformer_ques_bert.json \
  --choose_img_result exp_patch_transformer_ques_bert.json
```

### 3.3 当前仓库新增的 VQA 配对与难度分级流程

```bash
python ./build_vision_embedding.py
python ./build_text_embedding.py
python ./vqa_pair_stage1_matching.py
python ./vqa_pair_stage2_easy_medium_difficult.py
python ./vqa_pair_stage2_medium_difficult.py
```

这条流水线的目标是：

1. 为 `fill_in_blank/test` 构建图像与文本 embedding
2. 为每个问题找到相似问题对
3. 先把问题粗分为 `easy` 和 `medium_or_difficult`
4. 再把非 `easy` 的样本细分为 `medium` 和 `difficult`

## 4. 各脚本说明

### 4.1 下载脚本

#### `download_data_and_models.sh`

用途：

- 下载 IconQA 原始数据到 `../data`
- 下载原始项目提供的模型到 `../`

用法：

```bash
cd ..
bash ./tools/download_data_and_models.sh
```

主要输出：

- `../data/iconqa_data`
- `../saved_models`

#### `download_img_feats.sh`

用途：

- 下载官方预提取图像特征到 `../data`

用法：

```bash
cd ..
bash ./tools/download_img_feats.sh
```

主要输出：

- `../data/embeddings` 解压后的各类特征文件

### 4.2 基础预处理脚本

#### `create_dictionary.py`

用途：

- 根据 `../data/iconqa_data/problems.json` 生成问题词典

输入：

- `../data/iconqa_data/problems.json`

输出：

- `../data/dictionary.pkl`

用法：

```bash
python ./create_dictionary.py
```

备注：

- 代码里固定 `add_choice=True`
- `choose_txt` 题型会把候选文本一起加入词典

#### `create_ans_label.py`

用途：

- 为 3 个子任务生成答案到标签、标签到答案的映射

输入：

- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/pid_splits.json`

输出：

- `../data/trainval_fill_in_blank_ans2label.pkl`
- `../data/trainval_fill_in_blank_label2ans.pkl`
- `../data/trainval_choose_txt_ans2label.pkl`
- `../data/trainval_choose_txt_label2ans.pkl`
- `../data/trainval_choose_img_ans2label.pkl`
- `../data/trainval_choose_img_label2ans.pkl`

用法：

```bash
python ./create_ans_label.py
```

### 4.3 原始视觉特征脚本

#### `image_models.py`

用途：

- 为其他脚本提供 ResNet101 图像编码器
- 支持 ImageNet 预训练和 Icon645 预训练两种模式

直接运行：

- 不是独立入口，供 `generate_img_patch_feature.py` 和 `generate_img_choice_feature.py` 调用

依赖路径：

- `../saved_models/icon_classification_ckpt/icon_resnet101_LDAM_DRW_lr0.01_0/ckpt.epoch66_best.pth.tar`

#### `generate_img_patch_feature.py`

用途：

- 为 `fill_in_blank`、`choose_txt`、`choose_img` 的题目主图提取 patch 级视觉特征

输入：

- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/pid_splits.json`
- `../data/iconqa_data/iconqa/.../image.png`

输出：

- `../data/iconqa_data/resnet101_pool5_<patch_split>[_icon]/iconqa_<split>_<task>_resnet101_pool5_<patch_split>[_icon].pth`

常用命令：

```bash
python ./generate_img_patch_feature.py --icon_pretrained True --patch_split 79
```

可用参数：

- `--input_path`，默认 `../data/iconqa_data`
- `--output_path`，默认 `../data/iconqa_data`
- `--arch`，默认 `resnet101`
- `--layer`，默认 `pool5`
- `--icon_pretrained`，默认 `False`
- `--patch_split`，可选 `14/25/30/36/79`
- `--gpu`，默认 `0`

注意：

- 虽然脚本提供了 `--split` 和 `--task` 参数，但主函数里会强制遍历全部 `task` 和 `split`
- 也就是说，默认执行一次会处理 `train/val/test` 的 3 个任务

#### `generate_img_choice_feature.py`

用途：

- 为 `choose_img` 子任务中每个样本的候选图片提取视觉特征

输入：

- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/pid_splits.json`
- `../data/iconqa_data/iconqa/.../<choice_image>`

输出：

- `../data/iconqa_data/resnet101_pool5[_icon]/iconqa_<split>_choose_img_resnet101_pool5[_icon].pth`

常用命令：

```bash
python ./generate_img_choice_feature.py --icon_pretrained True
```

可用参数：

- `--input_path`，默认 `../data/iconqa_data`
- `--output_path`，默认 `../data/iconqa_data`
- `--arch`，默认 `resnet101`
- `--layer`，默认 `pool5`
- `--icon_pretrained`，默认 `False`
- `--gpu`，默认 `0`

注意：

- 主函数里固定 `task="choose_img"`
- 主函数会强制遍历 `train/val/test`，不会只处理命令行传入的单一 split

### 4.4 通用工具与评测脚本

#### `utils.py`

用途：

- 提供 `create_dir`
- 提供 `Dictionary`
- 提供 `Logger`

直接运行：

- 不是独立入口，供多个脚本共享

注意：

- `Logger.log()` 里使用了 `self.infos.iteritems()`，这是 Python 2 风格接口
- 当前仓库里这个 `Logger` 没有被主要流程依赖，但如果你单独调用，可能需要改成 `items()`

#### `sub_acc.py`

用途：

- 汇总 `fill_in_blank`、`choose_txt`、`choose_img` 3 个任务的结果文件
- 输出总体准确率、按任务准确率和按技能准确率

输入：

- `../results/fill_in_blank/<result_json>`
- `../results/choose_txt/<result_json>`
- `../results/choose_img/<result_json>`
- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/pid2skills.json`

用法：

```bash
python ./sub_acc.py \
  --fill_in_blank_result exp_patch_transformer_ques_bert.json \
  --choose_txt_result exp_patch_transformer_ques_bert.json \
  --choose_img_result exp_patch_transformer_ques_bert.json
```

### 4.5 新增 embedding 与 VQA 配对脚本

#### `build_vision_embedding.py`

用途：

- 使用本地 `MLCDVisionModel` 为题目主图提取视觉 embedding

默认模型目录：

- `../../mlcd-vit-large-patch14-336`

默认输入：

- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/pid_splits.json`
- `../data/iconqa_data/iconqa/.../image.png`

按当前代码的默认行为：

- 只处理 `fill_in_blank`
- 只处理 `test`

默认命令：

```bash
python ./build_vision_embedding.py
```

可用参数：

- `--input_path`，默认 `../data/iconqa_data`
- `--output_path`，默认 `../data/iconqa_data`
- `--arch`，默认 `../../mlcd-vit-large-patch14-336`
- `--patch_split`，默认 `14`
- `--gpu`，默认 `0`

注意：

- 主函数里 `tasks=["fill_in_blank"]`、`splits=["test"]` 是硬编码
- 输出目录按路径归一化后等价于 `../mlcd-vit-large-patch14-336_14`
- 脚本会每处理 2000 条保存一次 chunk，并在结束时再保存一份最终文件

#### `build_text_embedding.py`

用途：

- 使用本地 `AutoModel` 与 `AutoTokenizer` 为题目文本提取 embedding

默认模型目录：

- `../../gte-modernbert-base`

默认输入：

- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/pid_splits.json`

按当前代码的默认行为：

- 只处理 `fill_in_blank`
- 只处理 `test`

默认命令：

```bash
python ./build_text_embedding.py
```

可用参数：

- `--input_path`，默认 `../data/iconqa_data`
- `--output_path`，默认 `../data/iconqa_data`
- `--arch`，默认 `../../gte-modernbert-base`
- `--patch_split`，默认 `14`
- `--gpu`，默认 `0`

注意：

- 主函数里 `tasks=["fill_in_blank"]`、`splits=["test"]` 是硬编码
- 输出目录按路径归一化后等价于 `../gte-modernbert-base_14`
- 脚本会每处理 2000 条保存一次 chunk，并在结束时再保存一份最终文件

#### `vqa_pair_stage1_matching.py`

用途：

- 根据图像与文本 embedding，为每个 `fill_in_blank/test` 样本寻找一个相似样本

固定配置：

- `task="fill_in_blank"`
- `split="test"`
- 文本相似度阈值 `text_theta=0.85`
- 图像相似度阈值 `img_theta=0.7`

依赖输入：

- `../mlcd-vit-large-patch14-336_14`
- `../gte-modernbert-base_14`
- `../data/iconqa_data/pid_splits.json`
- `../data/iconqa_data/problems.json`
- `../data/iconqa_data/iconqa/test/fill_in_blank`

输出：

- `../data/iconqa_data/vqa_pairs_fill_in_blank_test.json`
- `./vqa_stage1.txt`

默认命令：

```bash
python ./vqa_pair_stage1_matching.py
```

注意：

- 脚本默认读取的 embedding 文件名是 `mlcd-vit-large-patch14-336_14_chunk_0.pth`、`mlcd-vit-large-patch14-336_14.pth`、`gte-modernbert-base_14_chunk_0.pth`、`gte-modernbert-base_14.pth`
- 如果你直接使用 `build_vision_embedding.py` 和 `build_text_embedding.py` 的原始输出，先确认文件命名是否与这里一致

#### `prompt_template.py`

用途：

- 定义 VQA 难度分级流水线使用的 4 个 prompt 常量

包含内容：

- `THINGKING_PROMPT`
- `CONTRASTING_PROMPT`
- `RETHINKING_PROMPT`
- `FINAL_PROMPT`

直接运行：

- 不是独立入口，供两个难度分级脚本导入

#### `vqa_pair_stage2_easy_medium_difficult.py`

用途：

- 对 `vqa_pair_stage1_matching.py` 的配对结果做第一阶段难度划分
- 标签为 `easy` 或 `medium_or_difficult`

依赖输入：

- `../data/iconqa_data/vqa_pairs_fill_in_blank_test.json`
- `../../Qwen3-VL-8B-Instruct`
- `DEEPSEEK_API_KEY`

输出：

- `../data/iconqa_data/difficulty_1_vqa_pairs_fill_in_blank_test.json`

默认命令：

```bash
python ./vqa_pair_stage2_easy_medium_difficult.py
```

当前流程说明：

1. 用 Qwen3-VL 根据题图和问题生成回答
2. 用 DeepSeek API 判断生成回答是否和标准答案一致
3. 一致则标记为 `easy`
4. 否则标记为 `medium_or_difficult`

注意：

- 脚本每处理 200 条会保存一次中间结果
- 脚本结尾没有额外的最终 `json.dump`，因此最后不足 200 条的结果可能不会落盘
- 如果中途中断，`difficulty_1_*.json` 里可能出现一部分样本缺少 `difficulty`

#### `vqa_pair_stage2_medium_difficult.py`

用途：

- 读取第一阶段结果
- 只筛出非 `easy` 样本
- 再细分为 `medium` 或 `difficult`

依赖输入：

- `../data/iconqa_data/difficulty_1_vqa_pairs_fill_in_blank_test.json`
- `../../Qwen3-VL-8B-Instruct`
- `DEEPSEEK_API_KEY`

输出：

- `../data/iconqa_data/difficulty_2_vqa_pairs_fill_in_blank_test.json`

默认命令：

```bash
python ./vqa_pair_stage2_medium_difficult.py
```

当前流程说明：

1. 过滤掉第一阶段中的 `easy` 样本
2. 生成 coarse rationale
3. 做两张图的对比分析
4. 用 DeepSeek API 重写 rationale
5. 再次回答问题
6. 根据回答是否命中标准答案标记为 `medium` 或 `difficult`

注意：

- 这个脚本要求输入文件里的每条记录都已经有 `difficulty`
- 如果第一阶段输出里有缺失字段，这里会直接报 `KeyError: 'difficulty'`
- 脚本同样只在每 200 条时保存一次，没有循环结束后的最终保存
- 当前代码里还存在若干明显的索引和字符串拼接问题，正式大规模运行前建议先自测小样本

## 5. 建议的运行顺序

如果你要复现原始 IconQA 预处理流程，建议顺序如下：

```bash
cd ..
bash ./tools/download_data_and_models.sh
cd ./tools
python ./create_dictionary.py
python ./create_ans_label.py
python ./generate_img_patch_feature.py --icon_pretrained True --patch_split 79
python ./generate_img_choice_feature.py --icon_pretrained True
```

如果你要运行当前仓库新增的 VQA 配对与难度分级流程，建议顺序如下：

```bash
python ./build_vision_embedding.py
python ./build_text_embedding.py
python ./vqa_pair_stage1_matching.py
python ./vqa_pair_stage2_easy_medium_difficult.py
python ./vqa_pair_stage2_medium_difficult.py
```

## 6. 已知注意事项

- `generate_img_patch_feature.py` 和 `generate_img_choice_feature.py` 的主函数会覆盖一部分命令行参数，实际处理范围比命令行看起来更大。
- `build_vision_embedding.py`、`build_text_embedding.py` 当前主函数只处理 `fill_in_blank/test`。
- `vqa_pair_stage1_matching.py` 对 embedding 文件名有固定假设，和 `build_vision_embedding.py`、`build_text_embedding.py` 的默认输出命名不完全一致，运行前请先核对。
- 两个难度分级脚本都只做分段保存，没有循环结束后的最终保存；如果你依赖输出文件继续跑下游步骤，建议先确认最后一批结果已经写入。
- `vqa_pair_stage2_medium_difficult.py` 依赖第一阶段结果完整，否则会因为缺少 `difficulty` 字段而中断。
