# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering

## 项目简介 | Project Introduction

本项目为论文 [Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](https://arxiv.org/pdf/2505.12831) 的官方配套代码，旨在通过对比提示（Contrastive Prompting, CP）和推理时激活干预（Inference-Time Steering）提升大语言模型（LLM）的句子表征能力。代码支持多种主流 LLM（如 LLaMA、Vicuna 等），并可在多种语义评测任务（STS/Transfer）上复现论文实验。

This repository provides the official code for the paper [Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](https://arxiv.org/pdf/2505.12831). It implements contrastive prompting (CP) and inference-time activation intervention to enhance sentence embeddings in large language models (LLMs). The code supports various LLMs (e.g., LLaMA, Vicuna) and enables reproduction of experiments on multiple semantic evaluation tasks (STS/Transfer/Full).

---



## 目录结构 | Directory Structure

```
CP/
├── evaluate_intervention.py      # 主评测与干预脚本 Main evaluation & intervention script
├── activation_additions/         # 干预相关钩子与工具 Hooks and utilities for intervention
├── SentEval/                     # SentEval 评测工具包 SentEval evaluation toolkit
├── senllm/                       # LLaMA等模型相关代码 Model-related code (e.g., LLaMA)
├── run_intervention_eval.sh      # 批量评测脚本 Batch evaluation shell script
├── README.md                     # 本说明文件 This README
```

---

## 环境依赖 | Requirements

**安装依赖 | Install dependencies:**
```bash
conda create -n cp_39 python=3.9
conda activate cp_39

pip install -r requirements.txt
```

---

## 快速开始 | Quick Start

### 1. 下载数据 | Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
```
### 2. 评测 | Evaluation

```bash
cd CP
python evaluate_intervention.py \
    --model_name_or_path /root/wzh/llms/Llama-2-7b-hf \
    --mode test \
    --task_set stsb \
    --prompt_method prompteol \
    --output_layer 27 \
    --batch_size 16 \
    --use_which_plan intervention \
    --intervention_plan self_scaled \
    --intervention_location layer \
    --coeff 0.5 \
    --act_layer 4
```

**主要参数说明 | Main Arguments:**
- `--model_name_or_path`：模型路径 | Model path
- `--mode`：评测模式（dev/test/fasttest）| Evaluation mode
- `--task_set`：任务集（sts/transfer/full/stsb等）| Task set
- `--prompt_method`：提示方法（prompteol/cot/ke/ck）| Prompt method
- `--output_layer`：输出层索引 | Output layer index
- `--use_which_plan`：干预方案（origin/intervention）| Plan (origin/intervention)
- `--intervention_plan`：干预策略（norm/scaled/scaled_norm/none/sub_head_norm/self_scaled）| Intervention strategy
- `--intervention_location`：干预位置（att_head/mlp/layer）| Intervention location
- `--coeff`：干预系数 | Intervention coefficient
- `--act_layer`：干预层索引 | Activation layer index

### 3. 批量评测 | Batch Evaluation

```bash
bash run_intervention_eval.sh
```

---



## 参考与致谢 | Reference & Acknowledgement

- 本项目部分代码参考自 [PromptEOL](https://github.com/kongds/scaling_sentemb)。
Our code is developed upon [PromptEOL](https://github.com/kongds/scaling_sentemb). We thank the authors of PromptEOL for their great efforts.




