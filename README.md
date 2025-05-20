# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering

## Project Introduction

This repository provides the official code for the paper [Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](https://arxiv.org/pdf/2505.12831). It implements contrastive prompting (CP) and inference-time activation intervention to enhance sentence embeddings in large language models (LLMs). The code supports various LLMs (e.g., LLaMA, Vicuna) and enables reproduction of experiments on multiple semantic evaluation tasks (STS/Transfer/Full).

---



## Directory Structure

```
CP/
├── evaluate_intervention.py      #  Main evaluation & intervention script
├── activation_intervention/         #  Hooks and utilities for intervention
├── SentEval/                     # SentEval  SentEval evaluation toolkit
├── senllm/                       #  Model-related code (e.g., LLaMA)
├── run_intervention_eval.sh      #  Batch evaluation shell script
├── README.md                     
```

---

## Requirements

**Install dependencies:**
```bash
conda create -n cp_39 python=3.9
conda activate cp_39

pip install -r requirements.txt
```

---

## Quick Start

### 1.  Download Data

``` sh
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
```
### 2. Evaluation

```bash
cd CP
python evaluate_intervention.py \
    --model_name_or_path llms/Llama-2-7b-hf \
    --mode test \
    --task_set stsb \
    --prompt_method prompteol \
    --output_layer 27 \
    --batch_size 16 \
    --use_which_plan intervention \
    --intervention_plan scaled \
    --intervention_location layer \
    --coeff 0.5 \
    --act_layer 4
```

**Main Arguments:**
- `--model_name_or_path`：Model path
- `--mode`：（dev/test/fasttest）| Evaluation mode
- `--task_set`： Task set（sts/transfer）
- `--prompt_method`： Prompt method（prompteol/cot/ke/ck）
- `--output_layer`： Output layer index
- `--use_which_plan`：Plan (origin/intervention)
- `--intervention_plan`：Intervention strategy（norm/scaled/scaled_norm/none/sub_head_norm/self_scaled）
- `--intervention_location`： Intervention location（att_head/mlp/layer）
- `--coeff`： Intervention coefficient
- `--act_layer`： Activation layer index

### 3. Batch Evaluation

```bash
bash run_intervention_eval.sh
```

---



##  Reference & Acknowledgement

- Our code is developed upon [PromptEOL](https://github.com/kongds/scaling_sentemb) and [ActAdd](https://github.com/UlisseMini/activation_additions_hf). We thank the authors of both projects for their great efforts.




