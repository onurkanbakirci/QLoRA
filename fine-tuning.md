---
title: "Fine-tuning LLaMA 2 with QLoRA"
subtitle: "A comprehensive guide to fine-tuning LLaMA 2 using Quantized LoRA"
description: "Learn how to efficiently fine-tune the LLaMA 2 language model using QLoRA (Quantized Low-Rank Adaptation) technique with detailed parameter explanations and step-by-step implementation."
date: "2024-03-21"
tags: ["LLaMA", "Fine-tuning", "QLoRA", "NLP", "Machine Learning"]
---


# Fine-tuning LLaMA 2 with QLoRA

## Introduction

This guide demonstrates how to fine-tune the LLaMA 2 language model using QLoRA (Quantized Low-Rank Adaptation), 
a memory-efficient technique that enables fine-tuning large language models on consumer GPUs.


## Required Libraries

First, let's install the necessary libraries:

```bash
# Install required libraries
!pip install -q -U transformers accelerate peft trl bitsandbytes datasets
```

### Library Descriptions:
- **peft**: Implements LoRA and QLoRA techniques
- **bitsandbytes**: Handles model quantization
- **accelerate**: Simplifies training across different hardware setups
- **transformers**: Provides pre-trained models and NLP tools
- **trl**: Integrates reinforcement learning with transformer models
- **datasets**: Offers ready-to-use datasets and processing tools

## Setup and Imports

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
```

## Configuration

### Model and Dataset Settings

```python
# The model that you want to train from the Hugging Face hub
model_name = "NousResearch/Llama-2-7b-chat-hf"

# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-finetunem"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 4

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}
```

## Model Training Setup

```python
# Load dataset (you can process it here)
dataset = load_dataset(dataset_name, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Training configuration
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    max_seq_length=512,          # Set your desired maximum sequence length
    packing=False,               # Set to True to enable packing of sequences
    dataset_text_field="text",   # Specify the dataset text field
)


# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    tokenizer=tokenizer,
    peft_config=peft_config,
)


# Train model
trainer.train()
```

## Save Trained Model

```python
# Save trained model
trainer.model.save_pretrained(new_model)
```

## Reload and Merge Model

```python
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

## Push Model to Hugging Face

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"

!huggingface-cli login

model.push_to_hub("onurkanbkrc/Llama-2-7b-chat-finetune", check_pr=True)

tokenizer.push_to_hub("onurkanbkrc/Llama-2-7b-chat-finetun",check_pr=True)
```

## Run Text Generation Pipeline

```python
# Run text generation pipeline with our next model
prompt = "How to own plane United States?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```
