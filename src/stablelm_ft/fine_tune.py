import os

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer

lora_config = LoraConfig(
    r=8,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)

model_id = "stabilityai/japanese-stablelm-2-base-1_6b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    warmup_steps=2,
    max_steps=700,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    output_dir="outputs_stablelm",
    optim="paged_adamw_8bit",
    # optim="adamw_bnb_8bit",
    # deepspeed="src/gemma2_ft/ds_config_zero.json",
)


tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,
)

dataset = load_dataset("text", data_files="src/dayone_prepare/out.txt")


def formatting_func(example):
    t = example["text"][0].replace(
        "\\n",
        """
""",
    )
    text = f"{t}<eos>"
    # text = f"{example['text'][0]}<eos>"
    # print(text)
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=args,
    peft_config=lora_config,
    formatting_func=formatting_func,
    dataset_batch_size=1,
    num_of_sequences=512,
    packing=False,
)

model.config.use_cache = False

trainer.train()
