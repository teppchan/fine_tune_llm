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

model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.environ["HF_TOKEN"],
    use_fast=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=os.environ["HF_TOKEN"],
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
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        warmup_steps=2,
        max_steps=700,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        # optim="adamw_bnb_8bit",
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
    dataset_batch_size=1,
    num_of_sequences=512,
    packing=False,
)
trainer.train()
