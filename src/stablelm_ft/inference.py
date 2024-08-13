import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import click


@click.command()
@click.option("--text", type=str)
@click.option("--max_new_tokens", type=int, default=100)
def infe(text, max_new_tokens):
    model_id = "stabilityai/japanese-stablelm-2-base-1_6b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    # print(model)

    device = "cuda:0"
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.99,
        top_p=0.95,
        do_sample=True,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    infe()
