import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import click


@click.command()
@click.option("--text", type=str)
@click.option("--max_new_tokens", type=int, default=100)
@click.option("--model_id", type=str, default="./outputs_gemma2/checkpoint-700")
def infe(text, max_new_tokens, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        torch_dtype=torch.float16
    )
    # print(model)

    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    # print(tokenizer.decode(outputs[0], skip_special_tokens=False))


if __name__ == "__main__":
    infe()
