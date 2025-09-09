import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# 重みは Thinking、コード/設定/トークナイザは PT 版
MODEL_WEIGHTS = "baidu/ERNIE-4.5-21B-A3B-Thinking"
CODE_AND_TOKENIZER = "baidu/ERNIE-4.5-21B-A3B-PT"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

config = AutoConfig.from_pretrained(
    CODE_AND_TOKENIZER,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    CODE_AND_TOKENIZER,
    trust_remote_code=True,
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_WEIGHTS,
    config=config,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

prompt = "Give me a short introduction to large language model."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], add_special_tokens=False, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

gen_ids = out[0][len(inputs.input_ids[0]):].tolist()
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
