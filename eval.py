# -*- coding: utf-8 -*-
import argparse
import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

DEFAULT_PROMPTS = [
    "こんにちは！",
    "自己紹介をお願いします。",
    "ありがとうございます。",
]

def get_best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32

def free_gpu(obj):
    try:
        del obj
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def wrap_prompt(user_text: str) -> str:
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"

def run_generation(model, tokenizer, prompts, max_new_tokens=30, do_sample=False, temperature=0.7, top_p=0.9):
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer)
    cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=True,
    )
    results = []
    for i, u in enumerate(prompts, 1):
        prompt = wrap_prompt(u)
        t0 = time.perf_counter()
        out = gen(prompt, **cfg)[0]["generated_text"]
        dt = time.perf_counter() - t0
        resp = out[len(prompt):].split("<end_of_turn>")[0].strip()
        print(f"[{i}] 入力: {u}")
        print(f"    応答: {resp}")
        print(f"    生成時間: {dt:.2f}s")
        results.append({"input": u, "response": resp, "time_sec": dt})
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="google/gemma-3-270m-it")
    ap.add_argument("--adapter_dir", type=str, default="./gemma3-kansaiben-lora")
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--do_sample", action="store_true", help="指定するとサンプリング生成")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--bf16_off", action="store_true", help="bf16 を無効化して既定精度にする")
    ap.add_argument("--prompts", type=str, default=None,help="カンマ区切りで明示指定（例: 'A,B,C'）")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS if args.prompts is None else [s.strip() for s in args.prompts.split(",")]

    dtype = get_best_dtype()
    if args.bf16_off and dtype == torch.bfloat16:
        # 明示的にbf16を切りたい場合
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1 Base Model
    print("\n=== Base モデル推論 ===")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
    )
    if hasattr(base.config, "use_cache"):
        base.config.use_cache = True

    run_generation(
        base, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    free_gpu(base)

    # 2) LoRA 適用
    print("\n=== LoRA 適用モデル推論 ===")
    base_for_lora = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="eager",
    )
    if hasattr(base_for_lora.config, "use_cache"):
        base_for_lora.config.use_cache = True

    lora_model = PeftModel.from_pretrained(base_for_lora, args.adapter_dir)

    run_generation(
        lora_model, tokenizer, prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n完了！")

if __name__ == "__main__":
    main()
