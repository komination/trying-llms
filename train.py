import random
import warnings
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# 設定
MODEL_NAME = "google/gemma-3-270m-it"
FINAL_MODEL_DIR = "./gemma3-kansaiben-lora"
SAVE_MERGED_MODEL = False

# まずは注意機構のみに LoRA を適用（安定＆軽量）
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "up_proj", "down_proj", "gate_proj",
    ],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

MAX_SEQ_LEN = 512
NUM_EPOCHS = 15
PER_DEVICE_BSZ = 4
GRAD_ACCUM = 4
LEARNING_RATE = 5e-5
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0
SEED = 42

# ログ関連
LOGGING_STEPS = 10
EVAL_RATIO = 0.05
EARLY_STOPPING_PATIENCE = 2


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def enable_tf32_if_available():
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def get_best_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32


def print_trainable_params(model: torch.nn.Module):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    ratio = 100 * trainable / total if total else 0
    print(f"訓練可能パラメータ: {trainable:,}")
    print(f"全パラメータ: {total:,}")
    print(f"訓練可能パラメータの割合: {ratio:.2f}%")


def assert_no_nan_in_trainable(model: torch.nn.Module, when: str = ""):
    for n, p in model.named_parameters():
        if p.requires_grad and (torch.isnan(p).any() or torch.isinf(p).any()):
            raise RuntimeError(f"[{when}] NaN/Inf 検出: {n}")


# データ前処理
def build_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    instr = (example.get("instruction") or example.get("input") or "").strip()
    out = (example.get("output") or "").strip()
    return {
        "messages": [
            {"role": "user", "content": instr},
            {"role": "assistant", "content": out},
        ]
    }


def tokenize_supervised(
    examples: Dict[str, List[Any]], tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """assistant 部分のみ損失をかける (-100 マスク) 版のトークナイズ"""
    input_ids_list, labels_list = [], []

    for msgs in examples["messages"]:
        # 完全文（user + assistant）
        full_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        # プロンプトのみ（最後の assistant を除外し、生成プロンプトを付与）
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )

        full = tokenizer(
            full_text,
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LEN,
        )
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LEN,
        )

        input_ids = full["input_ids"]
        labels = input_ids.copy()

        prompt_len = len(prompt["input_ids"])
        # 念のため境界チェック
        if prompt_len > len(labels):
            prompt_len = len(labels)
        labels[:prompt_len] = [-100] * prompt_len  # プロンプト部分の損失を無効化

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "labels": labels_list}


def data_collator(
    features: List[Dict[str, Any]], tokenizer: AutoTokenizer
) -> Dict[str, torch.Tensor]:
    input_ids = [f["input_ids"] for f in features]
    labels = [f["labels"] for f in features]
    max_len = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_token_id

    batch_input_ids, batch_attn, batch_labels = [], [], []
    for ids, lbl in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        batch_input_ids.append(ids + [pad_id] * pad_len)
        batch_attn.append([1] * len(ids) + [0] * pad_len)
        batch_labels.append(lbl + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


# モデルロード
def load_base_model_with_best_attn(model_name: str, dtype: torch.dtype):
    """
    FlashAttention2 -> SDPA -> eager の順で試す。
    使えた実装名を返しつつモデルをロード。
    """
    tried = []
    for impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                attn_implementation=impl,
            )
            print(f"attn_implementation: {impl}")
            return model, impl
        except Exception as e:
            tried.append((impl, str(e)[:200]))
    raise RuntimeError(f"all attn_implementation failed: {tried}")


def main():
    warnings.filterwarnings("default")

    print("Gemma3 Kansai-ben LoRA ファインチューニング開始")
    set_seed(SEED)
    enable_tf32_if_available()

    print("トークナイザをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = get_best_dtype()
    print(f"使用 dtype: {dtype}")

    print("ベースモデルをロード中...")
    base_model, attn_impl = load_base_model_with_best_attn(MODEL_NAME, dtype)

    # 学習時は use_cache をオフ（警告回避 & メモリ節約）
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    print("LoRA 設定を適用中...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(base_model, lora_config)
    print_trainable_params(model)

    print("データセットをロード/整形中...")
    raw_ds = load_dataset("shirochange/kansaiben", split="train")

    # 出力が空のサンプルは除去
    ds = raw_ds.filter(lambda e: (e.get("output") or "").strip() != "")

    # messages 列を作成
    ds = ds.map(build_messages)

    # トークナイズ（assistant のみ損失）
    tokenized = ds.map(
        lambda e: tokenize_supervised(e, tokenizer),
        batched=True,
        remove_columns=[c for c in ds.column_names if c != "messages"],
    )

    # 学習/評価分割
    split = tokenized.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_ds = split["train"].remove_columns(["messages"])
    eval_ds = split["test"].remove_columns(["messages"])

    print(f"学習サンプル数: {len(train_ds):,}, 評価サンプル数: {len(eval_ds):,}")

    use_bf16 = dtype == torch.bfloat16
    use_fp16 = dtype == torch.float16

    training_args = TrainingArguments(
        output_dir="./output-lora",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BSZ,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        weight_decay=0.0,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=LOGGING_STEPS,
        remove_unused_columns=False,
        report_to=None,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        fp16=use_fp16,
        bf16=use_bf16,
        tf32=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=lambda f: data_collator(f, tokenizer),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)
        ],
    )

    assert_no_nan_in_trainable(model, when="before_train")

    print("LoRA ファインチューニング開始...")
    trainer.train()

    assert_no_nan_in_trainable(model, when="after_train")

    model.save_pretrained(FINAL_MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    if SAVE_MERGED_MODEL:
        merged = model.merge_and_unload()
        merged.save_pretrained(FINAL_MODEL_DIR + "-merged", safe_serialization=True)

        tokenizer.save_pretrained(FINAL_MODEL_DIR + "-merged")

    print("完了")


if __name__ == "__main__":
    main()
