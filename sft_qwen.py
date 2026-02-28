"""
SFT (Supervised Fine-Tuning): Qwen2.5-3B on Reasoning Dataset
==============================================================
Fine-tune Qwen2.5-3B base model on nohurry/Opus-4.6-Reasoning-3000x-filtered
before performing Knowledge Distillation.

Pipeline: SFT (this script) → Distillation (distill_qwen.py)

Usage:
    python sft_qwen.py --config distill_config.yaml
    python sft_qwen.py --num_train_epochs 3 --learning_rate 2e-5
    python sft_qwen.py --max_steps 100  # quick test

After SFT, run distillation with the SFT checkpoint:
    python distill_qwen.py --student_model ./sft_output --config distill_config.yaml
"""

import argparse
import logging
import os
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# API Keys & Login
# ──────────────────────────────────────────────────────────────
MY_WANDB_KEY = "API_KEY"
MY_HF_TOKEN = "API_KEY"

try:
    if MY_WANDB_KEY:
        import wandb
        wandb.login(key=MY_WANDB_KEY)
        print("W&B logged in successfully!")
    else:
        print("W&B Key is empty. Skipping W&B login.")
except Exception as e:
    print(f"Failed to login to W&B: {e}")

try:
    if MY_HF_TOKEN:
        from huggingface_hub import login
        login(token=MY_HF_TOKEN)
        print("Hugging Face logged in successfully!")
    else:
        print("HF Token is empty. Skipping HF login.")
except Exception as e:
    print(f"Failed to login to Hugging Face: {e}")


# ──────────────────────────────────────────────────────────────
# Dataset Formatting
# ──────────────────────────────────────────────────────────────
def format_reasoning_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Format Opus Reasoning dataset into chat-style training text.

    Dataset columns: id, problem, thinking, solution, difficulty, category, timestamp, hash

    Format:
        <|im_start|>user
        {problem}<|im_end|>
        <|im_start|>assistant
        <think>
        {thinking}
        </think>

        {solution}<|im_end|>
    """
    problem = example.get("problem", "").strip()
    thinking = example.get("thinking", "").strip()
    solution = example.get("solution", "").strip()

    # Build the response with thinking + solution
    response_parts = []
    if thinking:
        response_parts.append(f"<think>\n{thinking}\n</think>")
    if solution:
        response_parts.append(solution)

    response = "\n\n".join(response_parts) if response_parts else solution

    # Qwen2.5 ChatML format
    text = (
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )

    return {"text": text}


def format_as_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format as messages list for SFTTrainer's native chat template support.

    Returns: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    problem = example.get("problem", "").strip()
    thinking = example.get("thinking", "").strip()
    solution = example.get("solution", "").strip()

    # Build assistant response
    response_parts = []
    if thinking:
        response_parts.append(f"<think>\n{thinking}\n</think>")
    if solution:
        response_parts.append(solution)

    response = "\n\n".join(response_parts) if response_parts else solution

    messages = [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": response},
    ]

    return {"messages": messages}


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
def load_sft_config(config_path: str, cli_args) -> Dict[str, Any]:
    """Load SFT config from YAML + CLI overrides."""
    defaults = {
        "student_model": "Qwen/Qwen2.5-3B",
        "dataset_name": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
        "max_seq_length": 8192,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_steps": -1,
        "sft_output_dir": "./sft_output",
        "logging_steps": 10,
        "save_steps": 500,
        "save_total_limit": 3,
        "report_to": "wandb",
        "push_to_hub": True,
        "hub_model_id": "Phonsiri/Qwen2.5-3B-SFT-Reasoning",
    }

    # Load YAML
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        # Map relevant keys
        for key in defaults:
            if key in yaml_config:
                defaults[key] = yaml_config[key]
        # Also check sft-specific section
        sft_section = yaml_config.get("sft", {})
        if sft_section:
            defaults.update(sft_section)

    # CLI overrides
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if key == "config":
            continue
        if value is not None and key in defaults:
            defaults[key] = value

    return defaults


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="SFT: Qwen2.5-3B on Reasoning Dataset")
    parser.add_argument("--config", type=str, default="distill_config.yaml")
    parser.add_argument("--student_model", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--sft_output_dir", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true", default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_sft_config(args.config, args)

    logger.info("=" * 60)
    logger.info("SFT Configuration (Phase 1: before Distillation)")
    logger.info("=" * 60)
    logger.info(f"  Model   : {config['student_model']}")
    logger.info(f"  Dataset : {config['dataset_name']}")
    logger.info(f"  MaxLen  : {config['max_seq_length']}")
    logger.info(f"  Mode    : Full Fine-Tuning (all parameters)")
    logger.info(f"  LR      : {config['learning_rate']}")
    logger.info(f"  Epochs  : {config['num_train_epochs']}")
    logger.info(f"  Output  : {config['sft_output_dir']}")
    logger.info("=" * 60)

    # --- Load tokenizer ---
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["student_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # --- Load dataset ---
    logger.info(f"Loading dataset: {config['dataset_name']}")
    dataset = load_dataset(config["dataset_name"], split="train")
    logger.info(f"  Raw examples: {len(dataset)}")

    # Format dataset
    logger.info("Formatting dataset (problem → thinking → solution)...")
    original_columns = dataset.column_names
    dataset = dataset.map(format_reasoning_example, remove_columns=original_columns)
    logger.info(f"  Formatted examples: {len(dataset)}")

    # Show a sample
    logger.info(f"\n--- Sample ---\n{dataset[0]['text'][:500]}...\n--------------")

    # --- Load model ---
    logger.info(f"Loading model: {config['student_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        device_map="auto",
        torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # Enable gradient checkpointing
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable params: {trainable_params:,} / {total_params:,} (100% - Full Fine-Tuning)")

    # --- SFT Config ---
    sft_config = SFTConfig(
        output_dir=config["sft_output_dir"],
        num_train_epochs=config["num_train_epochs"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        report_to=config["report_to"],
        push_to_hub=config["push_to_hub"],
        hub_model_id=config["hub_model_id"] if config["hub_model_id"] else None,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        packing=True,  # Pack multiple short examples into one sequence for efficiency
        optim="adamw_torch",
        lr_scheduler_type="cosine",
    )

    # --- Create SFT Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # --- Train ---
    logger.info("🚀 Starting SFT training...")
    train_result = trainer.train()

    # --- Save ---
    logger.info(f"Saving model to {config['sft_output_dir']}")
    trainer.save_model(config["sft_output_dir"])
    tokenizer.save_pretrained(config["sft_output_dir"])

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if config["push_to_hub"] and config["hub_model_id"]:
        logger.info(f"Pushing to Hub: {config['hub_model_id']}")
        trainer.push_to_hub()

    logger.info("✅ SFT complete!")
    logger.info(f"   Model saved to: {config['sft_output_dir']}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("Next step: Run Distillation")
    logger.info("=" * 60)
    logger.info(f"  python distill_qwen.py \\")
    logger.info(f"      --student_model {config['sft_output_dir']} \\")
    logger.info(f"      --config distill_config.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
