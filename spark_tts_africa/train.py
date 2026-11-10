#!/usr/bin/env python3
"""
Training script for Spark-TTS Africa.

Adapts Spark-TTS-0.5B LLM (Qwen2.5-0.5B backbone) to African speech
using the Africa-TTS warm-start / fine-tuning approach:
  - Input text: African phonetic IPA (africa-g2p)
  - Target: BiCodec global + semantic discrete acoustic tokens
  - Optimization: AdamW on native BF16 with early stopping on validation loss
  - Storage: Lean rolling checkpointing (keeps only last 2 + checkpoint-best)
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str, default="/mnt/volume_d2wey28/models/Spark-TTS-0.5B/LLM")
    parser.add_argument("--train_data", type=str, required=True, help="Path to tokenized train dataset directory or arrow/parquet")
    parser.add_argument("--val_data", type=str, default=None, help="Path to tokenized validation dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/spark_tts_africa")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 15, output_dir: str = ""):
        self.patience = patience
        self.output_dir = output_dir
        self.best_loss = float("inf")
        self.counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return control

        best_dir = os.path.join(self.output_dir, "checkpoint-best")
        if eval_loss < self.best_loss - 0.001:
            logger.info(f"Validation loss improved from {self.best_loss:.4f} to {eval_loss:.4f}. Saving best checkpoint...")
            self.best_loss = eval_loss
            self.counter = 0

            # Save best checkpoint
            os.makedirs(best_dir, exist_ok=True)
            kwargs["model"].save_pretrained(best_dir)
            kwargs["tokenizer"].save_pretrained(best_dir)
            with open(os.path.join(best_dir, "best_meta.json"), "w") as f:
                json.dump({"step": state.global_step, "eval_loss": eval_loss}, f, indent=2)
        else:
            self.counter += 1
            logger.info(f"Validation loss did not improve (best: {self.best_loss:.4f}, current: {eval_loss:.4f}). Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping triggered! Halting training.")
                control.should_training_stop = True

        return control


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading base LLM from {args.model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    logger.info(f"Loading training data from {args.train_data}...")
    if os.path.isdir(args.train_data):
        train_ds = load_from_disk(args.train_data)
    else:
        train_ds = Dataset.from_parquet(args.train_data)

    val_ds = None
    if args.val_data:
        logger.info(f"Loading validation data from {args.val_data}...")
        if os.path.isdir(args.val_data):
            val_ds = load_from_disk(args.val_data)
        else:
            val_ds = Dataset.from_parquet(args.val_data)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)

    logger.info("Tokenizing dataset sequences...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names) if val_ds else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=20,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.eval_steps if val_ds else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        tf32=True,
        dataloader_num_workers=4,
        report_to="none",
    )

    callbacks = [EarlyStoppingCallback(patience=args.early_stopping_patience, output_dir=args.output_dir)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=callbacks,
    )

    logger.info("Starting training loop...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
