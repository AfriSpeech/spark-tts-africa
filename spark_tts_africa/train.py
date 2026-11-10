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
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from spark_tts_africa.data import DataCollatorForTTSCompletion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str, default="/mnt/volume_d2wey28/models/Spark-TTS-0.5B/LLM")
    parser.add_argument("--tokenizer_dir", type=str, default="/mnt/volume_d2wey28/data/spark-tts-africa/tokenizer_4k")
    parser.add_argument("--train_data", type=str, required=True, help="Path to tokenized train dataset directory or arrow/parquet")
    parser.add_argument("--val_data", type=str, default=None, help="Path to tokenized validation dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/spark_tts_africa")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--warmup_steps", type=int, default=200)
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
            tok = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if tok:
                tok.save_pretrained(best_dir)
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

    # 1. Load or Build 4K African Tokenizer
    if os.path.exists(os.path.join(args.tokenizer_dir, "tokenizer.json")):
        logger.info(f"Loading 4K African Tokenizer from {args.tokenizer_dir}...")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)
    else:
        logger.info("Building 4K African Tokenizer from ipa_sp4000.json and Spark-TTS tokens...")
        os.makedirs(args.tokenizer_dir, exist_ok=True)
        sp_file = "/mnt/volume_d2wey28/data/africa-tts/tok/ipa_sp4000.json"
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=sp_file)
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.eos_token = "<|im_end|>"
        special_tokens = [
            "<|task_tts|>", "<|start_content|>", "<|end_content|>",
            "<|start_global_token|>", "<|end_global_token|>",
            "<|start_semantic_token|>", "<|end_semantic_token|>",
            "<|im_end|>", "<|endoftext|>",
        ]
        global_tokens = [f"<|bicodec_global_{i}|>" for i in range(4096)]
        semantic_tokens = [f"<|bicodec_semantic_{i}|>" for i in range(8192)]
        all_new = special_tokens + global_tokens + semantic_tokens
        tokenizer.add_special_tokens({"additional_special_tokens": all_new})
        tokenizer.save_pretrained(args.tokenizer_dir)

    # 2. Load Base Model and perform Warm-Start Surgery
    logger.info(f"Loading base LLM from {args.model_dir}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
    )

    old_embed = model.model.embed_tokens.weight.data.clone()
    target_vocab = len(tokenizer)
    logger.info(f"Africa Warm-Start: Resizing embeddings from {model.config.vocab_size} to {target_vocab}...")
    model.resize_token_embeddings(target_vocab)
    model.config.vocab_size = target_vocab
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = None

    # Initialize 4K text embeddings with N(0, 0.04)
    torch.nn.init.normal_(model.model.embed_tokens.weight.data[:4000], mean=0.0, std=0.04)

    # Transfer pretrained codec and special token embeddings
    transferred = 0
    for token_str, new_id in tokenizer.get_vocab().items():
        if new_id >= 4000:
            old_id = base_tokenizer.convert_tokens_to_ids(token_str)
            if old_id != base_tokenizer.unk_token_id and old_id < old_embed.size(0):
                model.model.embed_tokens.weight.data[new_id] = old_embed[old_id]
                transferred += 1
    logger.info(f"Warm-Start: Transferred {transferred} pretrained acoustic & special token embeddings!")

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
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    logger.info("Tokenizing dataset sequences with 4K African Tokenizer...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names) if val_ds else None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    data_collator = DataCollatorForTTSCompletion(
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
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
