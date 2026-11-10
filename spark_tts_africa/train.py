#!/usr/bin/env python3
"""Native PyTorch warm-start training for Spark-TTS Africa.

Warm-started from pretrained Spark-TTS 0.5B (Qwen2.5 backbone + BiCodec heads)
with the 4K African Tokenizer (ipa_sp4000.json). Completion prompt masking
ensures the model learns strictly acoustic token prediction conditioned on text.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from spark_tts_africa.data import DataCollatorForTTSCompletion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str, default="/mnt/volume_d2wey28/models/Spark-TTS-0.5B/LLM")
    parser.add_argument("--tokenizer_dir", type=str, default="/mnt/volume_d2wey28/data/spark-tts-africa/tokenizer_4k")
    parser.add_argument("--train_data", type=str, required=True, help="Path to parquet train data")
    parser.add_argument("--val_data", type=str, default=None, help="Path to parquet validation data")
    parser.add_argument("--output_dir", type=str, default="checkpoints/spark_tts_africa")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def evaluate(model, val_loader, device, max_batches=30):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            if not torch.isnan(out.loss):
                total_loss += out.loss.item()
                count += 1
            if max_batches > 0 and count >= max_batches:
                break
    model.train()
    return total_loss / max(1, count)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer
    logger.info(f"Loading 4K African Tokenizer from {args.tokenizer_dir}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_dir)

    # 2. Load Base Model and perform Warm-Start Surgery
    logger.info(f"Loading base LLM from {args.model_dir}...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)

    old_embed = model.model.embed_tokens.weight.data.clone()
    target_vocab = len(tokenizer)
    logger.info(f"Africa Warm-Start: Resizing embeddings from {model.config.vocab_size} to {target_vocab}...")
    model.resize_token_embeddings(target_vocab)
    model.config.vocab_size = target_vocab
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = None

    # Transfer pretrained acoustic & special token embeddings
    transferred = 0
    for token_str, new_id in tokenizer.get_vocab().items():
        if new_id >= 4000:
            old_id = base_tokenizer.convert_tokens_to_ids(token_str)
            if old_id != base_tokenizer.unk_token_id and old_id < old_embed.size(0):
                model.model.embed_tokens.weight.data[new_id] = old_embed[old_id]
                transferred += 1
    logger.info(f"Warm-Start: Transferred {transferred} pretrained acoustic & special token embeddings!")

    # 3. Setup Datasets & Pre-Tokenization
    logger.info(f"Loading training data from {args.train_data}...")
    train_ds = Dataset.from_parquet(args.train_data)
    val_ds = Dataset.from_parquet(args.val_data) if args.val_data else None

    logger.info("Pre-tokenizing training dataset with 4K African Tokenizer...")
    tok_train = train_ds.map(
        lambda x: tokenizer(x["text"], max_length=1024, truncation=True),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Pre-tokenizing train",
    )
    tok_val = (
        val_ds.map(
            lambda x: tokenizer(x["text"], max_length=1024, truncation=True),
            batched=True,
            remove_columns=val_ds.column_names,
            desc="Pre-tokenizing val",
        )
        if val_ds
        else None
    )

    collator = DataCollatorForTTSCompletion(tokenizer=tokenizer)

    train_loader = DataLoader(
        tok_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        drop_last=True,
    )
    val_loader = (
        DataLoader(tok_val, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        if tok_val
        else None
    )

    # 4. Setup Optimizer & Cosine Scheduler with Warmup
    trainable_p = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_p):,}")

    opt = torch.optim.AdamW(trainable_p, lr=args.learning_rate, weight_decay=0.01)

    def lr_lambda(current_step: int):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0
    step = 0
    train_iter = iter(train_loader)
    running_loss = 0.0
    start_time = time.time()

    best_dir = os.path.join(args.output_dir, "checkpoint-best")
    os.makedirs(best_dir, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"STARTING SPARK-TTS AFRICA TRAINING ({args.max_steps} steps, lr={args.learning_rate:.2e})")
    logger.info("=" * 65)

    while step < args.max_steps:
        opt.zero_grad()
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        gnorm = torch.nn.utils.clip_grad_norm_(trainable_p, args.max_grad_norm)
        opt.step()
        scheduler.step()
        step += 1
        running_loss += accum_loss

        # Logging
        if step % args.log_interval == 0:
            avg_l = running_loss / args.log_interval
            elapsed = time.time() - start_time
            speed = args.log_interval / max(0.1, elapsed)
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"[Spark-TTS] Step {step}/{args.max_steps}: loss={avg_l:.4f}, "
                f"grad_norm={gnorm.item():.2f}, lr={current_lr:.2e}, it/s={speed:.2f}"
            )
            running_loss = 0.0
            start_time = time.time()

        # Evaluation & Early Stopping
        if val_loader and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device)
            logger.info(f"[Spark-TTS Eval] Step {step}: val_loss={val_loss:.4f} (best={best_val_loss:.4f})")
            if val_loss < best_val_loss - 0.001:
                logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving best checkpoint...")
                best_val_loss = val_loss
                patience_counter = 0
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "best_meta.json"), "w") as f:
                    json.dump({"step": step, "val_loss": val_loss}, f, indent=2)
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience:
                    logger.info("Early stopping triggered! Halting training.")
                    break

        # Periodic Saving
        if step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved periodic checkpoint to {ckpt_dir}")

    # Final Save
    final_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Spark-TTS Africa Training Complete! Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
