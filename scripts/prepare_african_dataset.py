#!/usr/bin/env python3
"""
Prepares the 897-hour African TTS dataset for Spark-TTS training.

Reads train_final.jsonl and valid_final.jsonl:
- Slices audio at [start, start+duration] using fast soundfile reading
- Extracts BiCodec global and semantic acoustic tokens
- Universalizes / phonetizes text with Africa-G2P
- Saves clean, sharded Parquet files for training
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
from tqdm.auto import tqdm

# Add Spark-TTS source
SPARK_REPO = os.environ.get("SPARK_REPO", "/mnt/volume_d2wey28/projects/spark-tts-africa/Spark-TTS")
if os.path.isdir(SPARK_REPO):
    sys.path.insert(0, SPARK_REPO)

from spark_tts_africa.data import extract_bicodec_tokens, format_spark_sequence, load_audio_slice, text_to_ipa
from sparktts.models.audio_tokenizer import BiCodecTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", type=str, default="/mnt/volume_d2wey28/data/africa-tts/train_final.jsonl")
    parser.add_argument("--output_parquet", type=str, default="/mnt/volume_d2wey28/data/spark-tts-africa/train.parquet")
    parser.add_argument("--model_dir", type=str, default="/mnt/volume_d2wey28/models/Spark-TTS-0.5B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    out_file = Path(args.output_parquet)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing BiCodecTokenizer on {args.device}...")
    tokenizer = BiCodecTokenizer(args.model_dir, args.device)

    rows = []
    shards_written = 0
    writer = None

    logger.info(f"Streaming from {args.input_jsonl}...")
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        pbar = tqdm(desc="Tokenizing African Speech for Spark-TTS")
        for idx, line in enumerate(f):
            if args.max_samples and idx >= args.max_samples:
                break
            try:
                d = json.loads(line)
                wav_path = d.get("path")
                start = float(d.get("start", 0.0))
                dur = float(d.get("duration", 0.0))
                if not wav_path or not os.path.exists(wav_path) or dur < 0.5 or dur > 20.0:
                    continue

                # Use precomputed IPA transcript if available, else convert
                text = d.get("transcript") or text_to_ipa(d.get("transcript_orig") or d.get("text", ""), d.get("iso", "eng"))
                if not text:
                    continue

                # Load 16kHz mono audio slice
                wav_np = load_audio_slice(wav_path, start, dur, target_sr=16000)

                # Tokenize into BiCodec tokens
                global_toks, semantic_toks = extract_bicodec_tokens(tokenizer, wav_np, sr=16000)

                # Format Spark training sequence
                seq = format_spark_sequence(text, global_toks, semantic_toks)
                rows.append({"text": seq})
                pbar.update(1)

                if len(rows) >= args.shard_size:
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(str(out_file), table.schema)
                    writer.write_table(table)
                    rows.clear()
                    shards_written += 1

            except Exception as e:
                logger.warning(f"Skipping line {idx}: {e}")

        if rows:
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(str(out_file), table.schema)
            writer.write_table(table)

        if writer:
            writer.close()
        pbar.close()

    logger.info(f"Data preparation complete! Saved to {out_file}")


if __name__ == "__main__":
    main()
