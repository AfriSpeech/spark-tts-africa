#!/bin/bash
# Runner script for detached training of Spark-TTS Africa on NVIDIA H200
set -e

cd /mnt/volume_d2wey28/projects/spark-tts-africa
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/spark-tts-africa:${PYTHONPATH:-}"
export SPARK_REPO="/mnt/volume_d2wey28/projects/spark-tts-africa/Spark-TTS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Spark-TTS Africa Training in detached mode..."

python -m spark_tts_africa.train \
    --model_dir "/mnt/volume_d2wey28/models/Spark-TTS-0.5B/LLM" \
    --train_data "/mnt/volume_d2wey28/data/spark-tts-africa/train.parquet" \
    --val_data "/mnt/volume_d2wey28/data/spark-tts-africa/val.parquet" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/spark_tts_africa" \
    --learning_rate 5e-5 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_steps 30000 \
    --save_steps 500 \
    --eval_steps 250 \
    --early_stopping_patience 15 \
    2>&1 | tee -a train_spark_live.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed!"
