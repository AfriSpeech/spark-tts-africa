#!/usr/bin/env python3
"""
Inference CLI and API for Spark-TTS Africa.

Supports African speech synthesis with automatic Africa-G2P phonetic conversion
and zero-shot voice cloning.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

from spark_tts_africa.data import text_to_ipa

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SparkTTSAfrica:
    def __init__(
        self,
        model_path: str = "AfriSpeech/spark-tts-africa",
        spark_repo_path: Optional[str] = "/mnt/volume_d2wey28/projects/spark-tts-africa/Spark-TTS",
        bicodec_model_dir: str = "/mnt/volume_d2wey28/models/Spark-TTS-0.5B",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        if spark_repo_path and os.path.isdir(spark_repo_path):
            sys.path.insert(0, spark_repo_path)

        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        logger.info(f"Loading Spark-TTS Africa LLM from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device,
        )
        self.model.eval()

        logger.info(f"Loading BiCodecTokenizer from {bicodec_model_dir}...")
        self.audio_tokenizer = BiCodecTokenizer(bicodec_model_dir, device)
        self.sampling_rate = self.audio_tokenizer.config.get("sample_rate", 16000)

    def generate(
        self,
        text: str,
        language: Optional[str] = None,
        ref_audio: Optional[str] = None,
        use_ipa: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
    ) -> np.ndarray:
        """Synthesize speech audio from text.

        Args:
            text: Text to synthesize.
            language: Language code (e.g. 'twi', 'hau', 'yor', 'swa', 'eng').
            ref_audio: Optional path to reference audio clip for voice cloning.
            use_ipa: Whether to convert text to IPA using africa-g2p.
            temperature: Sampling temperature.
            top_k: Top-k sampling filter.
            top_p: Top-p nucleus sampling filter.
            max_new_tokens: Maximum acoustic tokens to generate.

        Returns:
            Numpy float32 waveform array at 16,000 Hz.
        """
        # Convert text to IPA if enabled
        if use_ipa and language:
            prompt_text = text_to_ipa(text, language)
        else:
            prompt_text = text

        # Voice cloning prefix if ref_audio provided
        global_prefix = ""
        if ref_audio and os.path.exists(ref_audio):
            wav_data, sr = sf.read(ref_audio, dtype="float32", always_2d=True)
            wav_data = wav_data.mean(1)
            if sr != 16000:
                wav_data = torchaudio.functional.resample(
                    torch.from_numpy(wav_data)[None], orig_freq=sr, new_freq=16000
                )[0].numpy()
            ref_clip = self.audio_tokenizer.get_ref_clip(wav_data)
            ref_tensor = torch.from_numpy(ref_clip).unsqueeze(0).float().to(self.device)
            wav_tensor = torch.from_numpy(wav_data).unsqueeze(0).float().to(self.device)
            processed = self.audio_tokenizer.processor(wav_data, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = processed.input_values.to(self.audio_tokenizer.feature_extractor.device)
            out = self.audio_tokenizer.feature_extractor(iv)
            feat = (out.hidden_states[11] + out.hidden_states[14] + out.hidden_states[16]) / 3
            batch = {"wav": wav_tensor, "ref_wav": ref_tensor, "feat": feat.to(self.device)}
            with torch.no_grad():
                _, global_ids = self.audio_tokenizer.model.tokenize(batch)
            global_prefix = "".join(f"<|bicodec_global_{i}|>" for i in global_ids.squeeze().cpu().numpy())

        prompt = "".join([
            "<|task_tts|>",
            "<|start_content|>", prompt_text, "<|end_content|>",
            "<|start_global_token|>", global_prefix,
        ])

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        decoded = self.tokenizer.batch_decode(trimmed, skip_special_tokens=False)[0]

        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", decoded)
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", decoded)

        if not semantic_matches:
            raise RuntimeError("No semantic acoustic tokens generated by the model.")

        pred_semantic = torch.tensor([int(t) for t in semantic_matches]).long().unsqueeze(0).to(self.device)
        if global_matches:
            pred_global = torch.tensor([int(t) for t in global_matches]).long().unsqueeze(0).unsqueeze(0).to(self.device)
        else:
            pred_global = torch.zeros((1, 1, 32), dtype=torch.long, device=self.device)

        wav_np = self.audio_tokenizer.detokenize(pred_global.squeeze(0), pred_semantic)
        return wav_np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="/mnt/volume_d2wey28/models/Spark-TTS-0.5B/LLM")
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--language", type=str, default="twi", help="Language code")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio clip for voice cloning")
    parser.add_argument("--output", type=str, default="spark_africa_output.wav", help="Output WAV path")
    args = parser.parse_args()

    tts = SparkTTSAfrica(model_path=args.model)
    audio = tts.generate(text=args.text, language=args.language, ref_audio=args.ref_audio)
    sf.write(args.output, audio, tts.sampling_rate)
    logger.info(f"Generated speech saved to {args.output}")


if __name__ == "__main__":
    main()
