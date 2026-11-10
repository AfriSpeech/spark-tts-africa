"""Data preparation for Spark-TTS Africa.

Processes audio-text manifests from the aligned African corpus:
1. Loads audio slices directly using soundfile.
2. Converts transcripts to IPA phonemes via africa-g2p.
3. Encodes audio into BiCodec global and semantic acoustic tokens.
4. Produces tokenized sequence strings for causal LM fine-tuning.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Canonical language alias mapping for Africa-G2P
RULE_ALIASES = {
    "tw": "twi", "ak": "twi", "aka": "twi", "akan": "twi",
    "sw": "swh", "swa": "swh",
    "ha": "hau", "hausa": "hau",
    "yo": "yor", "yoruba": "yor",
    "ig": "ibo", "igbo": "ibo",
    "zu": "zul", "zulu": "zul",
    "en": "eng", "english": "eng",
    "am": "amh", "amharic": "amh",
    "sn": "sna", "shona": "sna",
    "rw": "kin", "kinyarwanda": "kin",
    "ti": "tir", "tigrinya": "tir",
    "so": "som", "somali": "som",
    "ln": "lin", "lingala": "lin",
    "wo": "wol", "wolof": "wol",
    "xh": "xho", "xhosa": "xho",
}


def text_to_ipa(text: str, lang: str) -> str:
    """Convert text to phonetic IPA using africa-g2p with fallback."""
    if not text:
        return ""
    try:
        import africa_g2p
        code = RULE_ALIASES.get(lang.lower(), lang.lower())
        return africa_g2p.g2p(text, code, output="ipa")
    except Exception:
        return text


def load_audio_slice(path: str, start: float, duration: float, target_sr: int = 16000) -> np.ndarray:
    """Load a slice of audio file and resample to target_sr (mono float32)."""
    info = sf.info(path)
    sr = info.samplerate
    start_frame = max(0, int(round(start * sr)))
    frames = int(round(duration * sr))
    data, _ = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=True)
    wav = data.mean(1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav)[None], orig_freq=sr, new_freq=target_sr
        )[0].numpy()
    return wav


def extract_bicodec_tokens(
    audio_tokenizer,
    audio_np: np.ndarray,
    sr: int = 16000,
) -> tuple[str, str]:
    """Tokenize audio waveform into BiCodec global and semantic token strings."""
    device = audio_tokenizer.device
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).float().to(device)
    ref_clip_np = audio_tokenizer.get_ref_clip(audio_np)
    ref_tensor = torch.from_numpy(ref_clip_np).unsqueeze(0).float().to(device)

    wav_np = audio_tensor.squeeze(0).cpu().numpy()
    processed = audio_tokenizer.processor(wav_np, sampling_rate=16000, return_tensors="pt", padding=True)
    iv = processed.input_values.to(audio_tokenizer.feature_extractor.device)
    out = audio_tokenizer.feature_extractor(iv)
    feat = (out.hidden_states[11] + out.hidden_states[14] + out.hidden_states[16]) / 3

    batch = {
        "wav": audio_tensor,
        "ref_wav": ref_tensor,
        "feat": feat.to(device),
    }
    with torch.no_grad():
        semantic_ids, global_ids = audio_tokenizer.model.tokenize(batch)

    global_tokens = "".join(f"<|bicodec_global_{i}|>" for i in global_ids.squeeze().cpu().numpy())
    semantic_tokens = "".join(f"<|bicodec_semantic_{i}|>" for i in semantic_ids.squeeze().cpu().numpy())
    return global_tokens, semantic_tokens


def format_spark_sequence(text: str, global_tokens: str, semantic_tokens: str) -> str:
    """Formats prompt text and acoustic tokens into Spark-TTS training sequence."""
    return "".join([
        "<|task_tts|>",
        "<|start_content|>", text, "<|end_content|>",
        "<|start_global_token|>", global_tokens, "<|end_global_token|>",
        "<|start_semantic_token|>", semantic_tokens, "<|end_semantic_token|>",
        "<|im_end|>",
    ])
