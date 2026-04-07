import re
from typing import List

import numpy as np


SAMPLE_RATE = 24000

SHORT_TEXT_WORD_THRESHOLD = 2
SHORT_MODE_CANDIDATES = 2
SHORT_MODE_BUFFER_BASE_TEXT = "This is a buffer"


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text))


def should_use_short_text_mode(script: str, long_form_strategy: bool) -> bool:
    if long_form_strategy:
        return False
    script_text = script.strip()
    if not script_text:
        return False
    return count_words(script_text) <= SHORT_TEXT_WORD_THRESHOLD


def _get_target_terminal_punctuation(text: str) -> str:
    stripped = text.rstrip()
    if not stripped:
        return "."
    last = stripped[-1]
    if last in ("?", "!", "."):
        return last
    return "."


def build_short_text_template(target_text: str) -> str:
    cleaned = target_text.strip()
    if not cleaned:
        return cleaned

    punctuation = _get_target_terminal_punctuation(cleaned)
    buffer_text = f"{SHORT_MODE_BUFFER_BASE_TEXT}{punctuation}"
    return f"{buffer_text}\n\n{cleaned}\n\n{buffer_text}"


def _detect_voiced_segments(audio: np.ndarray,
                           threshold_ratio: float = 0.06,
                           min_silence_sec: float = 0.12,
                           min_segment_sec: float = 0.06) -> List[tuple[int, int]]:
    if audio.size == 0:
        return []

    x = np.abs(np.asarray(audio, dtype=np.float32))
    if x.size == 0:
        return []
    peak = float(np.max(x))
    if peak <= 0:
        return []

    threshold = max(peak * threshold_ratio, 0.003)
    voiced = x > threshold
    if not np.any(voiced):
        return []

    starts = np.where(np.logical_and(voiced, np.concatenate(([True], ~voiced[:-1]))))[0]
    ends = np.where(np.logical_and(voiced, np.concatenate((~voiced[1:], [True]))))[0] + 1
    segments = [(int(s), int(e)) for s, e in zip(starts, ends)]

    merged: List[tuple[int, int]] = []
    min_silence_samples = int(min_silence_sec * SAMPLE_RATE)
    for seg_start, seg_end in segments:
        if not merged:
            merged.append((seg_start, seg_end))
            continue
        last_start, last_end = merged[-1]
        if seg_start - last_end <= min_silence_samples:
            merged[-1] = (last_start, seg_end)
        else:
            merged.append((seg_start, seg_end))

    min_segment_samples = int(min_segment_sec * SAMPLE_RATE)
    return [(s, e) for s, e in merged if (e - s) >= min_segment_samples]


def extract_target_segment_from_template(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio

    segments = _detect_voiced_segments(audio)
    if not segments:
        return audio

    center = len(audio) / 2.0
    middle_index = min(range(len(segments)), key=lambda i: abs(((segments[i][0] + segments[i][1]) / 2.0) - center))
    seg_start, seg_end = segments[middle_index]

    pad = int(0.05 * SAMPLE_RATE)
    start = max(0, seg_start - pad)
    end = min(len(audio), seg_end + pad)
    clipped = audio[start:end]
    return np.array(clipped, dtype=np.float32)


def score_short_text_candidate(audio: np.ndarray, word_count: int) -> float:
    if audio.size == 0:
        return -1e9

    x = np.asarray(audio, dtype=np.float32)
    duration_sec = len(x) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(np.square(x)) + 1e-9))
    peak = float(np.max(np.abs(x)) + 1e-9)
    crest_ratio = peak / max(rms, 1e-5)

    expected_duration = 0.22 + (0.38 * max(word_count, 1))
    duration_penalty = abs(duration_sec - expected_duration)

    clipping_penalty = 2.0 if peak >= 0.995 else 0.0
    low_energy_penalty = 1.5 if rms < 0.01 else 0.0
    noisy_penalty = max(0.0, crest_ratio - 18.0) * 0.05

    return 4.0 - (2.2 * duration_penalty) - clipping_penalty - low_energy_penalty - noisy_penalty