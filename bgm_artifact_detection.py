from typing import List

import numpy as np


SAMPLE_RATE = 24000


def _iter_regions(mask: np.ndarray) -> List[tuple[int, int]]:
    if mask.size == 0:
        return []
    starts = np.where(np.logical_and(mask, np.concatenate(([True], ~mask[:-1]))))[0]
    ends = np.where(np.logical_and(mask, np.concatenate((~mask[1:], [True]))))[0] + 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def score_background_artifacts_in_silence(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float32).squeeze()
    if x.size < int(0.2 * SAMPLE_RATE):
        return 0.0

    abs_x = np.abs(x)
    win = max(1, int(0.02 * SAMPLE_RATE))
    smoothed = np.convolve(abs_x, np.ones(win, dtype=np.float32) / float(win), mode="same")

    high = float(np.percentile(smoothed, 95))
    if high <= 1e-6:
        return 0.0

    speech_mask = smoothed >= max(0.003, 0.18 * high)
    silence_mask = ~speech_mask
    silence_regions = _iter_regions(silence_mask)
    min_silence_len = int(0.08 * SAMPLE_RATE)
    silence_regions = [(s, e) for s, e in silence_regions if (e - s) >= min_silence_len]
    if not silence_regions:
        return 0.0

    speech_regions = _iter_regions(speech_mask)
    speech_rms_values: List[float] = []
    for s, e in speech_regions:
        seg = x[s:e]
        if seg.size > 0:
            speech_rms_values.append(float(np.sqrt(np.mean(np.square(seg)) + 1e-9)))
    speech_rms = float(np.median(speech_rms_values)) if speech_rms_values else 0.0
    if speech_rms <= 1e-6:
        return 0.0

    best_score = 0.0
    for s, e in silence_regions:
        seg = x[s:e]
        if seg.size < 32:
            continue
        seg_rms = float(np.sqrt(np.mean(np.square(seg)) + 1e-9))
        if seg_rms < max(0.0015, speech_rms * 0.06):
            continue

        window = np.hanning(seg.size).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(seg * window))
        if spectrum.size < 8:
            continue

        power = np.square(spectrum) + 1e-12
        peak_to_mean = float(np.max(power) / np.mean(power))
        flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))

        tonal_score = max(0.0, (peak_to_mean - 20.0) / 30.0) + max(0.0, (0.22 - flatness) / 0.22)
        rel_energy = seg_rms / speech_rms
        region_score = rel_energy * tonal_score
        if region_score > best_score:
            best_score = region_score

    return best_score