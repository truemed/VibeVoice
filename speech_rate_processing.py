import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np


MIN_SPEECH_RATE = 0.7
MAX_SPEECH_RATE = 1.3
RUBBERBAND_DIRNAME = "rubberband-4.0.0-gpl-executable-windows"


try:
    import pyrubberband as pyrb
except Exception:
    pyrb = None


def _find_rubberband_executable(search_dir: Path) -> Optional[Path]:
    if not search_dir.exists() or not search_dir.is_dir():
        return None

    for pattern in ("rubberband.exe", "rubberband*.exe"):
        matches = sorted(search_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _configure_rubberband_binary() -> None:
    # Support the requested relative location: .\TTS\rubberband-4.0.0-gpl-executable-windows\
    this_file = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "TTS" / RUBBERBAND_DIRNAME,
        this_file.parents[2] / "TTS" / RUBBERBAND_DIRNAME,
        this_file.parents[1].parent / RUBBERBAND_DIRNAME,
    ]

    for folder in candidates:
        if not folder.exists():
            continue

        folder_str = str(folder)
        current_path = os.environ.get("PATH", "")
        if folder_str not in current_path:
            os.environ["PATH"] = f"{folder_str}{os.pathsep}{current_path}" if current_path else folder_str

        exe = _find_rubberband_executable(folder)
        if exe is not None:
            os.environ["RUBBERBAND"] = str(exe)
        break


_configure_rubberband_binary()


def validate_speech_rate(speech_rate: float) -> None:
    if speech_rate < MIN_SPEECH_RATE or speech_rate > MAX_SPEECH_RATE:
        raise ValueError(f"speech_rate must be between {MIN_SPEECH_RATE} and {MAX_SPEECH_RATE}")


def apply_speech_rate(
    audio: np.ndarray,
    speech_rate: float,
    sample_rate: int,
    prefer_pyrubberband: bool = True,
) -> np.ndarray:
    if audio.size == 0:
        return np.asarray(audio, dtype=np.float32)

    x = np.asarray(audio, dtype=np.float32).squeeze()
    if x.ndim != 1:
        x = np.ravel(x)

    if abs(float(speech_rate) - 1.0) < 1e-6:
        return x

    if prefer_pyrubberband and pyrb is not None:
        try:
            stretched = pyrb.time_stretch(x, sample_rate, float(speech_rate))
            return np.asarray(stretched, dtype=np.float32)
        except Exception:
            pass

    stretched = librosa.effects.time_stretch(x, rate=float(speech_rate))
    return np.asarray(stretched, dtype=np.float32)