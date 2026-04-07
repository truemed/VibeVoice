"""
Batch TTS using the Gradio demo server.

Example:
python batch_tts_from_gradio.py --server http://127.0.0.1:7860 --input_dir ./demo/text_examples --output_dir ./batch_outputs --num_speakers 1 --speaker_names en-Alice_woman
"""

import argparse
import os
import shutil
import urllib.request
from typing import List, Optional

from gradio_client import Client, handle_file


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _iter_text_files(input_dir: str) -> List[str]:
    files = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".txt"):
            files.append(os.path.join(input_dir, name))
    return sorted(files)


def _download_or_copy(src: str, dest: str) -> None:
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src) as response, open(dest, "wb") as out_f:
            shutil.copyfileobj(response, out_f)
        return
    if os.path.exists(src):
        shutil.copy2(src, dest)
        return
    raise FileNotFoundError(f"Output file not found or unsupported path: {src}")


def _normalize_speakers(speakers: List[str]) -> List[Optional[str]]:
    normalized = list(speakers[:4])
    while len(normalized) < 4:
        normalized.append(None)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch TTS using the Gradio demo server")
    parser.add_argument("--server", required=True, help="Gradio server URL, e.g. http://127.0.0.1:7860")
    parser.add_argument("--input_dir", required=True, help="Directory containing .txt files")
    parser.add_argument("--output_dir", required=True, help="Directory to write .wav outputs")
    parser.add_argument("--num_speakers", type=int, default=1, help="Number of speakers (1-4)")
    parser.add_argument(
        "--speaker_names",
        nargs="+",
        default=["en-Alice_woman"],
        help="Speaker preset names in order (1-4 names)",
    )
    parser.add_argument("--reference_audio", default=None, help="Optional reference audio file path")
    parser.add_argument("--cfg_scale", type=float, default=1.3, help="CFG scale")
    parser.add_argument("--inference_steps", type=int, default=10, help="DDPM inference steps")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 for random)")
    parser.add_argument("--disable_voice_cloning", action="store_true", help="Disable voice cloning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    if args.num_speakers < 1 or args.num_speakers > 4:
        raise SystemExit("--num_speakers must be between 1 and 4")

    _ensure_dir(args.output_dir)

    client = Client(args.server)

    speaker_1, speaker_2, speaker_3, speaker_4 = _normalize_speakers(args.speaker_names)
    reference_audio = handle_file(args.reference_audio) if args.reference_audio else None
    seed = None if args.seed < 0 else int(args.seed)

    text_files = _iter_text_files(args.input_dir)
    if not text_files:
        raise SystemExit(f"No .txt files found in {args.input_dir}")

    for text_path in text_files:
        with open(text_path, "r", encoding="utf-8") as f:
            script = f.read().strip()

        if not script:
            print(f"Skipping empty file: {text_path}")
            continue

        result = client.predict(
            args.num_speakers,
            script,
            speaker_1,
            speaker_2,
            speaker_3,
            speaker_4,
            reference_audio,
            args.cfg_scale,
            args.inference_steps,
            seed,
            args.disable_voice_cloning,
            api_name="/generate",
        )

        # Expected outputs: [streaming_audio, complete_audio, log_output, streaming_status, generate_btn, stop_btn]
        complete_audio = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else None
        if complete_audio is None:
            print(f"No output for {text_path}")
            continue

        if isinstance(complete_audio, dict):
            output_src = complete_audio.get("path") or complete_audio.get("url")
        else:
            output_src = str(complete_audio)

        base_name = os.path.splitext(os.path.basename(text_path))[0]
        output_path = os.path.join(args.output_dir, f"{base_name}.wav")
        _download_or_copy(output_src, output_path)
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
