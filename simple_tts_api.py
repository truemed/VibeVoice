"""
Simple TTS API server for VibeVoice.

POST /generate (multipart/form-data) with fields:
- script (str, required)
- num_speakers (int, optional, default=1)
- speaker_1..speaker_4 (str, optional)
- cfg_scale (float, optional, default=1.3)
- inference_steps (int, optional, default=10)
- max_length_times (float, optional, default=2.0)
- max_new_tokens (int, optional)
- seed (int, optional, default=-1)
- long_form_strategy (bool, optional, default=false)
- chunk_join_silence_ms (float, optional, default=100.0)
- disable_voice_cloning (bool, optional, default=true)
- speech_rate (float, optional, default=1.0, range=0.7..1.3)
- reference_audio (file, optional)

Returns: WAV file (no streaming)
"""

import argparse
import os
import tempfile
from typing import List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
torch.backends.nnpack.enabled = False
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.background import BackgroundTasks
from transformers import set_seed

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from bgm_artifact_detection import score_background_artifacts_in_silence
from long_form_chunking import chunk_script_long_form
from speech_rate_processing import apply_speech_rate, validate_speech_rate
from short_text_mode import (
    SHORT_MODE_CANDIDATES,
    build_short_text_template,
    count_words,
    extract_target_segment_from_template,
    score_short_text_candidate,
    should_use_short_text_mode,
)


SAMPLE_RATE = 24000


def read_audio_file(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    wav, sr = sf.read(file_path)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


def convert_to_int16(data: np.ndarray) -> np.ndarray:
    data = np.array(data, dtype=np.float32)
    if np.max(np.abs(data)) > 1.0:
        data = data / np.max(np.abs(data))
    return (data * 32767).astype(np.int16)


def format_script_with_speakers(script: str, num_speakers: int) -> str:
    lines = [line.strip() for line in script.split("\n") if line.strip()]
    if not lines:
        return ""

    formatted_lines = []
    for line in lines:
        if line.lower().startswith("speaker ") and ":" in line:
            formatted_lines.append(line)
        else:
            speaker_id = len(formatted_lines) % num_speakers
            formatted_lines.append(f"Speaker {speaker_id}: {line}")
    return "\n".join(formatted_lines)




class SimpleTtsServer:
    def __init__(self, model_path: str, device: str, inference_steps: int):
        self.model_path = model_path
        self.device = device
        self.inference_steps = inference_steps
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        if self.device.lower() == "mpx":
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"

        self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)

        if self.device == "mps":
            load_dtype = torch.float32
            attn_impl = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else:
            load_dtype = torch.float32
            attn_impl = "sdpa"

        try:
            if self.device == "mps":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            elif self.device == "cuda":
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map="cuda",
                )
            else:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=attn_impl,
                    device_map="cpu",
                )
        except Exception as exc:
            if attn_impl == "flash_attention_2":
                fallback_attn = "sdpa"
                print(f"Warning: {exc}")
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    attn_implementation=fallback_attn,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

    def generate(self,
                 script: str,
                 num_speakers: int,
                 speaker_names: List[Optional[str]],
                 cfg_scale: float,
                 inference_steps: int,
                 max_length_times: float,
                 max_new_tokens: Optional[int],
                 seed: Optional[int],
                 long_form_strategy: bool,
                 chunk_join_silence_ms: float,
                 disable_voice_cloning: bool,
                 speech_rate: float,
                 reference_audio: Optional[np.ndarray],
                 output_tail_silence_sec: float = 0.1) -> str:
        voice_samples = None
        if not disable_voice_cloning:
            if reference_audio is None:
                raise ValueError("reference_audio is required when voice cloning is enabled")
            voice_samples = [reference_audio for _ in range(num_speakers)]
        target_device = self.device if self.device in ("cuda", "mps") else "cpu"

        def synthesize_audio(raw_script: str,
                            run_inference_steps: int,
                            run_max_length_times: float,
                            run_cfg_scale: float,
                            run_seed: Optional[int]) -> np.ndarray:
            if run_seed is not None:
                set_seed(run_seed)

            self.model.set_ddpm_inference_steps(num_steps=run_inference_steps)

            formatted_script = format_script_with_speakers(raw_script, num_speakers)
            if not formatted_script:
                raise ValueError("script is required")

            scripts_to_generate = [formatted_script]
            if long_form_strategy:
                scripts_to_generate = chunk_script_long_form(formatted_script)
                if not scripts_to_generate:
                    raise ValueError("script is required")
                print(f"Long-form strategy enabled: {len(scripts_to_generate)} chunk(s).")

            generated_chunks: List[np.ndarray] = []
            for idx, script_chunk in enumerate(scripts_to_generate, start=1):
                inputs = self.processor(
                    text=[script_chunk],
                    voice_samples=[voice_samples] if voice_samples is not None else None,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(target_device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    max_length_times=run_max_length_times,
                    cfg_scale=run_cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    is_prefill=not disable_voice_cloning,
                )

                if getattr(outputs, "reach_max_step_sample", None) is not None:
                    if torch.any(outputs.reach_max_step_sample).item():
                        print(
                            f"Warning: chunk {idx} hit max-step cap. "
                            "Try increasing max_length_times or max_new_tokens."
                        )

                if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                    raise RuntimeError(f"No audio output generated for chunk {idx}")

                audio_chunk = outputs.speech_outputs[0]
                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype == torch.bfloat16:
                        audio_chunk = audio_chunk.float()
                    audio_chunk = audio_chunk.detach().cpu().numpy().astype(np.float32)
                audio_chunk = np.array(audio_chunk, dtype=np.float32).squeeze()

                if audio_chunk.size == 0:
                    raise RuntimeError(f"Generated empty audio for chunk {idx}")

                generated_chunks.append(audio_chunk)

            if len(generated_chunks) == 1:
                return generated_chunks[0]

            join_silence_samples = int((chunk_join_silence_ms / 1000.0) * SAMPLE_RATE)
            if join_silence_samples > 0:
                join_silence = np.zeros(join_silence_samples, dtype=np.float32)
                pieces: List[np.ndarray] = []
                for idx, chunk in enumerate(generated_chunks):
                    if idx > 0:
                        pieces.append(join_silence)
                    pieces.append(chunk)
                return np.concatenate(pieces)

            return np.concatenate(generated_chunks)

        def retry_seed(base_seed: Optional[int], offset: int = 7919) -> int:
            if base_seed is None:
                return int(np.random.randint(0, 2**31 - 1))
            return int(base_seed + offset)

        short_mode = should_use_short_text_mode(script, long_form_strategy=long_form_strategy)
        if short_mode:
            script_word_count = max(1, count_words(script))
            wrapped_script = build_short_text_template(script)

            candidates: List[tuple[float, np.ndarray]] = []
            for idx in range(SHORT_MODE_CANDIDATES):
                candidate_seed = (seed + idx) if seed is not None else int(np.random.randint(0, 2**31 - 1))
                candidate_audio = synthesize_audio(
                    raw_script=wrapped_script,
                    run_inference_steps=inference_steps,
                    run_max_length_times=max_length_times,
                    run_cfg_scale=cfg_scale,
                    run_seed=candidate_seed,
                )
                target_audio = extract_target_segment_from_template(candidate_audio)
                candidate_score = score_short_text_candidate(target_audio, script_word_count)
                candidates.append((candidate_score, target_audio))

            candidates.sort(key=lambda x: x[0], reverse=True)
            audio = candidates[0][1]

            artifact_score = score_background_artifacts_in_silence(audio)
            if artifact_score >= 0.30:
                retry_audio = synthesize_audio(
                    raw_script=wrapped_script,
                    run_inference_steps=inference_steps,
                    run_max_length_times=max_length_times,
                    run_cfg_scale=cfg_scale,
                    run_seed=retry_seed(seed),
                )
                retry_target_audio = extract_target_segment_from_template(retry_audio)
                retry_artifact_score = score_background_artifacts_in_silence(retry_target_audio)
                if retry_artifact_score < artifact_score:
                    audio = retry_target_audio
        else:
            audio = synthesize_audio(
                raw_script=script,
                run_inference_steps=inference_steps,
                run_max_length_times=max_length_times,
                run_cfg_scale=cfg_scale,
                run_seed=seed,
            )

            artifact_score = score_background_artifacts_in_silence(audio)
            if artifact_score >= 0.30:
                retry_audio = synthesize_audio(
                    raw_script=script,
                    run_inference_steps=inference_steps,
                    run_max_length_times=max_length_times,
                    run_cfg_scale=cfg_scale,
                    run_seed=retry_seed(seed),
                )
                retry_artifact_score = score_background_artifacts_in_silence(retry_audio)
                if retry_artifact_score < artifact_score:
                    audio = retry_audio

        audio = apply_speech_rate(
            audio=audio,
            speech_rate=speech_rate,
            sample_rate=SAMPLE_RATE,
            prefer_pyrubberband=True,
        )

        # Add a short configurable silence tail to avoid clipped sounding endings.
        if output_tail_silence_sec > 0:
            pad_samples = int(SAMPLE_RATE * output_tail_silence_sec)
            if pad_samples > 0:
                audio = np.concatenate([audio, np.zeros(pad_samples, dtype=np.float32)])

        audio_int16 = convert_to_int16(audio)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, audio_int16, SAMPLE_RATE, subtype="PCM_16")
        return tmp.name


app = FastAPI()
server: Optional[SimpleTtsServer] = None


@app.post("/generate")
async def generate_audio(
    background_tasks: BackgroundTasks,
    script: str = Form(...),
    num_speakers: int = Form(1),
    speaker_1: Optional[str] = Form(None),
    speaker_2: Optional[str] = Form(None),
    speaker_3: Optional[str] = Form(None),
    speaker_4: Optional[str] = Form(None),
    cfg_scale: float = Form(1.3),
    inference_steps: int = Form(10),
    max_length_times: float = Form(2.0),
    max_new_tokens: Optional[int] = Form(None),
    seed: int = Form(-1),
    long_form_strategy: bool = Form(False),
    chunk_join_silence_ms: float = Form(100.0),
    disable_voice_cloning: bool = Form(True),
    speech_rate: float = Form(1.0),
    output_tail_silence_sec: float = Form(0.1),
    reference_audio: Optional[UploadFile] = File(None),
):
    if server is None:
        raise RuntimeError("Server not initialized")

    if not script.strip():
        raise ValueError("script is required")

    if num_speakers < 1 or num_speakers > 4:
        raise ValueError("num_speakers must be between 1 and 4")

    if max_length_times <= 0:
        raise ValueError("max_length_times must be > 0")

    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0 when provided")

    if output_tail_silence_sec < 0 or output_tail_silence_sec > 5:
        raise ValueError("output_tail_silence_sec must be between 0 and 5")

    if chunk_join_silence_ms < 0 or chunk_join_silence_ms > 2000:
        raise ValueError("chunk_join_silence_ms must be between 0 and 2000")

    validate_speech_rate(speech_rate)

    speaker_names = [speaker_1, speaker_2, speaker_3, speaker_4][:num_speakers]

    reference_audio_data = None
    reference_audio_path = None
    if reference_audio is not None:
        filename = reference_audio.filename or "reference_audio"
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext not in (".wav", ".mp3"):
            raise ValueError("reference_audio must be .wav or .mp3")

        tmp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        reference_audio_path = tmp_ref.name
        tmp_ref.close()
        file_bytes = await reference_audio.read()
        with open(reference_audio_path, "wb") as out_f:
            out_f.write(file_bytes)
        reference_audio_data = read_audio_file(reference_audio_path)

    seed_value = None if seed < 0 else int(seed)

    wav_path = server.generate(
        script=script,
        num_speakers=num_speakers,
        speaker_names=speaker_names,
        cfg_scale=cfg_scale,
        inference_steps=inference_steps,
        max_length_times=max_length_times,
        max_new_tokens=max_new_tokens,
        seed=seed_value,
        long_form_strategy=long_form_strategy,
        chunk_join_silence_ms=chunk_join_silence_ms,
        disable_voice_cloning=disable_voice_cloning,
        speech_rate=speech_rate,
        reference_audio=reference_audio_data,
        output_tail_silence_sec=output_tail_silence_sec,
    )

    background_tasks.add_task(lambda: os.remove(wav_path) if os.path.exists(wav_path) else None)
    if reference_audio_path is not None:
        background_tasks.add_task(lambda: os.remove(reference_audio_path) if os.path.exists(reference_audio_path) else None)
    return FileResponse(wav_path, media_type="audio/wav", filename="output.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple VibeVoice TTS API")
    parser.add_argument("--model_path", required=True, help="Path or HF ID for VibeVoice model")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")),
        help="Device for inference: cuda | mps | cpu",
    )
    parser.add_argument("--inference_steps", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    global server
    args = parse_args()
    server = SimpleTtsServer(
        model_path=args.model_path,
        device=args.device,
        inference_steps=args.inference_steps,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
