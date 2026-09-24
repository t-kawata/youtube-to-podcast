#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base-8bit"
DEFAULT_REFERENCE_AUDIO = ROOT / "wavs" / "reference.wav"
DEFAULT_REFERENCE_TEXT = ROOT / "wavs" / "reference.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-TTS long-form Japanese voice cloning. Generates one WAV per "
            "sentence in a temporary directory, joins them with natural pauses, "
            "writes one final WAV, then removes temporary files."
        )
    )
    parser.add_argument(
        "input_text",
        type=Path,
        help="UTF-8 input .txt file, e.g. texts/sample01.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Final WAV path. Default: outputs/<input-stem>.wav",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-audio", type=Path, default=DEFAULT_REFERENCE_AUDIO)
    parser.add_argument("--reference-text", type=Path, default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--language", default="Japanese")
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=360,
        help="Pause after normal sentence endings (default: 360)",
    )
    parser.add_argument(
        "--paragraph-pause-ms",
        type=int,
        default=720,
        help="Pause between paragraphs (default: 720)",
    )
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.82)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.7)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep per-sentence WAV files for inspection instead of deleting them",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(message)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def split_long_unit(unit: str, limit: int = 110) -> list[str]:
    if len(unit) <= limit:
        return [unit]

    pieces = []
    remaining = unit
    separators = "、，；："

    while len(remaining) > limit:
        cut = max(remaining.rfind(sep, 0, limit + 1) for sep in separators)
        if cut < max(20, limit // 3):
            cut = limit
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        else:
            pieces.append(remaining[: cut + 1].strip())
            remaining = remaining[cut + 1 :].strip()

    if remaining:
        pieces.append(remaining)
    return pieces


def split_document(text: str) -> list[tuple[str, int]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    items: list[tuple[str, int]] = []

    for paragraph_index, paragraph in enumerate(paragraphs):
        paragraph = re.sub(r"\s*\n\s*", " ", paragraph)
        units = re.split(r"(?<=[。！？!?])\s*", paragraph)
        sentences: list[str] = []

        for unit in units:
            unit = unit.strip()
            if not unit:
                continue
            sentences.extend(split_long_unit(unit))

        for sentence_index, sentence in enumerate(sentences):
            is_last_in_paragraph = sentence_index == len(sentences) - 1
            is_last_paragraph = paragraph_index == len(paragraphs) - 1
            pause_kind = 2 if is_last_in_paragraph and not is_last_paragraph else 1
            items.append((sentence, pause_kind))

    if items:
        last_text, _ = items[-1]
        items[-1] = (last_text, 0)
    return items


def write_manifest(temp_dir: Path, jobs: list[tuple[str, int]]) -> None:
    lines = []
    for index, (text, pause_kind) in enumerate(jobs, start=1):
        lines.append(f"{index:04d}\tpause_kind={pause_kind}\t{text}")
    (temp_dir / "manifest.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = args.input_text.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    reference_audio = args.reference_audio.expanduser().resolve()
    reference_text_path = args.reference_text.expanduser().resolve()

    for path, label in (
        (input_path, "Input text"),
        (model_path, "Model directory"),
        (reference_audio, "Reference audio"),
        (reference_text_path, "Reference transcript"),
    ):
        if not path.exists():
            fail(f"{label} not found: {path}")

    if args.pause_ms < 0 or args.paragraph_pause_ms < 0:
        fail("Pause durations must be non-negative.")

    source_text = normalize_text(input_path.read_text(encoding="utf-8"))
    reference_text = normalize_text(reference_text_path.read_text(encoding="utf-8"))
    if not source_text:
        fail(f"Input text is empty: {input_path}")
    if not reference_text:
        fail(f"Reference transcript is empty: {reference_text_path}")

    jobs = split_document(source_text)
    if not jobs:
        fail("No synthesizable sentences were found.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else (DEFAULT_OUTPUT_DIR / f"{input_path.stem}.wav").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_parent = ROOT / ".tmp_tts"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{input_path.stem}-", dir=temp_parent))

    print(f"Input: {input_path}")
    print(f"Sentences: {len(jobs)}")
    print(f"Temporary directory: {temp_dir}")
    print(f"Output: {output_path}")
    write_manifest(temp_dir, jobs)

    try:
        print(f"Loading model: {model_path}", flush=True)
        model = load_model(str(model_path))

        final_parts: list[np.ndarray] = []
        sample_rate: int | None = None

        for index, (sentence, pause_kind) in enumerate(jobs, start=1):
            print(f"[{index:04d}/{len(jobs):04d}] {sentence}", flush=True)
            results = model.generate(
                text=sentence,
                language=args.language,
                ref_audio=str(reference_audio),
                ref_text=reference_text,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_tokens=args.max_tokens,
            )
            result = next(iter(results), None)
            if result is None:
                fail(f"No audio generated for sentence {index}: {sentence}")

            audio = np.asarray(result.audio).squeeze()
            if audio.ndim != 1 or audio.size == 0:
                fail(f"Invalid audio generated for sentence {index}: {sentence}")

            if sample_rate is None:
                sample_rate = int(result.sample_rate)
            elif sample_rate != int(result.sample_rate):
                fail(
                    f"Sample-rate mismatch at sentence {index}: "
                    f"expected {sample_rate}, got {result.sample_rate}"
                )

            part_path = temp_dir / f"{index:04d}.wav"
            sf.write(part_path, audio, sample_rate, subtype="PCM_16")
            final_parts.append(audio)

            if pause_kind:
                pause_ms = args.paragraph_pause_ms if pause_kind == 2 else args.pause_ms
                if pause_ms:
                    final_parts.append(
                        np.zeros(round(sample_rate * pause_ms / 1000), dtype=audio.dtype)
                    )

        if sample_rate is None or not final_parts:
            fail("No final audio was produced.")

        merged = np.concatenate(final_parts)
        sf.write(output_path, merged, sample_rate, subtype="PCM_16")
        print(
            f"Completed: {output_path} "
            f"({sample_rate} Hz, {len(jobs)} sentence files generated)",
            flush=True,
        )

    except KeyboardInterrupt:
        print("\nInterrupted. Final output was not completed.", file=sys.stderr)
        raise SystemExit(130)
    finally:
        if args.keep_temp:
            print(f"Temporary files retained: {temp_dir}", flush=True)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            try:
                temp_parent.rmdir()
            except OSError:
                pass
            print("Temporary sentence WAV files removed.", flush=True)


if __name__ == "__main__":
    main()
