#!/usr/bin/env python3
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from mlx_audio.tts.utils import load_model

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base-8bit"
WAVS_DIR = ROOT / "wavs"
DEFAULT_REFERENCE_TEXT = WAVS_DIR / "reference.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_BGM = WAVS_DIR / "background.wav"

FMT = "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-TTS long-form Japanese voice cloning. Generates one WAV per "
            "sentence in a temporary directory, joins them with natural pauses, "
            "optionally mixes background music (wavs/background.wav) with "
            "loudness normalization, writes one final WAV, then removes "
            "temporary files."
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
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=None,
        help=(
            "Reference voice WAV. Default: pick one .wav at random from wavs/ "
            "(only those with a same-name .txt transcript, unless --reference-text is given)"
        ),
    )
    parser.add_argument(
        "--reference-text",
        type=Path,
        default=None,
        help=(
            "Reference transcript. Default: <reference-audio-stem>.txt next to the "
            "reference audio (falls back to wavs/reference.txt if --reference-audio "
            "is given and no same-name .txt exists)"
        ),
    )
    parser.add_argument("--language", default="Japanese")
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=850,
        help="Pause after normal sentence endings (default: 850)",
    )
    parser.add_argument(
        "--paragraph-pause-ms",
        type=int,
        default=720,
        help="Pause between paragraphs (default: 720)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help=(
            "Linear gain multiplier applied to the final WAV (default: 1.0, e.g. 1.5). "
            "Ignored when BGM is mixed (output is loudness-normalized instead)."
        ),
    )
    parser.add_argument(
        "--bgm",
        type=Path,
        default=DEFAULT_BGM,
        help="Background music WAV (default: wavs/background.wav; skipped if the default is missing)",
    )
    parser.add_argument(
        "--no-bgm",
        action="store_true",
        help="Skip BGM mixing and output the voice only",
    )
    parser.add_argument("--voice-lufs", type=float, default=-16.0)
    parser.add_argument(
        "--bgm-offset-db",
        type=float,
        default=22.0,
        help="BGM is this many dB below the voice (default: 22)",
    )
    parser.add_argument("--true-peak", type=float, default=-1.5)
    parser.add_argument("--lra", type=float, default=11.0)
    parser.add_argument(
        "--bgm-tail",
        type=float,
        default=5.0,
        help="Seconds the BGM continues after the voice ends (default: 5)",
    )
    parser.add_argument(
        "--bgm-fade",
        type=float,
        default=3.0,
        help="BGM fade-out length in seconds (default: 3)",
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


def resolve_reference(
    audio_arg: Path | None, text_arg: Path | None
) -> tuple[Path, Path, bool]:
    """Return (reference_audio, reference_text_path, randomly_selected)."""
    if audio_arg is not None:
        audio = audio_arg.expanduser().resolve()
        if text_arg is not None:
            text = text_arg.expanduser().resolve()
        else:
            sibling = audio.with_suffix(".txt")
            text = sibling if sibling.exists() else DEFAULT_REFERENCE_TEXT.resolve()
        return audio, text, False

    if not WAVS_DIR.is_dir():
        fail(f"Reference directory not found: {WAVS_DIR}")

    candidates = sorted(
        p
        for p in WAVS_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".wav"
        and p.name != DEFAULT_BGM.name
    )
    if text_arg is None:
        candidates = [p for p in candidates if p.with_suffix(".txt").is_file()]
    if not candidates:
        if text_arg is None:
            fail(
                f"No .wav with a same-name .txt transcript found in: {WAVS_DIR} "
                f"(or pass --reference-text)"
            )
        fail(f"No .wav files found in: {WAVS_DIR}")

    audio = random.choice(candidates).resolve()
    text = (
        text_arg.expanduser().resolve()
        if text_arg is not None
        else audio.with_suffix(".txt")
    )
    return audio, text, True


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


def _fmt_t(sec: float) -> str:
    sec = max(int(sec), 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        fail(f"Command failed: {' '.join(cmd[:3])} ...\n{p.stderr[-1500:]}")
    return p.stderr + p.stdout


def _run_ffmpeg(cmd_args: list[str], total_sec: float, label: str) -> str:
    """Run ffmpeg with live progress (percent, elapsed, ETA). Returns stderr."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-progress", "pipe:1", *cmd_args]
    tty = sys.stderr.isatty()
    start = time.monotonic()
    last_pct = -10
    with tempfile.TemporaryFile("w+") as errf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=errf, text=True, bufsize=1
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                key, _, val = line.strip().partition("=")
                if key != "out_time_us" or not val.lstrip("-").isdigit():
                    continue
                done = max(int(val) / 1_000_000, 0.0)
                frac = min(done / total_sec, 1.0) if total_sec > 0 else 0.0
                pct = int(frac * 100)
                elapsed = time.monotonic() - start
                eta = elapsed * (1 - frac) / frac if frac > 0.01 else 0.0
                msg = (f"  {label}: {pct:3d}%  {_fmt_t(done)}/{_fmt_t(total_sec)}"
                       f"  elapsed {_fmt_t(elapsed)}  ETA {_fmt_t(eta)}")
                if tty:
                    print("\r" + msg + "   ", end="", file=sys.stderr, flush=True)
                elif pct - last_pct >= 10:
                    last_pct = pct
                    print(msg, file=sys.stderr, flush=True)
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        errf.seek(0)
        err = errf.read()
    if tty:
        print(file=sys.stderr)
    if rc != 0:
        fail(f"ffmpeg failed ({label}):\n{err[-1500:]}")
    print(f"  {label}: done in {_fmt_t(time.monotonic() - start)}",
          file=sys.stderr, flush=True)
    return err


def _loudnorm(path: Path, target: float, tp: float, lra: float,
              total_sec: float, label: str) -> str:
    """Pass 1: measure. Returns the pass-2 (linear) loudnorm filter string."""
    err = _run_ffmpeg([
        "-i", str(path),
        "-af", f"{FMT},loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ], total_sec, label)
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    if m["input_i"] in ("-inf", "inf"):
        fail(f"Cannot measure loudness (silent input?): {path}")
    print(f"  measured: I={m['input_i']} LUFS, TP={m['input_tp']} dB, "
          f"LRA={m['input_lra']} LU", file=sys.stderr, flush=True)
    return (
        f"loudnorm=I={target}:TP={tp}:LRA={lra}:linear=true"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}"
    )


def _duration(path: Path) -> float:
    return float(_run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ]).strip())


def _stage(n: int, text: str) -> None:
    print(f"[BGM {n}/6] {text}", file=sys.stderr, flush=True)


def mix_bgm(
    voice: Path,
    bgm: Path,
    out: Path,
    work: Path,
    *,
    voice_lufs: float,
    bg_offset: float,
    tp: float,
    lra: float,
    tail: float,
    fade: float,
) -> None:
    """Normalize voice and BGM, mix (BGM loops, continues `tail` s after the
    voice and fades out), then normalize the mix. Output: 16 kHz mono 16-bit."""
    voice_dur = _duration(voice)
    bgm_dur = _duration(bgm)
    total = voice_dur + tail
    fade_st = max(total - fade, 0.0)
    print(f"Voice {voice_dur:.1f}s, BGM {bgm_dur:.1f}s -> output {total:.1f}s, "
          f"fade-out starts at {fade_st:.1f}s", file=sys.stderr, flush=True)

    _stage(1, "Measuring voice loudness")
    vf = _loudnorm(voice, voice_lufs, tp, lra, voice_dur, "voice")
    _stage(2, "Measuring BGM loudness")
    bf = _loudnorm(bgm, voice_lufs - bg_offset, tp, lra, bgm_dur, "bgm")

    pre = work / "pre_mix.wav"
    _stage(3, "Mixing voice + BGM")
    _run_ffmpeg([
        "-y", "-v", "error", "-i", str(voice), "-i", str(bgm),
        "-filter_complex",
        f"[0:a]{FMT},{vf},aresample=16000,apad=pad_dur={tail}[v];"
        f"[1:a]{FMT},{bf},aresample=16000,aloop=loop=-1:size=2147483647,"
        f"atrim=0:{total:.6f},afade=t=out:st={fade_st:.6f}:d={fade}[b];"
        f"[v][b]amix=inputs=2:duration=longest:normalize=0,atrim=0:{total:.6f},"
        f"aformat=sample_fmts=s16:channel_layouts=mono[m]",
        "-map", "[m]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(pre),
    ], total, "mix")
    _stage(4, "Measuring mix loudness")
    ff = _loudnorm(pre, voice_lufs, tp, lra, total, "mix")
    _stage(5, "Normalizing final output")
    _run_ffmpeg([
        "-y", "-v", "error", "-i", str(pre),
        "-af", f"{ff},{FMT}",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out),
    ], total, "final")
    _stage(6, "Done")


def main() -> None:
    args = parse_args()
    input_path = args.input_text.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    reference_audio, reference_text_path, randomly_selected = resolve_reference(
        args.reference_audio, args.reference_text
    )

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
    if args.gain <= 0:
        fail("Gain must be positive.")

    bgm_path: Path | None = None
    if not args.no_bgm:
        candidate = args.bgm.expanduser().resolve()
        if candidate.is_file():
            bgm_path = candidate
        elif args.bgm != DEFAULT_BGM:
            fail(f"BGM not found: {candidate}")
        else:
            print(f"Note: {candidate} not found; BGM mixing skipped.", file=sys.stderr)
    if bgm_path is not None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            fail("ffmpeg/ffprobe not found in PATH.")
        if args.gain != 1.0:
            print("Note: --gain is ignored when BGM is mixed.", file=sys.stderr)

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
    print(
        f"Reference audio: {reference_audio}"
        f"{' (randomly selected)' if randomly_selected else ''}"
    )
    print(f"Reference transcript: {reference_text_path}")
    print(f"Temporary directory: {temp_dir}")
    print(f"Output: {output_path}")
    print(f"Gain: {args.gain}")
    print(f"BGM: {bgm_path if bgm_path else 'none'}")
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

        merged = np.concatenate(final_parts).astype(np.float32)

        if bgm_path is not None:
            voice_wav = temp_dir / "voice.wav"
            sf.write(voice_wav, np.clip(merged, -1.0, 1.0), sample_rate, subtype="PCM_16")
            print("Mixing BGM and normalizing loudness...", flush=True)
            mix_bgm(
                voice_wav,
                bgm_path,
                output_path,
                temp_dir,
                voice_lufs=args.voice_lufs,
                bg_offset=args.bgm_offset_db,
                tp=args.true_peak,
                lra=args.lra,
                tail=args.bgm_tail,
                fade=args.bgm_fade,
            )
            sample_rate = 16000
        else:
            peak = float(np.max(np.abs(merged)))
            scaled_peak = peak * args.gain
            if scaled_peak > 1.0:
                print(
                    f"Warning: peak {peak:.3f} x {args.gain} = {scaled_peak:.3f} "
                    f"exceeds 1.0; clipping",
                    file=sys.stderr,
                )
            merged = np.clip(merged * args.gain, -1.0, 1.0)
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
