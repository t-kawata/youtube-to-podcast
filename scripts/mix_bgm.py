#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FMT = "aresample=16000,aformat=sample_fmts=s16:channel_layouts=mono"
STAGES = 6


def fail(message: str) -> None:
    raise SystemExit(message)


def fmt_t(sec: float) -> str:
    sec = max(int(sec), 0)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        fail(f"Command failed: {' '.join(cmd[:3])} ...\n{p.stderr[-1500:]}")
    return p.stderr + p.stdout


def run_ffmpeg(args: list[str], total_sec: float, label: str) -> str:
    """Run ffmpeg with live progress (percent, speed, elapsed, ETA). Returns stderr."""
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-progress", "pipe:1", *args]
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
                msg = (f"  {label}: {pct:3d}%  {fmt_t(done)}/{fmt_t(total_sec)}"
                       f"  elapsed {fmt_t(elapsed)}  ETA {fmt_t(eta)}")
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
    print(f"  {label}: done in {fmt_t(time.monotonic() - start)}", file=sys.stderr, flush=True)
    return err


def loudnorm_filter(path: Path, target: float, tp: float, lra: float,
                    total_sec: float, label: str) -> str:
    err = run_ffmpeg([
        "-i", str(path),
        "-af", f"{FMT},loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ], total_sec, label)
    m = json.loads(err[err.rindex("{"): err.rindex("}") + 1])
    if m["input_i"] in ("-inf", "inf"):
        fail(f"Cannot measure loudness (silent input?): {path}")
    print(f"  measured: I={m['input_i']} LUFS, TP={m['input_tp']} dB, LRA={m['input_lra']} LU",
          file=sys.stderr, flush=True)
    return (
        f"loudnorm=I={target}:TP={tp}:LRA={lra}:linear=true"
        f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
        f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
        f":offset={m['target_offset']}"
    )


def duration(path: Path) -> float:
    return float(run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ]).strip())


def stage(n: int, text: str) -> None:
    print(f"[{n}/{STAGES}] {text}", file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Mix background music into a podcast WAV. Both inputs are loudness-"
            "normalized; BGM loops/trims to voice length + tail and fades out. "
            "Output: 16 kHz mono 16-bit WAV."
        )
    )
    ap.add_argument("-p", "--podcast", type=Path, required=True, help="Podcast (voice) WAV")
    ap.add_argument("-b", "--bgm", type=Path, required=True,
                    help="Background music file (any ffmpeg-readable format, e.g. WAV/Opus)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output WAV (default: <podcast-stem>_bgm.wav next to the podcast)")
    ap.add_argument("--voice-lufs", type=float, default=-16.0)
    ap.add_argument("--bgm-offset-db", type=float, default=22.0,
                    help="BGM is this many dB below the voice (default: 18)")
    ap.add_argument("--true-peak", type=float, default=-1.5)
    ap.add_argument("--lra", type=float, default=11.0)
    ap.add_argument("--tail", type=float, default=5.0,
                    help="Seconds BGM continues after the voice ends (default: 5)")
    ap.add_argument("--fade", type=float, default=3.0,
                    help="BGM fade-out length in seconds (default: 3)")
    return ap.parse_args()


def main() -> None:
    t0 = time.monotonic()
    a = parse_args()
    voice = a.podcast.expanduser().resolve()
    bgm = a.bgm.expanduser().resolve()
    for p, label in ((voice, "Podcast"), (bgm, "BGM")):
        if not p.is_file():
            fail(f"{label} file not found: {p}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        fail("ffmpeg/ffprobe not found in PATH.")
    if a.tail < 0 or a.fade < 0 or a.bgm_offset_db < 0:
        fail("--tail, --fade, --bgm-offset-db must be non-negative.")

    out = (a.output.expanduser().resolve() if a.output
           else voice.with_name(f"{voice.stem}_bgm.wav"))
    if out in (voice, bgm):
        fail("Output path must differ from input paths.")
    out.parent.mkdir(parents=True, exist_ok=True)

    voice_dur = duration(voice)
    bgm_dur = duration(bgm)
    total = voice_dur + a.tail
    fade_st = max(total - a.fade, 0.0)
    print(f"Podcast: {voice} ({voice_dur:.1f}s)\nBGM: {bgm} ({bgm_dur:.1f}s)\nOutput: {out}")
    print(f"Voice+tail: {total:.3f}s, fade-out starts at {fade_st:.3f}s", flush=True)

    stage(1, "Measuring podcast loudness")
    vf = loudnorm_filter(voice, a.voice_lufs, a.true_peak, a.lra, voice_dur, "podcast")
    stage(2, "Measuring BGM loudness")
    bf = loudnorm_filter(bgm, a.voice_lufs - a.bgm_offset_db, a.true_peak, a.lra, bgm_dur, "bgm")

    pre = out.with_name(f".{out.stem}.pre.wav")
    try:
        stage(3, "Mixing podcast + BGM")
        run_ffmpeg([
            "-y", "-v", "error", "-i", str(voice), "-i", str(bgm),
            "-filter_complex",
            f"[0:a]{FMT},{vf},aresample=16000,apad=pad_dur={a.tail}[v];"
            f"[1:a]{FMT},{bf},aresample=16000,aloop=loop=-1:size=2147483647,"
            f"atrim=0:{total:.6f},afade=t=out:st={fade_st:.6f}:d={a.fade}[b];"
            f"[v][b]amix=inputs=2:duration=longest:normalize=0,"
            f"atrim=0:{total:.6f},aformat=sample_fmts=s16:channel_layouts=mono[m]",
            "-map", "[m]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(pre),
        ], total, "mix")
        stage(4, "Measuring mix loudness")
        ff = loudnorm_filter(pre, a.voice_lufs, a.true_peak, a.lra, total, "mix")
        stage(5, "Normalizing final output")
        run_ffmpeg([
            "-y", "-v", "error", "-i", str(pre),
            "-af", f"{ff},{FMT}",
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out),
        ], total, "final")
    finally:
        pre.unlink(missing_ok=True)

    stage(6, "Verifying output")
    print(f"Completed: {out} ({duration(out):.3f}s, 16000 Hz, mono, 16-bit) "
          f"in {fmt_t(time.monotonic() - t0)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
