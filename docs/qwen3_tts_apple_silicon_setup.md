# Qwen3-TTS 1.7B Base（Apple Silicon / macOS）再現手順

Apple Silicon Mac上で、Qwen3-TTS 1.7B Base 8-bitをローカル実行し、参照音声から特定話者をクローンして、日本語の長文を文単位で生成・連結するまでの手順です。

- 対象: Apple Silicon（M1 / M2 / M3 / M4）Mac
- 実行: ローカル
- TTS: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- 出力: 24 kHz mono WAV
- 前提: 本人の声、または明示的に許諾された話者の参照音声のみを使用すること

## 0. ディレクトリと前提

以下ではプロジェクトを `~/shyme/qwen3-tts-apple-silicon` に置く。

```bash
export PROJECT="$HOME/shyme/qwen3-tts-apple-silicon"
mkdir -p "$HOME/shyme"
```

Xcode Command Line Toolsを入れる。既に導入済みなら何もしない。

```bash
xcode-select --install
```

Homebrewがなければ導入する。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Apple SiliconのHomebrewをPATHへ入れる。`
~/.zprofile` へ追記して、現在のシェルにも反映する。

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

必要なOSツールを導入する。

```bash
brew install python@3.12 git ffmpeg
```

確認する。

```bash
python3.12 --version
git --version
ffmpeg -version | head -n 1
```

## 1. リポジトリ取得と仮想環境

```bash
git clone https://github.com/kapi2800/qwen3-tts-apple-silicon.git "$PROJECT"
cd "$PROJECT"

/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt
python -m pip install -U mlx-audio soundfile huggingface_hub
```

依存関係を確認する。

```bash
python - <<'PY'
import importlib.metadata
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

print("mlx-audio:", importlib.metadata.version("mlx-audio"))
print("imports: OK")
PY
```

期待値:

```text
mlx-audio: 0.5.3
imports: OK
```

バージョンは将来変わってもよい。`imports: OK` が出ることを確認する。

## 2. Qwen3-TTSモデル取得

Hugging Face CLIの現行コマンドは `hf`。

```bash
cd "$PROJECT"
source .venv/bin/activate

hf --help
mkdir -p models

HF_HUB_DISABLE_XET=1 \
hf download \
  mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

取得内容を確認する。

```bash
find models/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  -maxdepth 1 \
  -type f \
  -print
```

少なくとも以下が見えること。

```text
model.safetensors
config.json
generation_config.json
tokenizer_config.json
vocab.json
merges.txt
preprocessor_config.json
```

認証エラーやgated modelの要求が出た場合だけログインする。

```bash
hf auth login
```

## 3. 参照音声を準備

ディレクトリを作る。

```bash
cd "$PROJECT"
mkdir -p wavs texts outputs
```

音声ファイルを `wavs/source.mp3` として置く。本人または明示的な許可を得た話者の音声だけを使用する。

24 kHz / mono / PCM WAVへ変換する。

```bash
ffmpeg -y \
  -i wavs/source.mp3 \
  -vn \
  -ac 1 \
  -ar 24000 \
  -c:a pcm_s16le \
  wavs/reference.wav
```

長さを確認する。

```bash
ffprobe -v error \
  -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  wavs/reference.wav
```

推奨:

- 単独話者
- BGM・効果音・他人の声がない
- ノイズ・反響が少ない
- 5〜30秒程度
- 生成対象に近い平常時の話し方

`wavs/reference.txt` を作る。内容は、`reference.wav` 内で実際に発話されている文字列に可能な限り正確に一致させる。

```bash
cat > wavs/reference.txt <<'EOF'
ここにreference.wavで実際に話している全文を記入する。
音声にない語句を追加せず、言い直しやフィラーも必要に応じて含める。
EOF
```

確認する。

```bash
cat wavs/reference.txt
```

## 4. 単文クローンの動作確認

```bash
cd "$PROJECT"
source .venv/bin/activate

python -m mlx_audio.tts.generate \
  --model models/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --text 'これは話者クローンの品質と、話者一貫性を確認する短いテストです。' \
  --lang_code Japanese \
  --ref_audio wavs/reference.wav \
  --ref_text "$(cat wavs/reference.txt)" \
  --output_path outputs \
  --file_prefix clone-test \
  --audio_format wav \
  --join_audio \
  --temperature 0.55 \
  --top_p 0.82 \
  --top_k 40 \
  --repetition_penalty 1.7 \
  --max_tokens 160
```

出力を確認し、再生する。

```bash
find outputs -type f -name 'clone-test*.wav' -print
afplay "$(find outputs -type f -name 'clone-test*.wav' | head -n 1)"
```

## 5. 長文用スクリプトを作成

プロジェクト直下に `longform_clone_tts.py` を置く。以下をそのまま実行する。

```bash
cd "$PROJECT"

cat > longform_clone_tts.py <<'PY'
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
PY

chmod +x longform_clone_tts.py
```

## 6. 長文生成

テキストを `texts/sample01.txt` として作る。UTF-8で保存する。空行は段落区切りとして扱われ、通常の文間より長い無音が入る。

```bash
cat > texts/sample01.txt <<'EOF'
これは一文目です。話者クローン条件を固定し、文単位で音声を生成しています。
二文目でも、声の高さ、音色、話者の印象が維持されるか確認します。

ここから第二段落です。段落間には通常より長い無音が入ります。
長文をローカルのApple Silicon Macで処理します。
EOF
```

生成する。

```bash
cd "$PROJECT"
source .venv/bin/activate

python longform_clone_tts.py texts/sample01.txt
```

出力:

```text
outputs/sample01.wav
```

再生する。

```bash
afplay outputs/sample01.wav
```

進捗例:

```text
Input: /Users/<USER>/shyme/qwen3-tts-apple-silicon/texts/sample01.txt
Sentences: 4
Temporary directory: /Users/<USER>/shyme/qwen3-tts-apple-silicon/.tmp_tts/sample01-xxxxxx
Output: /Users/<USER>/shyme/qwen3-tts-apple-silicon/outputs/sample01.wav
Loading model: /Users/<USER>/shyme/qwen3-tts-apple-silicon/models/Qwen3-TTS-12Hz-1.7B-Base-8bit
[0001/0004] これは一文目です。
[0002/0004] 話者クローン条件を固定し、文単位で音声を生成しています。
...
Completed: .../outputs/sample01.wav (24000 Hz, 4 sentence files generated)
Temporary sentence WAV files removed.
```

## 7. オプション

文間を450 ms、段落間を950 msにする。

```bash
python longform_clone_tts.py texts/sample01.txt \
  --pause-ms 450 \
  --paragraph-pause-ms 950
```

出力名を変える。

```bash
python longform_clone_tts.py texts/sample01.txt \
  --output outputs/sample01-narration.wav
```

文ごとの一時WAVを残して検査する。通常は指定しない。

```bash
python longform_clone_tts.py texts/sample01.txt --keep-temp
```

一時WAVの保存先:

```text
.tmp_tts/sample01-<ランダム文字列>/
```

通常は成功・失敗・Ctrl-Cのいずれでも一時WAVを削除する。`--keep-temp` 指定時だけ残す。

完成WAV全体を約10%遅くする。ピッチを変えずに話速だけ変える。

```bash
ffmpeg -y \
  -i outputs/sample01.wav \
  -filter:a 'atempo=0.90' \
  -ar 24000 \
  -ac 1 \
  outputs/sample01-slow.wav
```

約10%速くする場合:

```bash
ffmpeg -y \
  -i outputs/sample01.wav \
  -filter:a 'atempo=1.10' \
  -ar 24000 \
  -ac 1 \
  outputs/sample01-fast.wav
```

## 8. 日常運用

```bash
cd ~/shyme/qwen3-tts-apple-silicon
source .venv/bin/activate

# texts/<name>.txt を作成・編集後
python longform_clone_tts.py texts/<name>.txt

# 再生
# afplay outputs/<name>.wav
```

## 9. 主な確認・障害対応

仮想環境が有効か確認する。

```bash
which python
python --version
```

`which python` が以下のように `.venv/bin/python` を指すこと。

```text
.../qwen3-tts-apple-silicon/.venv/bin/python
```

`ModuleNotFoundError: No module named 'mlx_audio'` の場合:

```bash
source .venv/bin/activate
python -m pip install -U mlx-audio soundfile
```

モデルがない場合:

```bash
HF_HUB_DISABLE_XET=1 \
hf download \
  mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

出力が話者に似ない、または文ごとに安定しない場合:

```bash
# 参照テキストを再確認
cat wavs/reference.txt

# 必要に応じて文別WAVを残す
python longform_clone_tts.py texts/sample01.txt --keep-temp
```

最初に見直すべきものは、`wavs/reference.wav` の単独話者・明瞭さと、`wavs/reference.txt` が音声と完全に対応しているかどうか。
