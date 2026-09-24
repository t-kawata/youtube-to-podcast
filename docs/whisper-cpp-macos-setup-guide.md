# Mac 環境における whisper.cpp 音声処理パイプライン構築手順

対象環境: macOS（Apple Silicon、Metal 対応）
最終目的: YouTube 動画から 16 kHz / mono / 16-bit PCM WAV を取得し、DeepFilterNet3 でデノイズし、whisper.cpp（VAD付き）で文字起こしし、OpenAI API でポッドキャスト原稿を生成する一連のパイプラインを動作させる。

この文書は、実際に発生したエラーとその解決を反映した手順のみを記載する。理論上の代替手段は含まない。

---

## 0. 前提

- Homebrew が導入済みであること。
- Xcode Command Line Tools が導入済みであること（`cmake` のビルドに必要）。

```bash
xcode-select --install
```

すでに導入済みの場合はエラーが出るが無視してよい。

---

## 1. 基盤コマンドラインツールの導入

```bash
brew install cmake ffmpeg git yt-dlp uv libsndfile
```

各パッケージの用途は以下。

| パッケージ | 用途 |
|---|---|
| `cmake` | whisper.cpp のビルド |
| `ffmpeg` | 音声フォーマット変換・リサンプリング |
| `git` | whisper.cpp のクローン |
| `yt-dlp` | YouTube からの音声取得 |
| `uv` | Python ツール（DeepFilterNet）の隔離環境管理 |
| `libsndfile` | `soundfile` パッケージが依存するネイティブライブラリ |

`libsndfile` は、後述する DeepFilterNet の音声入出力エラーを解決するために必須である。省略すると、`soundfile` パッケージのインストール自体は成功するが、実行時に音声デコード用の backend が見つからず失敗する。

---

## 2. whisper.cpp の取得とビルド

```bash
mkdir -p ~/shyme
cd ~/shyme
git clone https://github.com/ggml-org/whisper.cpp.git
cd ~/shyme/whisper.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build -j"$(sysctl -n hw.ncpu)"
```

`-DGGML_METAL=ON` により、Apple Silicon の GPU（Metal）を推論に使う。ビルド完了後、以下が存在することを確認する。

```bash
ls ~/shyme/whisper.cpp/build/bin/whisper-cli
```

---

## 3. モデルの取得

### 3.1 音声認識モデル

```bash
cd ~/shyme/whisper.cpp
./models/download-ggml-model.sh large-v3-turbo
```

これにより `~/shyme/whisper.cpp/models/ggml-large-v3-turbo.bin` が生成される。

### 3.2 VAD（発話区間検出）モデル

```bash
./models/download-vad-model.sh silero-v5.1.2
```

これにより `~/shyme/whisper.cpp/models/` 内に `*silero-v5.1.2*.bin` に一致するファイルが生成される。正確なファイル名はビルドにより変動するため、パイプライン側では固定名を仮定せず、グロブ検索で自動検出する設計にしている（後述のスクリプト参照）。

### 3.3 動作確認

```bash
~/shyme/whisper.cpp/build/bin/whisper-cli --help
```

ヘルプが表示されれば、ビルドとモデル配置は完了している。

---

## 4. DeepFilterNet3 の導入

### 4.1 初回インストール（この形だけでは不完全）

```bash
uv tool install deepfilternet
```

この状態で `deepFilter --help` を実行すると、次のエラーが発生する。

```text
ModuleNotFoundError: No module named 'torch'
```

**原因**: DeepFilterNet の CLI は PyTorch を必須依存として宣言しておらず、実行時に `import torch` する。`uv tool install` は宣言された依存だけを解決するため、torch が入らない。

### 4.2 torch / torchaudio を追加

```bash
uv tool install --reinstall deepfilternet \
  --with torch \
  --with torchaudio
```

この状態で `deepFilter --help` を実行すると、次のエラーが発生する。

```text
ModuleNotFoundError: No module named 'torchaudio.backend'
```

**原因**: 最新の `torchaudio`（2.9系相当のAPI変更を先取りした版、または2.8以降の一部リリース）では `torchaudio.backend.common.AudioMetaData` の import 経路が変更・削除されている。DeepFilterNet 0.5.6 はこの古い import 経路にまだ依存しており、追随していない。公式 issue でも、回避策として `torchaudio` を 2.9 未満に固定することが案内されている。

### 4.3 torch / torchaudio をバージョン固定して再構築

```bash
uv tool uninstall deepfilternet

uv tool install deepfilternet \
  --with 'torch==2.8.0' \
  --with 'torchaudio==2.8.0'
```

`torch` と `torchaudio` は必ず同一系列のリリースで揺れなく組み合わせる。異なる組み合わせは公式にサポートされていない。

この時点で `deepFilter --help` は正常に動作するようになる（ヘルプ表示のみなら音声デコードを伴わないため）。

### 4.4 実行時エラー: 音声デコード backend が見つからない

実際に音声ファイルを渡すと、次のエラーが発生する。

```text
RuntimeError: Couldn't find appropriate backend to handle uri ... and format None.
```

**原因**: TorchAudio 2.8 は音声の読み込みに FFmpeg / SoX / SoundFile のいずれかの backend を必要とするが、`uv tool install` で構築した隔離環境にはこれらの Python 側 backend パッケージが入っていない。

### 4.5 soundfile backend を追加

```bash
uv tool install --reinstall deepfilternet \
  --with 'torch==2.8.0' \
  --with 'torchaudio==2.8.0' \
  --with 'soundfile>=0.12.1'
```

このために、手順1で `libsndfile` を Homebrew から導入済みである必要がある。`soundfile` パッケージ自体は Python 側のバインディングであり、実体のデコード処理は `libsndfile`（Cライブラリ）が担う。

導入後、以下で backend が認識されているか確認する。

```bash
DF_PY="$HOME/.local/share/uv/tools/deepfilternet/bin/python"

"$DF_PY" - <<'PY'
import soundfile
import torchaudio

print("soundfile:", soundfile.__version__)
print("torchaudio:", torchaudio.__version__)
print("backends:", torchaudio.list_audio_backends())
PY
```

期待される出力（一部警告は無視してよい）。

```text
soundfile: 0.14.0
torchaudio: 2.8.0
backends: ['soundfile']
```

`backends` に `soundfile` が含まれていれば、Python レベルでは backend が使える状態になっている。

### 4.6 それでも失敗する場合: DeepFilterNet 側のコード修正

`soundfile` backend が存在しても、DeepFilterNet 0.5.6 の `df/io.py` は `torchaudio.info` / `torchaudio.load` を **backend 無指定**で呼ぶため、TorchAudio 側の自動選択に失敗し、依然としてエラーになる場合がある。この場合、DeepFilterNet 側のソースを直接パッチする。

```bash
DF_SITE="$HOME/.local/share/uv/tools/deepfilternet/lib/python3.12/site-packages"
DF_IO="$DF_SITE/df/io.py"

cp -p "$DF_IO" "$DF_IO.bak.$(date +%Y%m%d-%H%M%S)"

DF_PY="$HOME/.local/share/uv/tools/deepfilternet/bin/python"

"$DF_PY" - <<'PY' "$DF_IO"
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()

replacements = {
    "ta.info(file, **ikwargs)": 'ta.info(file, backend="soundfile", **ikwargs)',
    "ta.load(file, **ikwargs)": 'ta.load(file, backend="soundfile", **ikwargs)',
    "ta.save(file, audio, sr, **okwargs)": 'ta.save(file, audio, sr, backend="soundfile", **okwargs)',
}

for old, new in replacements.items():
    if old in s:
        s = s.replace(old, new)

p.write_text(s)
PY
```

**重要な注意**: `uv tool install --reinstall` や `uv tool uninstall` を再実行すると、このパッチは失われる。DeepFilterNet を再構築した場合は、このパッチも再適用が必要である。

Python バージョンが `3.12` 以外の場合は、`DF_SITE` 内のパスを実際のバージョンに合わせて変更する。

```bash
ls "$HOME/.local/share/uv/tools/deepfilternet/lib/"
```

### 4.7 最終確認

```bash
deepFilter --help
```

エラーなくヘルプが表示され、かつ実際の WAV ファイルに対して以下のように実行できることを確認する。

```bash
deepFilter \
  --model-base-dir DeepFilterNet3 \
  --output-dir /tmp/df-test-out \
  --no-suffix \
  /path/to/test-input.wav
```

`/tmp/df-test-out` にデノイズ済みの WAV が出力されれば、DeepFilterNet3 の実行環境は完成している。

初回実行時は、モデルが `~/Library/Caches/DeepFilterNet/DeepFilterNet3/` へ自動的にダウンロード・キャッシュされる。

---

## 5. 依存関係の全体像（最終状態）

| コンポーネント | 導入方法 | バージョン固定の理由 |
|---|---|---|
| `cmake`, `ffmpeg`, `git`, `yt-dlp`, `uv`, `libsndfile` | Homebrew | - |
| whisper.cpp本体 | GitHubクローン + cmakeビルド（Metal有効） | - |
| `ggml-large-v3-turbo.bin` | `download-ggml-model.sh` | - |
| VADモデル（silero-v5.1.2） | `download-vad-model.sh` | - |
| `torch` | uv tool の `--with` | `torchaudio` と同一系列に固定 |
| `torchaudio` | uv tool の `--with` | `2.9` 以降は `AudioMetaData` の import 経路が削除され、DeepFilterNet 0.5.6 が追随していないため `2.8.0` に固定 |
| `soundfile` / `libsndfile` | uv tool の `--with` + Homebrew | TorchAudio の音声デコード backend として必要 |
| DeepFilterNet の `df/io.py` パッチ | 手動編集 | backend 自動選択がTorchAudio 2.8で失敗するため明示指定が必要 |

---

## 6. トラブルシューティング早見表

実際に遭遇したエラーと、その対応を再掲する。

| エラーメッセージ | 原因 | 対応 |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | DeepFilterNet CLI が torch を宣言依存に含めていない | `uv tool install ... --with torch --with torchaudio` |
| `ModuleNotFoundError: No module named 'torchaudio.backend'` | torchaudio が新しいAPI体系に移行し、旧import経路が削除された | `torchaudio==2.8.0` に固定して再構築 |
| `RuntimeError: Couldn't find appropriate backend to handle uri ...` | 音声デコード用backend（soundfile等）が環境に存在しない | `libsndfile`（Homebrew）+ `soundfile`（uv tool の `--with`）を追加 |
| 上記を解消しても同エラーが再発する | DeepFilterNet の `df/io.py` が backend 無指定で `torchaudio.info/load` を呼んでいる | `df/io.py` を直接パッチし、`backend="soundfile"` を明示 |
| `fatal: not a git repository` （DeepFilterNet実行時の警告） | DeepFilterNet がバージョン情報取得のためGitリポジトリ判定を試みるが、実行時のカレントディレクトリがGit管理下にない | 実害なし。無視してよい |
| `UserWarning: 'pin_memory' ... not supported on MPS` | PyTorch DataLoaderの最適化フラグがMPSでは効かないという情報警告 | 実害なし。無視してよい |

---

## 7. パイプライン全体の起動確認

以上の環境構築が完了していれば、以下のコマンドで一連の処理（ダウンロード→デノイズ→リサンプル→検証→文字起こし→原稿生成）が動作する。

```bash
chmod +x youtube-to-podcast.sh

./youtube-to-podcast.sh \
  -k "$OPENAI_API_KEY" \
  -o ./out \
  -n <出力ファイルの基幹名> \
  'https://www.youtube.com/watch?v=<動画ID>'
```

スクリプト自体は本ドキュメントの対象外だが、依存する外部コマンドは次の通りであり、本手順で全て導入済みである。

```text
yt-dlp, ffmpeg, ffprobe, deepFilter, whisper-cli, curl, jq
```

`jq` と `curl` は macOS 標準搭載であることが多いが、古い場合は以下で明示的に導入しておくと安定する。

```bash
brew install curl jq
```

---

## 8. 再構築時の注意（環境が壊れた場合）

DeepFilterNet の環境を作り直す必要が生じた場合、必ず以下の順序を踏むこと。単に `uv tool install deepfilternet` だけを実行すると、4.1〜4.6 のエラーを再び順番に踏むことになる。

```bash
uv tool uninstall deepfilternet

uv tool install deepfilternet \
  --with 'torch==2.8.0' \
  --with 'torchaudio==2.8.0' \
  --with 'soundfile>=0.12.1'

# 4.6 のパッチを再適用する（再構築後は必ず消えている）
```

whisper.cpp 側は、モデルファイルとビルド済みバイナリが存在していれば再構築不要である。ビルドツールチェーン（Xcode Command Line Tools）やモデルファイルを誤って削除した場合のみ、手順2・3を再実行する。
