# youtube-to-podcast

YouTube動画からポッドキャスト原稿を生成し、さらに話者クローンTTSで音声化するパイプライン。

## セットアップ

対象: Apple Silicon Mac（M1 / M2 / M3 / M4）。

```bash
git clone git@github.com:t-kawata/youtube-to-podcast.git ~/shyme/youtube-to-podcast
cd ~/shyme/youtube-to-podcast
./setup.sh
```

`~/shyme/youtube-to-podcast` に clone してから実行すること。他の場所に clone すると、パス前提が崩れるため `setup.sh` が起動直後に中断する。

`setup.sh` が自動で行うこと:

- Xcode Command Line Tools / Homebrew の確認・導入
- 必要パッケージ（`cmake` `ffmpeg` `git` `yt-dlp` `uv` `libsndfile` `curl` `jq` `python@3.12`）の導入
- `~/shyme/whisper.cpp` の clone、Metal有効ビルド、音声認識モデル・VADモデルの取得
- `~/shyme/qwen3-tts-apple-silicon` の clone、Python venv構築、Qwen3-TTSモデルの取得
- `scripts/youtube-to-podcast.sh` を `~/shyme/whisper.cpp/` へ、`scripts/longform_clone_tts.py` を `~/shyme/qwen3-tts-apple-silicon/` へ配置
- DeepFilterNet3の導入（`torch`/`torchaudio`のバージョン固定、`soundfile` backendパッチ）

各ステップは実行後に必ず検査され、通らない場合は原因と対処コマンドを表示して即座に中断する。中断した場合は表示された対処に従い、`./setup.sh` を再実行する（既に完了した工程は自動的にスキップされる）。

`setup.sh` が完了しても、以下は自動化されない領域なので手動確認が必要。

- `~/shyme/qwen3-tts-apple-silicon/wavs/reference.wav` / `reference.txt`: 本人、または明示的に許諾された話者の音声か、実際の発話内容と文字起こしが一致しているか
- OpenAI APIキー: `-k` オプションまたは環境変数として自前で用意する

セットアップ完了後の詳しい環境構築の背景は `docs/whisper-cpp-macos-setup-guide.md` と `docs/qwen3_tts_apple_silicon_setup.md` を参照。

## ディレクトリ構成

```
~/shyme/
├── whisper.cpp/                    # 文字起こし + youtube-to-podcast.sh 本体
│   └── youtube-to-podcast.sh
├── qwen3-tts-apple-silicon/        # 話者クローンTTS
│   ├── longform_clone_tts.py
│   ├── wavs/reference.wav          # 参照音声（本人・許諾済み話者のみ）
│   ├── wavs/reference.txt          # 参照音声の全文文字起こし
│   └── texts/                      # 原稿テキスト置き場
└── youtube-to-podcast/             # 本リポジトリ（scripts/docs/setup.sh）
```

原稿生成は `~/shyme/whisper.cpp/` で、音声合成は `~/shyme/qwen3-tts-apple-silicon/` で実行する。

## 全体の流れ

1. `youtube-to-podcast.sh` にYouTube URLを渡し、ポッドキャスト原稿（`.txt`）を作る。
2. 生成された原稿を `longform_clone_tts.py` に渡し、話者クローン音声（`.wav`）を作る。

```bash
cd ~/shyme/whisper.cpp
./youtube-to-podcast.sh -k "$OPENAI_API_KEY" -o ./out 'https://www.youtube.com/watch?v=XXXXXXXXXXX'
# -> ./out/<動画タイトル>-<動画ID>.txt が生成される

cp ./out/<動画タイトル>-<動画ID>.txt ~/shyme/qwen3-tts-apple-silicon/texts/

cd ~/shyme/qwen3-tts-apple-silicon
source .venv/bin/activate
./longform_clone_tts.py texts/<動画タイトル>-<動画ID>.txt
# -> outputs/<動画タイトル>-<動画ID>.wav が生成される

afplay outputs/<動画タイトル>-<動画ID>.wav
```

## 1. 原稿生成: youtube-to-podcast.sh

```bash
cd ~/shyme/whisper.cpp
./youtube-to-podcast.sh -k <OPENAI_API_KEY> [options] <YouTube_URL> [YouTube_URL...]
```

既存の文字起こしテキストから原稿だけ作りたい場合は、ダウンロード・デノイズ・文字起こし（手順1〜5）を全てスキップできる。

```bash
./youtube-to-podcast.sh -k <OPENAI_API_KEY> -f <TRANSCRIPT_FILE> [options]
```

### よく使うオプション

| オプション | 用途 | デフォルト |
|---|---|---|
| `-k, --api-key KEY` | OpenAI APIキー（原稿生成に必須） | なし（必須） |
| `-o, --output-dir DIR` | 出力先ディレクトリ | `./out` |
| `-n, --name NAME` | 出力ファイルの基幹名。複数URL指定時は `NAME-1, NAME-2, ...` | 動画タイトル+動画ID / `-f`のファイル名 |
| `-f, --transcript-file PATH` | 既存の文字起こしを直接使い、手順1〜5をスキップ | - |
| `--keep-audio` | 最終デノイズ済み音声も `<name>.wav` として保存 | 保存しない |
| `--keep-transcript` | 生の文字起こしも `<name>.transcript.txt` として保存 | 保存しない |
| `--no-transcribe` | 文字起こし・原稿生成をスキップし、デノイズ済み音声のみ出力 | - |
| `--language LANG` | 文字起こし言語コード（例: `ja`, `en`） | `ja` |
| `--openai-model MODEL` | 原稿生成に使うOpenAIモデル | `gpt-5.6-luna` |
| `-l, --min-length N` | 原稿が超えるべき文字数（Unicodeコードポイント数） | `6000` |
| `-t, --tries N` | 原稿生成の最大試行回数 | `3` |
| `-v, --verbose` | 外部コマンドの詳細ログを表示 | 非表示 |

複数URLを一括処理する例:

```bash
./youtube-to-podcast.sh -k "$OPENAI_API_KEY" -o ./out -n episode \
  'https://www.youtube.com/watch?v=AAAA' \
  'https://www.youtube.com/watch?v=BBBB'
# -> ./out/episode-1.txt, ./out/episode-2.txt
```

同名の最終成果物が既に `output-dir` に存在する場合、そのURL（またはファイル）の処理は丸ごとスキップされる。再生成したい場合は該当ファイルを削除するか `-n` で別名を指定する。

デフォルトでは原稿本文の `.txt` だけが残り、音声や生の文字起こしは自動削除される。中間ファイルを残したい場合は `--keep-audio` / `--keep-transcript` / `--keep-intermediate` を明示する。

## 2. 音声合成: longform_clone_tts.py

```bash
cd ~/shyme/qwen3-tts-apple-silicon
source .venv/bin/activate
./longform_clone_tts.py texts/<name>.txt
```

入力は手順1で作った原稿の `.txt`（UTF-8）。空行は段落区切りとして扱われ、通常の文間より長い無音が挿入される。出力は既定で `outputs/<name>.wav`（24kHz mono WAV）。

### よく使うオプション

| オプション | 用途 | デフォルト |
|---|---|---|
| `--output PATH` | 出力WAVのパスを明示指定 | `outputs/<入力ファイル名>.wav` |
| `--pause-ms N` | 通常の文末に挿入する無音（ミリ秒） | `360` |
| `--paragraph-pause-ms N` | 段落間に挿入する無音（ミリ秒） | `720` |
| `--reference-audio PATH` | 参照音声を差し替え | `wavs/reference.wav` |
| `--reference-text PATH` | 参照音声の文字起こしを差し替え | `wavs/reference.txt` |
| `--keep-temp` | 文単位の一時WAVを削除せず残す（検査用） | 残さない |

文間・段落間の間を調整する例:

```bash
./longform_clone_tts.py texts/sample01.txt --pause-ms 450 --paragraph-pause-ms 950
```

生成後に話速を変える（ピッチは変えない）:

```bash
ffmpeg -y -i outputs/sample01.wav -filter:a 'atempo=0.90' -ar 24000 -ac 1 outputs/sample01-slow.wav
```

## 日常運用の最短コマンド

```bash
# 原稿生成
cd ~/shyme/whisper.cpp
./youtube-to-podcast.sh -k "$OPENAI_API_KEY" 'https://www.youtube.com/watch?v=XXXX'

# 音声化
cp out/*.txt ~/shyme/qwen3-tts-apple-silicon/texts/
cd ~/shyme/qwen3-tts-apple-silicon
source .venv/bin/activate
./longform_clone_tts.py texts/<name>.txt
afplay outputs/<name>.wav
```

## 注意事項

- `wavs/reference.wav` は本人、または明示的に許諾された話者の音声のみを使用すること。
- 環境構築でエラーが出た場合は `docs/whisper-cpp-macos-setup-guide.md` と `docs/qwen3_tts_apple_silicon_setup.md` のトラブルシューティング節、または `./setup.sh` の中断メッセージを確認する。
