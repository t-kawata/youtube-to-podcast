#!/usr/bin/env bash
#
# setup.sh — youtube-to-podcast パイプライン一式を Apple Silicon Mac の ~/shyme に構築する。
#
# 前提: このファイルは youtube-to-podcast リポジトリのルートに置かれ、
#       ユーザーが以下を手動で実行済みであること。
#
#   git clone git@github.com:t-kawata/youtube-to-podcast.git ~/shyme/youtube-to-podcast
#   cd ~/shyme/youtube-to-podcast
#   ./setup.sh
#
# 設計方針:
#   - 各ステップは「実行 → 検査 → 不合格なら親切なメッセージを出して即中断」を徹底する。
#   - 警告して継続する、という曖昧な状態は作らない（安全側に倒す）。
#   - 既に条件を満たしているステップはスキップする（再実行に対して冪等）。
#
# 参照元:
#   docs/whisper-cpp-macos-setup-guide.md
#   docs/qwen3_tts_apple_silicon_setup.md
set -uo pipefail

# ============================================================================
# 表示・中断ユーティリティ
# ============================================================================
STEP_NO=0
STEP_NAME=""

step() {
  STEP_NO=$((STEP_NO + 1))
  STEP_NAME="$1"
  printf '\n[%02d] %s\n' "$STEP_NO" "$STEP_NAME"
}

info() { printf '     - %s\n' "$1"; }
pass() { printf '     [OK] %s\n' "$1"; }

die() {
  local reason="$1"
  local remedy="${2:-}"
  printf '\n' >&2
  printf '=============================================================\n' >&2
  printf ' 中断: ステップ %02d 「%s」で点検に失敗しました\n' "$STEP_NO" "$STEP_NAME" >&2
  printf '-------------------------------------------------------------\n' >&2
  printf ' 原因: %s\n' "$reason" >&2
  if [ -n "$remedy" ]; then
    printf ' 対処: %s\n' "$remedy" >&2
  fi
  printf '=============================================================\n' >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# 0. 事前点検: OS / アーキテクチャ / 自分自身のclone位置
# ============================================================================
step "事前点検: 実行環境の妥当性"

[ "$(uname -s)" = "Darwin" ] || die \
  "macOS 以外で実行されています（uname -s = $(uname -s)）。" \
  "このスクリプトは macOS 専用です。"
pass "OS: macOS"

[ "$(uname -m)" = "arm64" ] || die \
  "Apple Silicon 以外のCPUで実行されています（uname -m = $(uname -m)）。" \
  "Intel Mac は対象外です（Metal ビルドおよび mlx-audio が前提のため）。"
pass "CPU: Apple Silicon (arm64)"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_Y2P_DIR="$HOME/shyme/youtube-to-podcast"

[ "$SCRIPT_PATH" = "$EXPECTED_Y2P_DIR" ] || die \
  "setup.sh の実行位置が想定と異なります（現在地: $SCRIPT_PATH）。" \
  "youtube-to-podcast は必ず $EXPECTED_Y2P_DIR に clone してから実行してください。
       理由: youtube-to-podcast.sh の --whisper-dir デフォルト値が ~/shyme/whisper.cpp
       に固定されており、兄弟リポジトリも ~/shyme 直下を前提に配置するためです。
       例: rm -rf $SCRIPT_PATH && git clone git@github.com:t-kawata/youtube-to-podcast.git $EXPECTED_Y2P_DIR"
pass "clone位置: $EXPECTED_Y2P_DIR"

[ -f "$SCRIPT_PATH/scripts/youtube-to-podcast.sh" ] || die \
  "scripts/youtube-to-podcast.sh が見つかりません。" \
  "clone が不完全な可能性があります。リポジトリを clone し直してください。"
[ -f "$SCRIPT_PATH/scripts/longform_clone_tts.py" ] || die \
  "scripts/longform_clone_tts.py が見つかりません。" \
  "clone が不完全な可能性があります。リポジトリを clone し直してください。"
pass "必要スクリプトの存在確認"

SHYME="$HOME/shyme"
WHISPER_DIR="$SHYME/whisper.cpp"
QWEN_DIR="$SHYME/qwen3-tts-apple-silicon"
Y2P_DIR="$SCRIPT_PATH"
QWEN_REMOTE="https://github.com/kapi2800/qwen3-tts-apple-silicon.git"
WHISPER_REMOTE="https://github.com/ggml-org/whisper.cpp.git"
DF_TORCH_VER="2.8.0"
DF_TORCHAUDIO_VER="2.8.0"

# ============================================================================
# 1. Xcode Command Line Tools
# ============================================================================
step "Xcode Command Line Tools"

if xcode-select -p >/dev/null 2>&1; then
  pass "導入済み: $(xcode-select -p)"
else
  info "未導入のためインストーラを起動します（GUIでの同意が必要です）"
  xcode-select --install || true
  die \
    "Xcode Command Line Tools が未導入です。" \
    "起動したインストーラでインストールを完了させてから、再度 ./setup.sh を実行してください。"
fi

# ============================================================================
# 2. Homebrew
# ============================================================================
step "Homebrew"

if ! require_cmd brew; then
  info "未導入のため導入します"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    || die "Homebrew のインストールに失敗しました。" "ネットワーク接続とディスク空き容量を確認してください。"
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> "$HOME/.zprofile"
fi
eval "$(/opt/homebrew/bin/brew shellenv)"

require_cmd brew || die \
  "Homebrew のインストール後も brew コマンドが見つかりません。" \
  "新しいターミナルを開くか 'eval \"\$(/opt/homebrew/bin/brew shellenv)\"' を実行してから再試行してください。"
pass "brew: $(brew --version | head -n1)"

# ============================================================================
# 3. Homebrew パッケージ
# ============================================================================
step "Homebrew パッケージ導入"

BREW_PKGS=(python@3.12 git ffmpeg cmake yt-dlp uv libsndfile curl jq)
for pkg in "${BREW_PKGS[@]}"; do
  if brew list --formula --versions "$pkg" >/dev/null 2>&1; then
    info "skip (導入済み): $pkg"
  else
    info "installing: $pkg"
    brew install "$pkg" || die \
      "パッケージ '$pkg' の導入に失敗しました。" \
      "'brew doctor' で環境診断してから 'brew install $pkg' を単体で再試行してください。"
  fi
done

for bin in git ffmpeg ffprobe cmake yt-dlp uv curl jq; do
  require_cmd "$bin" || die \
    "'$bin' コマンドが PATH 上に見つかりません。" \
    "'brew link $bin' または新しいターミナルでのPATH再読込を試してください。"
done
PY312="$(brew --prefix python@3.12)/bin/python3.12"
[ -x "$PY312" ] || die \
  "python3.12 バイナリが見つかりません ($PY312)。" \
  "'brew reinstall python@3.12' を試してください。"
pass "全パッケージのコマンド疎通を確認"

mkdir -p "$SHYME"

# ============================================================================
# 4. whisper.cpp: clone
# ============================================================================
step "whisper.cpp の取得"

if [ -d "$WHISPER_DIR/.git" ]; then
  info "既存: $WHISPER_DIR（pull更新）"
  git -C "$WHISPER_DIR" pull --ff-only || die \
    "whisper.cpp の pull に失敗しました（ローカル変更等の可能性）。" \
    "cd $WHISPER_DIR && git status で状態を確認し、不要な変更なら 'git reset --hard' 後に再実行してください。"
else
  git clone "$WHISPER_REMOTE" "$WHISPER_DIR" || die \
    "whisper.cpp の clone に失敗しました。" "ネットワーク接続を確認してください。"
fi
[ -d "$WHISPER_DIR/.git" ] || die "whisper.cpp の clone 後も .git が存在しません。" "手動で $WHISPER_DIR を削除して再実行してください。"
pass "whisper.cpp: $WHISPER_DIR"

# ============================================================================
# 5. whisper.cpp: ビルド (Metal)
# ============================================================================
step "whisper.cpp ビルド (Metal)"

if [ -x "$WHISPER_DIR/build/bin/whisper-cli" ]; then
  info "既にビルド済み"
else
  cmake -S "$WHISPER_DIR" -B "$WHISPER_DIR/build" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
    || die "cmake の構成に失敗しました。" "Xcode Command Line Tools の再導入を確認してください。"
  cmake --build "$WHISPER_DIR/build" -j"$(sysctl -n hw.ncpu)" \
    || die "whisper.cpp のビルドに失敗しました。" "'$WHISPER_DIR/build' を削除して再実行してください。"
fi

[ -x "$WHISPER_DIR/build/bin/whisper-cli" ] || die \
  "ビルド後も whisper-cli バイナリが見つかりません。" \
  "$WHISPER_DIR/build を削除して setup.sh を再実行してください。"
"$WHISPER_DIR/build/bin/whisper-cli" --help >/dev/null 2>&1 || die \
  "whisper-cli --help が正常終了しません。" "バイナリが壊れている可能性があります。buildディレクトリを削除して再実行してください。"
pass "whisper-cli 動作確認"

# ============================================================================
# 6. whisper.cpp: モデル取得
# ============================================================================
step "whisper.cpp モデル取得"

WHISPER_MODEL="$WHISPER_DIR/models/ggml-large-v3-turbo.bin"
if [ -s "$WHISPER_MODEL" ]; then
  info "skip: ggml-large-v3-turbo.bin 既存"
else
  ( cd "$WHISPER_DIR" && ./models/download-ggml-model.sh large-v3-turbo ) || die \
    "音声認識モデルのダウンロードに失敗しました。" "ネットワーク接続を確認し再実行してください。"
fi
[ -s "$WHISPER_MODEL" ] || die "モデルファイルが空、または存在しません。" "$WHISPER_MODEL を削除して再実行してください。"
pass "認識モデル: $WHISPER_MODEL"

if ls "$WHISPER_DIR"/models/*silero-v5.1.2* >/dev/null 2>&1; then
  info "skip: VADモデル既存"
else
  ( cd "$WHISPER_DIR" && ./models/download-vad-model.sh silero-v5.1.2 ) || die \
    "VADモデルのダウンロードに失敗しました。" "ネットワーク接続を確認し再実行してください。"
fi
ls "$WHISPER_DIR"/models/*silero-v5.1.2* >/dev/null 2>&1 || die \
  "VADモデルが見つかりません。" "download-vad-model.sh を手動実行して原因を確認してください。"
pass "VADモデル確認済み"

# ============================================================================
# 7. qwen3-tts-apple-silicon: clone
# ============================================================================
step "qwen3-tts-apple-silicon の取得"

if [ -d "$QWEN_DIR/.git" ]; then
  info "既存: $QWEN_DIR（pull更新）"
  git -C "$QWEN_DIR" pull --ff-only || die \
    "qwen3-tts-apple-silicon の pull に失敗しました。" \
    "cd $QWEN_DIR && git status を確認してください。"
else
  git clone "$QWEN_REMOTE" "$QWEN_DIR" || die \
    "qwen3-tts-apple-silicon の clone に失敗しました。" "ネットワーク接続を確認してください。"
fi
[ -d "$QWEN_DIR/.git" ] || die "clone後も .git が存在しません。" "$QWEN_DIR を削除して再実行してください。"
mkdir -p "$QWEN_DIR/wavs" "$QWEN_DIR/texts" "$QWEN_DIR/outputs" "$QWEN_DIR/models"
pass "qwen3-tts-apple-silicon: $QWEN_DIR"

# ============================================================================
# 8. qwen3-tts-apple-silicon: venv / 依存関係
# ============================================================================
step "Python venv 構築"

if [ ! -x "$QWEN_DIR/.venv/bin/python" ]; then
  "$PY312" -m venv "$QWEN_DIR/.venv" || die "venv の作成に失敗しました。" "python@3.12 の再導入を試してください。"
fi
VENV_PY_VER="$("$QWEN_DIR/.venv/bin/python" --version 2>&1)"
case "$VENV_PY_VER" in
  "Python 3.12."*) : ;;
  *) die "venv の Python バージョンが想定外です（$VENV_PY_VER）。" "$QWEN_DIR/.venv を削除して再実行してください。" ;;
esac
pass "venv: $VENV_PY_VER"

step "Python 依存関係インストール"

# shellcheck disable=SC1091
source "$QWEN_DIR/.venv/bin/activate"
python -m pip install -q -U pip wheel setuptools || die "pip自体の更新に失敗しました。" "ネットワーク接続を確認してください。"
if [ -f "$QWEN_DIR/requirements.txt" ]; then
  python -m pip install -q -r "$QWEN_DIR/requirements.txt" || die \
    "requirements.txt のインストールに失敗しました。" "エラーログを確認し個別に原因パッケージを特定してください。"
fi
python -m pip install -q -U mlx-audio soundfile huggingface_hub || die \
  "mlx-audio 等のインストールに失敗しました。" "ネットワーク接続を確認してください。"

python - <<'PY' || die "mlx_audio のインポートに失敗しました。" "依存関係の再インストールを試してください。"
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio
PY
deactivate
pass "依存関係のインポート確認"

# ============================================================================
# 9. Qwen3-TTS モデル取得
# ============================================================================
step "Qwen3-TTS モデル取得"

QWEN_MODEL_DIR="$QWEN_DIR/models/Qwen3-TTS-12Hz-1.7B-Base-8bit"
if [ -s "$QWEN_MODEL_DIR/model.safetensors" ]; then
  info "skip: モデル既存"
else
  # shellcheck disable=SC1091
  source "$QWEN_DIR/.venv/bin/activate"
  HF_HUB_DISABLE_XET=1 hf download \
    mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit \
    --local-dir "$QWEN_MODEL_DIR"
  hf_status=$?
  deactivate
  if [ "$hf_status" -ne 0 ]; then
    die \
      "Qwen3-TTS モデルのダウンロードに失敗しました。" \
      "gated model 認証が必要な場合があります。次を実行してください:
       cd $QWEN_DIR && source .venv/bin/activate && hf auth login
       その後 ./setup.sh を再実行してください。"
  fi
fi
[ -s "$QWEN_MODEL_DIR/model.safetensors" ] || die \
  "モデルファイル model.safetensors が存在しません。" "$QWEN_MODEL_DIR を削除して再実行してください。"
pass "モデル確認: $QWEN_MODEL_DIR"

# ============================================================================
# 10. スクリプト配置
# ============================================================================
step "パイプラインスクリプトの配置"

cp -f "$Y2P_DIR/scripts/youtube-to-podcast.sh" "$WHISPER_DIR/youtube-to-podcast.sh" || die \
  "youtube-to-podcast.sh のコピーに失敗しました。" "$WHISPER_DIR への書き込み権限を確認してください。"
chmod +x "$WHISPER_DIR/youtube-to-podcast.sh"
"$WHISPER_DIR/youtube-to-podcast.sh" --help >/dev/null 2>&1 || die \
  "配置後の youtube-to-podcast.sh --help が失敗しました。" "curl/jq の導入状況を確認してください。"
pass "配置: $WHISPER_DIR/youtube-to-podcast.sh"

cp -f "$Y2P_DIR/scripts/longform_clone_tts.py" "$QWEN_DIR/longform_clone_tts.py" || die \
  "longform_clone_tts.py のコピーに失敗しました。" "$QWEN_DIR への書き込み権限を確認してください。"
chmod +x "$QWEN_DIR/longform_clone_tts.py"
# shellcheck disable=SC1091
source "$QWEN_DIR/.venv/bin/activate"
python "$QWEN_DIR/longform_clone_tts.py" --help >/dev/null 2>&1
lct_status=$?
deactivate
[ "$lct_status" -eq 0 ] || die \
  "配置後の longform_clone_tts.py --help が失敗しました。" "venv の依存関係インストールをやり直してください。"
pass "配置: $QWEN_DIR/longform_clone_tts.py"

# ============================================================================
# 11. 参照音声の配置
# ============================================================================
step "参照音声（reference.wav / reference.txt）の配置確認"

if [ -f "$Y2P_DIR/wavs/reference.wav" ] && [ ! -f "$QWEN_DIR/wavs/reference.wav" ]; then
  cp -f "$Y2P_DIR/wavs/reference.wav" "$QWEN_DIR/wavs/reference.wav"
  info "コピー: reference.wav"
fi
if [ -f "$Y2P_DIR/wavs/reference.txt" ] && [ ! -f "$QWEN_DIR/wavs/reference.txt" ]; then
  cp -f "$Y2P_DIR/wavs/reference.txt" "$QWEN_DIR/wavs/reference.txt"
  info "コピー: reference.txt"
fi

if [ ! -s "$QWEN_DIR/wavs/reference.wav" ] || [ ! -s "$QWEN_DIR/wavs/reference.txt" ]; then
  die \
    "参照音声 ($QWEN_DIR/wavs/reference.wav) または文字起こし ($QWEN_DIR/wavs/reference.txt) が未配置です。" \
    "本人または明示的に許諾された話者の5〜30秒程度の単独話者音声を
       $QWEN_DIR/wavs/reference.wav (24kHz/mono/PCM16) として、
       その発話内容全文を $QWEN_DIR/wavs/reference.txt として配置してから、
       ./setup.sh を再実行してください。"
fi
pass "参照音声・文字起こしの存在確認"

# ============================================================================
# 12. DeepFilterNet3
# ============================================================================
step "DeepFilterNet3 導入"

DF_TOOL_DIR="$HOME/.local/share/uv/tools/deepfilternet"
if [ -x "$DF_TOOL_DIR/bin/deepFilter" ]; then
  info "skip: 導入済み"
else
  uv tool install deepfilternet \
    --with "torch==${DF_TORCH_VER}" \
    --with "torchaudio==${DF_TORCHAUDIO_VER}" \
    --with 'soundfile>=0.12.1' \
    || die "deepfilternet の uv tool install に失敗しました。" "ネットワーク接続を確認し再実行してください。"
fi
[ -x "$DF_TOOL_DIR/bin/deepFilter" ] || die \
  "deepFilter バイナリが見つかりません ($DF_TOOL_DIR/bin/deepFilter)。" \
  "'uv tool uninstall deepfilternet' 後に setup.sh を再実行してください。"
pass "deepFilter バイナリ確認"

step "DeepFilterNet 音声backend確認"

DF_PY="$DF_TOOL_DIR/bin/python"
[ -x "$DF_PY" ] || die "$DF_PY が見つかりません。" "deepfilternet の再インストールが必要です。"

"$DF_PY" - <<'PY'
import sys
try:
    import soundfile
    import torchaudio
except Exception as e:
    print(f"import error: {e}", file=sys.stderr)
    sys.exit(1)
if "soundfile" not in torchaudio.list_audio_backends():
    print("soundfile backend not registered", file=sys.stderr)
    sys.exit(1)
PY
[ $? -eq 0 ] || die \
  "torchaudio の soundfile backend が認識されていません。" \
  "libsndfile が導入済みか確認してください（brew list libsndfile）。改善しない場合は
       'uv tool uninstall deepfilternet' 後に setup.sh を再実行してください。"
pass "soundfile backend 確認"

step "DeepFilterNet io.py パッチ確認/適用"

PYVER_DIR="$(ls "$DF_TOOL_DIR/lib" 2>/dev/null | head -n1)"
[ -n "$PYVER_DIR" ] || die "$DF_TOOL_DIR/lib 内にPython版ディレクトリが見つかりません。" "deepfilternet の再インストールが必要です。"
DF_IO="$DF_TOOL_DIR/lib/$PYVER_DIR/site-packages/df/io.py"
[ -f "$DF_IO" ] || die "df/io.py が見つかりません ($DF_IO)。" "DeepFilterNetのバージョン差異の可能性があります。手動確認してください。"

if grep -q 'backend="soundfile"' "$DF_IO"; then
  info "skip: パッチ適用済み"
else
  cp -p "$DF_IO" "$DF_IO.bak.$(date +%Y%m%d-%H%M%S)"
  "$DF_PY" - "$DF_IO" <<'PY'
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
fi
grep -q 'backend="soundfile"' "$DF_IO" || die \
  "df/io.py へのパッチ適用後も backend=\"soundfile\" が見つかりません。" \
  "対象文字列が想定と異なる可能性があります。$DF_IO を手動確認してください。"
pass "df/io.py パッチ確認"

step "DeepFilterNet 実動作確認"

DF_TEST_OUT="$(mktemp -d)"
DF_TEST_IN="$QWEN_DIR/wavs/reference.wav"
"$DF_TOOL_DIR/bin/deepFilter" \
  --model-base-dir DeepFilterNet3 \
  --output-dir "$DF_TEST_OUT" \
  --no-suffix \
  "$DF_TEST_IN" >/dev/null 2>&1
df_status=$?
DF_TEST_RESULT="$DF_TEST_OUT/$(basename "$DF_TEST_IN")"
rm -rf "$DF_TEST_OUT"
[ "$df_status" -eq 0 ] || die \
  "deepFilter の実行に失敗しました（reference.wavでのテスト）。" \
  "上記トラブルシューティングを一通り再確認してください。初回実行時はモデルのダウンロードで時間がかかる場合があります。"
pass "deepFilter 実動作確認（reference.wavでデノイズ成功）"

# ============================================================================
# 13. 最終疎通確認
# ============================================================================
step "最終疎通確認"

"$WHISPER_DIR/build/bin/whisper-cli" --help >/dev/null 2>&1 || die "whisper-cli の最終確認に失敗しました。" "上のビルド手順を再確認してください。"
pass "whisper-cli: OK"

"$WHISPER_DIR/youtube-to-podcast.sh" --help >/dev/null 2>&1 || die "youtube-to-podcast.sh の最終確認に失敗しました。" "スクリプト配置手順を再確認してください。"
pass "youtube-to-podcast.sh: OK"

# shellcheck disable=SC1091
source "$QWEN_DIR/.venv/bin/activate"
python "$QWEN_DIR/longform_clone_tts.py" --help >/dev/null 2>&1
lct_final_status=$?
deactivate
[ "$lct_final_status" -eq 0 ] || die "longform_clone_tts.py の最終確認に失敗しました。" "venv構築手順を再確認してください。"
pass "longform_clone_tts.py: OK"

"$DF_TOOL_DIR/bin/deepFilter" --help >/dev/null 2>&1 || die "deepFilter の最終確認に失敗しました。" "DeepFilterNet導入手順を再確認してください。"
pass "deepFilter: OK"

cat <<EOF

=============================================================
 すべての点検を通過しました。環境構築は完了です。
=============================================================

実行例:
  cd $WHISPER_DIR
  ./youtube-to-podcast.sh -k \$OPENAI_API_KEY -o ./out 'https://www.youtube.com/watch?v=XXXX'

  cd $QWEN_DIR
  source .venv/bin/activate
  ./longform_clone_tts.py ./texts/<name>.txt
EOF
