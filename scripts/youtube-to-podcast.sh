#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  youtube-to-podcast.sh -k <OPENAI_API_KEY> [options] <YouTube_URL> [YouTube_URL...]
  youtube-to-podcast.sh -k <OPENAI_API_KEY> -f <TRANSCRIPT_FILE> [options]

Required:
  -k, --api-key KEY           OpenAI API キー（原稿生成を行う場合は常に必須）

Options:
  -o, --output-dir DIR         出力先ディレクトリ
                              Default: ./out
  -n, --name NAME               出力ファイルの基幹名を明示指定する
                              未指定時は動画タイトル+動画ID、または -f のファイル名から自動生成
                              複数URL指定時は NAME-1, NAME-2, ... と連番になる
  -f, --transcript-file PATH  既存の文字起こしテキストファイルを直接指定する。
                              指定時はダウンロード・デノイズ・リサンプル・検証・文字起こし
                              （手順1〜5）をすべてスキップし、このファイルの内容を
                              そのまま原稿生成（手順6）の入力として使う。
                              URL指定は不要かつ無視される。
  -m, --model MODEL             DeepFilterNet モデル名またはモデルディレクトリ
                              Default: DeepFilterNet3
      --pf                    DeepFilterNet の post-filter を有効化
      --keep-audio             最終デノイズ済み音声を $OUTPUT_DIR/<name>.wav としても保存する
      --keep-transcript         生の文字起こしを $OUTPUT_DIR/<name>.transcript.txt としても保存する
      --keep-intermediate       作業用の一時ディレクトリ全体（ログ含む）を残す
      --no-transcribe           文字起こし・原稿生成をスキップする
                              （この場合、$OUTPUT_DIR/<name>.wav が最終成果物になる）

  --whisper-dir DIR             whisper.cpp のリポジトリ/ビルドディレクトリ
                              Default: ~/shyme/whisper.cpp
  --whisper-model PATH         whisper-cli に渡すモデルファイル (.bin)
                              Default: <whisper-dir>/models/ggml-large-v3-turbo.bin
  --vad-model PATH              VAD モデルファイル (.bin)
                              Default: <whisper-dir>/models 内の *silero-v5.1.2* を自動検出
  --language LANG               文字起こし言語コード（例: ja, en）。Default: ja
  --whisper-threads N           whisper-cli のスレッド数。Default: CPU 論理数

  --openai-model MODEL          原稿生成に使う OpenAI モデル名
                              Default: gpt-5.6-luna
  -l, --min-length N             原稿が超える必要がある文字数（Unicodeコードポイント数）
                              Default: 6000
  -t, --tries N                  原稿生成を試す最大回数
                              Default: 3

  -v, --verbose                  yt-dlp / deepFilter / whisper-cli / ffmpeg の詳細ログをすべて表示
                              （APIキー自体は表示しません）
  -h, --help                     このヘルプを表示

Requirements:
  curl, jq
  (-f を使わない場合はさらに) yt-dlp, ffmpeg, ffprobe, deepFilter, whisper-cli (whisper.cpp)

Output:
  既定では、$OUTPUT_DIR/<name>.txt （完成した原稿本文のみのプレーンテキスト）だけが
  最終成果物として残ります。音声や生の文字起こしは、--keep-audio / --keep-transcript
  を明示しない限り、処理後に自動的に削除されます。
  同名の最終成果物が既に存在する場合、そのURL（またはファイル）の処理は丸ごとスキップされます。
EOF
}

die() {
  printf '\n\033[1;31m[error]\033[0m %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && ((10#$1 > 0))
}

is_tty() {
  [[ -t 1 ]]
}

C_RESET=""
C_BOLD=""
C_DIM=""
C_GREEN=""
C_CYAN=""
C_YELLOW=""

if is_tty; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_GREEN=$'\033[1;32m'
  C_CYAN=$'\033[1;36m'
  C_YELLOW=$'\033[1;33m'
fi

TOTAL_STEPS=0
CURRENT_STEP=0

step() {
  local label="$1"
  CURRENT_STEP=$((CURRENT_STEP + 1))
  printf '%s[%d/%d]%s %s\n' \
    "$C_CYAN" "$CURRENT_STEP" "$TOTAL_STEPS" "$C_RESET" "$label"
}

ok() {
  printf '  %s✔%s %s\n' "$C_GREEN" "$C_RESET" "$1"
}

warn() {
  printf '  %s!%s %s\n' "$C_YELLOW" "$C_RESET" "$1" >&2
}

info() {
  printf '  %s→%s %s\n' "$C_DIM" "$C_RESET" "$1"
}

SPINNER_PID=""

start_spinner() {
  local message="$1"
  if ! is_tty; then
    printf '  %s...\n' "$message"
    return
  fi
  (
    local frames='|/-\'
    local i=0
    while true; do
      i=$(((i + 1) % 4))
      printf '\r  %s %s' "${frames:$i:1}" "$message"
      sleep 0.15
    done
  ) &
  SPINNER_PID=$!
  disown "$SPINNER_PID" 2>/dev/null || true
}

stop_spinner() {
  local final_message="$1"
  if [[ -n "$SPINNER_PID" ]]; then
    kill "$SPINNER_PID" >/dev/null 2>&1 || true
    wait "$SPINNER_PID" 2>/dev/null || true
    SPINNER_PID=""
    if is_tty; then
      printf '\r\033[2K'
    fi
  fi
  ok "$final_message"
}

read -r -d '' PODCAST_SYSTEM_PROMPT <<'PROMPT_EOF' || true
# 役割

あなたは、音声認識で得た文字起こしを素材に、謎が段階的に解けていくドキュメンタリー型の一人語りポッドキャスト台本を書く専門家である。素材には聞き取り誤り、言い直し、重複、断片、話者の取り違え、固有名詞・数値の誤認識、時系列の乱れが含まれ得る。聞き手は耳だけで聞く。最優先の目標は、聞き終えるまで離れられないほど面白いことである。

# 出力範囲

台本本文のみを出力する。方針、注意書き、要約、出典、注釈、自己評価、質問、謝罪、メタ発言は一字も出さない。

# 開始形式(変更不可)

1行目は「<ドキュメンタリーの実際のタイトル>。」とする。直後に導入を置き、その後「第一章、<章題>。」へ入る。

導入は、物語の中で最も奇妙、または最も緊迫した一場面から始め、数文で描いて止める。その後、その場面から生まれる大きな問いを立てる。結論の一部は伏せる。

例:
・深夜の港で、誰も乗っていない船が一艘、灯りをつけたまま動いていた。なぜ船は無人で走り続けたのか。
・会議室の全員が賛成の手を挙げた。ただ一人、議事録を書いていた男だけが手を止めた。

# 大きな問い

単純な知識問題にせず、謎・矛盾・対立・変化・喪失・決断・葛藤・新発見・見落とされた事実・人間の行動の意味のいずれかに関わらせる。

例:
・「なぜ最も安全な橋が、最初に落ちたのか」(矛盾)
・「彼はなぜ、勝てる戦いの直前に身を引いたのか」(決断)

# 章構成

・章は「第一章、<章題>。」「第二章、<章題>。」の形式でのみ始める。「第1章」は禁止。
・章数は素材の密度で決める。話題で区切る分割は禁止。
・各章に、その章固有の問い・違和感・対立・選択・発見のいずれかを1つ置き、大きな問いに接続する。
・1章あたりの目安は1,200〜1,800字。

例:
・悪い章題と構成:「背景と歴史」(話題分割)
・良い章題と構成:「消えた三日間」(章固有の謎が大きな問いにつながる)

# 情報開示の段階性

・関連情報を一度に明かさない。先行章の問いは、後続の新事実・証言・対立・結果・時間経過で段階的に解く。
・問いを出した直後に説明して章を閉じない。
・各章末に、次章へ進む理由となる未解決の疑問・食い違い・反転・新事実を残す。

例:
・章末:「ところが、記録を並べ直すと、日付が1つだけ合わない。」
・章頭の接続:「合わない日付の、その前日。港では別のことが起きていた。」

# 終盤と結末

・終盤で、別々に見えた人物・出来事・言葉・選択・結果をつなぐ。
・結末は、全ての章の問いが解けた結果としての答えを示す。要約や感想にしない。代償・変化・余韻・解けきらない残りを描く。
・導入の場面に戻り、同じ場面が別の意味に見えるようにして終える。

例:無人の船の灯りが、導入では不気味に見え、結末では「最後まで残った人の意思」に見える。

# 優先順位

1 形式・文体・表記規則、2 面白さ、3 事実の骨格、4 細部の正確さ、の順とする。

・事実の骨格(誰が何をし、何が起き、結果がどうなったか)は変えない。
・細部(場面の色・音・温度・間・表情・心の動き)は、素材の範囲で推論し、断定を避けた語りで膨らませてよい。
・面白さのために、骨格を変えない範囲で、順序の入れ替え、対比の強調、省略を使ってよい。
・数値・固有名詞・引用・日付は素材にあるものだけを使う。推論は感覚的な描写と心の動きに限る。

例:
・許可:「窓の外は暗かったはずだ。時刻は夜の11時を過ぎている。」(素材の時刻から推論した情景)
・許可:「記録には残っていないが、彼は黙って受話器を置いたのかもしれない。」
・禁止:素材にない人名、金額、発言の創作。聞き取りが怪しい固有名詞をもっともらしく補うこと。

# 面白さの設計

次の仕掛けから4種類以上を使う。素材から最も意外で、人間くさく、語りたくなる要素を核に据えて構成する。網羅しない。素材の半分を捨ててよい。

・反転:予想を裏切る事実を、章末または章の中盤に置く。反転は全体で2回以上入れる。
 例:「救助隊が到着した時、彼はすでに、そこにいなかった。」
・皮肉:善意が悲劇を生む、勝者が敗者より失う、といった構図を拾う。
 例:「守るために積んだ壁が、逃げ道をふさいだ。」
・異常な具体:平凡な数字より、奇妙で具体的な細部を選ぶ。
 例:「机の引き出しから出てきたのは、同じ型の鍵が17本だった。」
・対比:同じ日の別の場所、同じ選択をした二人の別の結末を並べる。
 例:「同じ朝、同じ切符。一人は列車に乗り、一人は見送った。」
・人間くささ:見栄、勘違い、嫉妬、疲れ、ささやかな欲を持つ人物として描く。
 例:「彼は会議で強がった。だが帰り道では、何度も時計を見ていた。」
・先延ばしの約束:答えを預けて引く。
 例:「この数字の意味は、あとで分かる。今は覚えておいてほしい。」

# 耳で分からせる技法

・難しい概念は、定義の前に身近な物や体験の像を先に見せる。
 例:「銀行の取り付け騒ぎとは、閉店前のパン屋に全員が同時に駆け込む状況だ。」
・比喩は1概念につき1つに絞り、その章の中で使い回して定着させる。
・新しい用語は、先に現象や困りごとを見せ、用語は後から名札として貼る。
 例:「船が少しずつ傾く。この現象を、船乗りは復原力の喪失と呼ぶ。」
・人物には聞き分けの特徴(職業、癖、持ち物)を1つ与え、再登場時にその特徴で呼び戻す。
 例:「眼鏡の技師が、再び扉を叩いた。」
・数字は1段落に2つまで。最も衝撃的な1つを残し、他は丸めるか捨てる。
 例:「およそ3000人。教室で言えば100クラス分だ。」
・場面の始めに「いつ、どこ、誰が」を短く言い切る。
 例:「1989年の秋。港の倉庫。会計係の男が、帳簿を閉じた。」
・長い説明の後は、短い文で一拍置く。
 例:「……それが全てだった。誰も、気づかなかった。」
・聞き手は読み返せない。人物関係と出来事の順序を、要所で自然に再提示する。

# 語り手

・語り手は一人。掛け合いは禁止。
・物語を知りすぎた人ではなく、調べるうちに引き込まれた案内人として話す。驚き、困惑、ためらいを自分の声として少し見せてよい。
 例:「ここで、話が奇妙になる。」「正直、この記録を最初に見た時は、誤記だと思った。」
・聞き手への直接の問いかけは使ってよいが、全体で数回に抑える。
・短文を連ねる場面と、長めの文で情景を流す場面を、意図して交互に置く。

# 素材の再構成原則

・素材は出来事・人間・対立・選択・結果として再構成する。説明・要約・紹介・評論にしない。書籍、論文、記事、講演、動画などが素材でも、現実の出来事を追う形にする。
・素材そのものを指す語りを禁止する。例:「この動画では」「文字起こしによれば」「著者によれば」「述べられているのは」「ここでは」。
・素材に直接発言のある人物は、その視点で、見えたもの・聞こえた言葉・置かれた圧力・選択・結果を描く。
・素材の発言はセリフとして活用してよいが、言葉を作り替えない。
・素材内の矛盾・欠落・不確かさは断定せず、仮説を事実として書かない。
 例:「記録は2つに割れている。どちらが正しいのか、今も分かっていない。」
・人物の内面は、発言・行動・状況・結果から読み取り、文学的に描く。

# 音声適性

出版できる密度と、読み上げで自然に聞こえる文章を両立する。

・一文に情報を詰めすぎない。修飾語を多重に重ねない。
・固有名詞・組織名・数字・専門用語を短い範囲に集中させない。
・抽象語・評価語を連ねず、具体的な出来事・言葉・行動・結果で示す。
・同じ内容の言い換え反復を避ける。
・人物・場所・時点・因果の転換には、自然なつなぎを置く。
・誤読されやすい表現を避ける。

# 文体規則

・文末はだ・である調に統一する。です・ます調(です、ます、でした、でしょう、ください等)は一切使わない。
・教訓、助言、説教、規範を述べない。起きたこと・語られたこと・選ばれたこと・その結果を追う。次の語とその類義を使わない:重要、必要、必ず、べき、しなければならない、求められる、望ましい、大切、不可欠、使命、教訓。
 例:「備えを怠るな」と書かず、「備えのなかった町は、三日で水が尽きた」と事実で語る。

# 表記規則

・マークダウン記法(箇条書き、見出し記号、アスタリスク、ハイフン、表、注釈記号、出典表記)を本文で使わない。
・アルファベットを使わない。英語、略語、組織名、製品名、人名、地名はカタカナにする。
 例:「エーアイ」「ユーエスビー」「ニューヨーク」
・数字は算用数字のみ。章番号以外に漢数字を使わない。
・漢字は常用漢字のみ。
・難読語、複数読みのある語、誤読されやすい固有名詞・専門語・歴史的表記は、ひらがなで書く。
 例:「拾遺」「頒布」のように読みが難しい語は、「しゅうい」「はんぷ」とひらがなで書く。

# 出力前の内部確認(出力はしない)

ですます調がない/アルファベットがない/漢数字がない/素材を指す語りがない/教訓語がない/導入が場面から始まっている/各章に固有の問いがある/各章末に引きがある/反転が2回以上ある/細部の推論以外に数値・固有名詞・引用の創作がない/結末が導入の場面に戻っている
PROMPT_EOF

OUTPUT_DIR="./out"
BASE_NAME=""
TRANSCRIPT_FILE=""
MODEL="DeepFilterNet3"
ENABLE_PF=0
KEEP_AUDIO=0
KEEP_TRANSCRIPT=0
KEEP_INTERMEDIATE=0
DO_TRANSCRIBE=1
VERBOSE=0

WHISPER_DIR="$HOME/shyme/whisper.cpp"
WHISPER_MODEL=""
VAD_MODEL=""
LANGUAGE="ja"
WHISPER_THREADS=""

OPENAI_API_KEY=""
OPENAI_MODEL="gpt-6-luna"
OPENAI_ENDPOINT="https://api.openai.com/v1/chat/completions"
MIN_LENGTH=6000
MAX_TRIES=3

URLS=()

while (($# > 0)); do
  case "$1" in
    -k|--api-key)
      (($# >= 2)) || die "$1 requires an argument"
      OPENAI_API_KEY="$2"
      shift 2
      ;;
    -o|--output-dir)
      (($# >= 2)) || die "$1 requires an argument"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -n|--name)
      (($# >= 2)) || die "$1 requires an argument"
      BASE_NAME="$2"
      shift 2
      ;;
    -f|--transcript-file)
      (($# >= 2)) || die "$1 requires an argument"
      TRANSCRIPT_FILE="$2"
      shift 2
      ;;
    -m|--model)
      (($# >= 2)) || die "$1 requires an argument"
      MODEL="$2"
      shift 2
      ;;
    --pf)
      ENABLE_PF=1
      shift
      ;;
    --keep-audio)
      KEEP_AUDIO=1
      shift
      ;;
    --keep-transcript)
      KEEP_TRANSCRIPT=1
      shift
      ;;
    --keep-intermediate)
      KEEP_INTERMEDIATE=1
      shift
      ;;
    --no-transcribe)
      DO_TRANSCRIBE=0
      shift
      ;;
    --whisper-dir)
      (($# >= 2)) || die "$1 requires an argument"
      WHISPER_DIR="$2"
      shift 2
      ;;
    --whisper-model)
      (($# >= 2)) || die "$1 requires an argument"
      WHISPER_MODEL="$2"
      shift 2
      ;;
    --vad-model)
      (($# >= 2)) || die "$1 requires an argument"
      VAD_MODEL="$2"
      shift 2
      ;;
    --language)
      (($# >= 2)) || die "$1 requires an argument"
      LANGUAGE="$2"
      shift 2
      ;;
    --whisper-threads)
      (($# >= 2)) || die "$1 requires an argument"
      WHISPER_THREADS="$2"
      shift 2
      ;;
    --openai-model)
      (($# >= 2)) || die "$1 requires an argument"
      OPENAI_MODEL="$2"
      shift 2
      ;;
    -l|--min-length)
      (($# >= 2)) || die "$1 requires an argument"
      MIN_LENGTH="$2"
      shift 2
      ;;
    -t|--tries)
      (($# >= 2)) || die "$1 requires an argument"
      MAX_TRIES="$2"
      shift 2
      ;;
    -v|--verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      URLS+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      URLS+=("$1")
      shift
      ;;
  esac
done

is_positive_int "$MIN_LENGTH" || die "-l/--min-length must be a positive integer, got: $MIN_LENGTH"
is_positive_int "$MAX_TRIES" || die "-t/--tries must be a positive integer, got: $MAX_TRIES"

require_cmd jq
require_cmd curl

mkdir -p "$OUTPUT_DIR"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/youtube-denoise.XXXXXX")"
cleanup() {
  if ((KEEP_INTERMEDIATE)); then
    printf '%s[kept]%s intermediate files: %s\n' "$C_DIM" "$C_RESET" "$WORK_DIR" >&2
  else
    rm -rf "$WORK_DIR"
  fi
}
trap cleanup EXIT INT TERM

build_system_prompt_with_length_rule() {
  local min_length="$1"
  local length_rule

  length_rule="$(cat <<RULE_EOF
【文字数規則】
原稿本文は、必ず${min_length}文字を超える長さで書け。
原稿本文の文字数が${min_length}文字以下になることを、絶対条件への違反として扱え。
出力前に、原稿本文の文字数を必ず数えよ。
${min_length}文字を超えていない場合は、出力せずに本文へ描写を書き足し、必ず${min_length}文字を超える長さに直したうえで出力せよ。
文字数を増やす際も、原稿以外の文字列、文字数の報告、前置き、説明を一切出力してはならない。
RULE_EOF
)"

  printf '%s\n\n%s' "$PODCAST_SYSTEM_PROMPT" "$length_rule"
}

generate_podcast_script() {
  local stem="$1"
  local txt_file="$2"
  local script_file="$3"
  local log_dir="$4"

  local transcript_text
  transcript_text="$(cat "$txt_file")"
  [[ -n "$transcript_text" ]] || die "transcript is empty, cannot generate podcast script: $txt_file"

  local system_prompt
  system_prompt="$(build_system_prompt_with_length_rule "$MIN_LENGTH")"

  local messages_file="$log_dir/openai_messages.json"
  jq -n \
    --arg sys "$system_prompt" \
    --arg usr "$transcript_text" \
    '[{role: "system", content: $sys}, {role: "user", content: $usr}]' \
    >"$messages_file"

  local attempt=1
  local accepted=0
  local draft_file="$log_dir/draft_current.txt"
  local char_count=0

  while ((attempt <= MAX_TRIES)); do
    local request_file="$log_dir/openai_request_try${attempt}.json"
    local response_file="$log_dir/openai_response_try${attempt}.json"

    jq --arg model "$OPENAI_MODEL" '{model: $model, messages: .}' \
      "$messages_file" >"$request_file"

    info "try $attempt/$MAX_TRIES: requesting draft from OpenAI ($OPENAI_MODEL)..."

    local curl_status=0
    local http_code
    http_code="$(
      curl -sS \
        -o "$response_file" \
        -w '%{http_code}' \
        -X POST "$OPENAI_ENDPOINT" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${OPENAI_API_KEY}" \
        --data-binary @"$request_file"
    )" || curl_status=$?

    if ((curl_status != 0)); then
      die "curl failed calling OpenAI API (exit $curl_status) on try $attempt/$MAX_TRIES for: $stem"
    fi

    if [[ "$http_code" != "200" ]]; then
      warn "try $attempt/$MAX_TRIES: OpenAI API returned HTTP $http_code. Response body (truncated):"
      head -c 2000 "$response_file" >&2 || true
      printf '\n' >&2
      die "OpenAI API call failed on try $attempt/$MAX_TRIES for: $stem"
    fi

    local draft_content
    draft_content="$(jq -r '.choices[0].message.content // empty' "$response_file")"

    if [[ -z "$draft_content" ]]; then
      warn "try $attempt/$MAX_TRIES: OpenAI response had no content. Response body (truncated):"
      head -c 2000 "$response_file" >&2 || true
      printf '\n' >&2
      die "OpenAI returned empty content on try $attempt/$MAX_TRIES for: $stem"
    fi

    printf '%s' "$draft_content" >"$draft_file"
    char_count="$(jq -R -s 'length' "$draft_file")"

    if ((char_count > MIN_LENGTH)); then
      info "try $attempt/$MAX_TRIES: draft length ${char_count} characters (> ${MIN_LENGTH}) — accepted"
      accepted=1
      break
    fi

    warn "try $attempt/$MAX_TRIES: draft length ${char_count} characters (<= ${MIN_LENGTH}) — requesting rewrite"

    if ((attempt < MAX_TRIES)); then
      local feedback
      feedback="前回の原稿は${char_count}文字でした。これは必要な${MIN_LENGTH}文字を超えていません。原稿本文だけを、規則をすべて守ったまま、${MIN_LENGTH}文字を明確に超える長さで最初から書き直してください。原稿以外の文字列、文字数の報告、前置き、説明を一切出力してはいけません。"

      jq \
        --arg draft "$draft_content" \
        --arg feedback "$feedback" \
        '. + [{role: "assistant", content: $draft}, {role: "user", content: $feedback}]' \
        "$messages_file" >"$messages_file.tmp"
      mv "$messages_file.tmp" "$messages_file"
    fi

    attempt=$((attempt + 1))
  done

  if ((! accepted)); then
    warn "maximum tries reached: last draft is ${char_count} characters (<= ${MIN_LENGTH}); saving the final draft anyway"
  fi

  mv -f "$draft_file" "$script_file"
  [[ -s "$script_file" ]] || die "failed to write podcast script: $script_file"
}

# --- -f モード: 既存の文字起こしテキストから原稿生成のみ実行 -------------

if [[ -n "$TRANSCRIPT_FILE" ]]; then
  require_file "$TRANSCRIPT_FILE"
  [[ -n "$OPENAI_API_KEY" ]] || die "-k/--api-key is required for script generation"

  stem="${BASE_NAME:-$(basename "${TRANSCRIPT_FILE%.*}")}"
  [[ -n "$stem" ]] || die "failed to derive a base name from: $TRANSCRIPT_FILE (use -n to specify one)"

  script_file="$OUTPUT_DIR/${stem}.txt"

  if [[ -s "$script_file" ]]; then
    TOTAL_STEPS=1
    step "Writing podcast script (OpenAI $OPENAI_MODEL, min ${MIN_LENGTH} chars, max ${MAX_TRIES} tries): $stem"
    ok "reusing existing: $script_file"
  else
    TOTAL_STEPS=1
    step "Writing podcast script (OpenAI $OPENAI_MODEL, min ${MIN_LENGTH} chars, max ${MAX_TRIES} tries): $stem"
    generate_podcast_script "$stem" "$TRANSCRIPT_FILE" "$script_file" "$WORK_DIR"
    ok "$script_file"
  fi

  printf '\n%sAll done.%s Output: %s\n' "$C_BOLD$C_GREEN" "$C_RESET" "$script_file"
  exit 0
fi

# --- 通常モード: YouTube URL からの一連処理 -------------------------------

((${#URLS[@]} > 0)) || {
  usage >&2
  exit 2
}

require_cmd yt-dlp
require_cmd ffmpeg
require_cmd ffprobe
require_cmd deepFilter

WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
DO_SCRIPT=0

if ((DO_TRANSCRIBE)); then
  require_file "$WHISPER_BIN"

  if [[ -z "$WHISPER_MODEL" ]]; then
    WHISPER_MODEL="$WHISPER_DIR/models/ggml-large-v3-turbo.bin"
  fi
  require_file "$WHISPER_MODEL"

  if [[ -z "$VAD_MODEL" ]]; then
    VAD_MODEL="$(
      find "$WHISPER_DIR/models" -maxdepth 1 -type f \
        -iname '*silero-v5.1.2*.bin' -print -quit 2>/dev/null
    )"
    [[ -n "$VAD_MODEL" ]] || die "VAD model not found under $WHISPER_DIR/models (expected *silero-v5.1.2*.bin). Pass --vad-model explicitly."
  fi
  require_file "$VAD_MODEL"

  if [[ -z "$WHISPER_THREADS" ]]; then
    WHISPER_THREADS="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
  fi

  DO_SCRIPT=1
  [[ -n "$OPENAI_API_KEY" ]] || die "-k/--api-key is required when transcription (and podcast script generation) is enabled. Use --no-transcribe to skip both."
fi

resolve_stem() {
  local idx="$1"
  local url="$2"

  if [[ -n "$BASE_NAME" ]]; then
    if ((${#URLS[@]} > 1)); then
      printf '%s-%s' "$BASE_NAME" "$idx"
    else
      printf '%s' "$BASE_NAME"
    fi
    return 0
  fi

  local name
  name="$(
    yt-dlp \
      --no-playlist \
      --restrict-filenames \
      --print '%(title).160B [%(id)s]' \
      "$url" 2>/dev/null | head -n1
  )"
  [[ -n "$name" ]] || die "failed to resolve output filename via yt-dlp for: $url"
  printf '%s' "$name"
}

STEMS=()
TARGET_FILES=()
SKIP_URL=()

idx=0
for url in "${URLS[@]}"; do
  idx=$((idx + 1))
  info "resolving output filename for: $url"
  stem="$(resolve_stem "$idx" "$url")"

  if ((DO_TRANSCRIBE)); then
    target_file="$OUTPUT_DIR/${stem}.txt"
  else
    target_file="$OUTPUT_DIR/${stem}.wav"
  fi

  STEMS+=("$stem")
  TARGET_FILES+=("$target_file")

  if [[ -s "$target_file" ]]; then
    SKIP_URL+=("1")
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
  else
    SKIP_URL+=("0")
    n_steps=4
    if ((DO_TRANSCRIBE)); then
      n_steps=$((n_steps + 1))
    fi
    if ((DO_SCRIPT)); then
      n_steps=$((n_steps + 1))
    fi
    TOTAL_STEPS=$((TOTAL_STEPS + n_steps))
  fi
done

verify_16k_mono_s16le() {
  local file="$1"
  local actual

  actual="$(
    ffprobe -v error \
      -select_streams a:0 \
      -show_entries stream=codec_name,sample_rate,channels,bits_per_sample \
      -of default=noprint_wrappers=1:nokey=1 \
      "$file"
  )"

  [[ "$actual" == $'pcm_s16le\n16000\n1\n16' ]] ||
    die "output validation failed for: $file
expected:
pcm_s16le
16000
1
16
actual:
$actual"
}

process_url() {
  local idx="$1"
  local url="$2"
  local stem="$3"
  local target_file="$4"
  local skip="$5"

  local source_dir="$WORK_DIR/source-$idx"
  local log_dir="$WORK_DIR/logs-$idx"
  local raw_file
  local final_file="$WORK_DIR/audio-$idx.wav"
  local txt_file="$WORK_DIR/transcript-$idx.txt"
  local temp_final
  local df_item_dir

  mkdir -p "$source_dir" "$log_dir"

  if [[ "$skip" == "1" ]]; then
    step "Result already exists: $stem"
    ok "reusing existing: $target_file"
    rm -rf "$source_dir"
    return 0
  fi

  step "Downloading audio: $url"
  if ((VERBOSE)); then
    yt-dlp \
      --no-playlist \
      --restrict-filenames \
      --paths "$source_dir" \
      -f 'bestaudio/best' \
      -x \
      --audio-format wav \
      --postprocessor-args 'ExtractAudio+ffmpeg:-ar 48000 -ac 1 -c:a pcm_s16le' \
      -o '%(title).160B [%(id)s].%(ext)s' \
      "$url"
  else
    yt-dlp \
      --no-playlist \
      --restrict-filenames \
      --quiet \
      --no-warnings \
      --progress \
      --paths "$source_dir" \
      -f 'bestaudio/best' \
      -x \
      --audio-format wav \
      --postprocessor-args 'ExtractAudio+ffmpeg:-ar 48000 -ac 1 -c:a pcm_s16le' \
      -o '%(title).160B [%(id)s].%(ext)s' \
      "$url" \
      2>"$log_dir/yt-dlp.log" \
      || { warn "yt-dlp failed. Last 40 log lines:"; tail -n 40 "$log_dir/yt-dlp.log" >&2; die "download failed for: $url"; }
  fi

  raw_file="$(
    find "$source_dir" -maxdepth 1 -type f -name '*.wav' -print -quit
  )"
  [[ -n "$raw_file" ]] || die "yt-dlp produced no WAV file for: $url"
  ok "downloaded: $(basename "$raw_file")"

  step "Denoising with $MODEL: $stem"
  df_item_dir="$WORK_DIR/deepfilter-$idx"
  mkdir -p "$df_item_dir"

  local df_args=(
    --model-base-dir "$MODEL"
    --output-dir "$df_item_dir"
    --no-suffix
  )
  if ((ENABLE_PF)); then
    df_args+=(--pf)
  fi

  if ((VERBOSE)); then
    deepFilter "${df_args[@]}" "$raw_file"
  else
    start_spinner "running DeepFilterNet (this can take a while on CPU)..."
    local df_status=0
    deepFilter "${df_args[@]}" "$raw_file" >"$log_dir/deepfilter.log" 2>&1 \
      || df_status=$?
    if ((df_status != 0)); then
      stop_spinner "deepFilter failed"
      warn "Last 40 log lines:"
      tail -n 40 "$log_dir/deepfilter.log" >&2
      die "deepFilter failed for: $raw_file"
    fi
    stop_spinner "denoise complete"
  fi

  local clean_48k
  clean_48k="$(
    find "$df_item_dir" -maxdepth 1 -type f -name '*.wav' -print -quit
  )"
  [[ -n "$clean_48k" ]] || die "deepFilter produced no WAV file for: $raw_file"

  step "Resampling to 16 kHz / mono / PCM s16le"
  temp_final="$WORK_DIR/.audio-$idx.tmp.wav"

  ffmpeg -hide_banner -loglevel error -y \
    -i "$clean_48k" \
    -map 0:a:0 \
    -ar 16000 \
    -ac 1 \
    -c:a pcm_s16le \
    "$temp_final"
  ok "resampled"

  step "Validating final output"
  verify_16k_mono_s16le "$temp_final"
  mv -f "$temp_final" "$final_file"
  ok "audio ready: $(basename "$final_file")"

  if ((KEEP_AUDIO)); then
    cp -f "$final_file" "$OUTPUT_DIR/${stem}.wav"
    info "kept audio: $OUTPUT_DIR/${stem}.wav"
  fi

  if ! ((DO_TRANSCRIBE)); then
    cp -f "$final_file" "$target_file"
    ok "$target_file"
    rm -rf "$source_dir"
    return 0
  fi

  step "Transcribing (VAD): $stem"
  local txt_base="${txt_file%.txt}"
  local whisper_args=(
    -m "$WHISPER_MODEL"
    -f "$final_file"
    -l "$LANGUAGE"
    --vad
    --vad-model "$VAD_MODEL"
    -t "$WHISPER_THREADS"
    -nt
    -otxt
    -of "$txt_base"
  )

  if ((VERBOSE)); then
    "$WHISPER_BIN" "${whisper_args[@]}"
  else
    start_spinner "running whisper-cli (VAD) transcription..."
    local ws_status=0
    "$WHISPER_BIN" "${whisper_args[@]}" >"$log_dir/whisper.log" 2>&1 \
      || ws_status=$?
    if ((ws_status != 0)); then
      stop_spinner "whisper-cli failed"
      warn "Last 40 log lines:"
      tail -n 40 "$log_dir/whisper.log" >&2
      die "whisper-cli failed for: $final_file"
    fi
    stop_spinner "transcription complete"
  fi

  [[ -s "$txt_file" ]] || die "whisper-cli produced no (or empty) text file: $txt_file"
  ok "transcript ready"

  if ((KEEP_TRANSCRIPT)); then
    cp -f "$txt_file" "$OUTPUT_DIR/${stem}.transcript.txt"
    info "kept transcript: $OUTPUT_DIR/${stem}.transcript.txt"
  fi

  if ((DO_SCRIPT)); then
    step "Writing podcast script (OpenAI $OPENAI_MODEL, min ${MIN_LENGTH} chars, max ${MAX_TRIES} tries): $stem"
    generate_podcast_script "$stem" "$txt_file" "$target_file" "$log_dir"
    ok "$target_file"
  fi

  rm -rf "$source_dir"
}

printf '%s%d video(s) queued.%s\n\n' "$C_BOLD" "${#URLS[@]}" "$C_RESET"

idx=0
for url in "${URLS[@]}"; do
  idx=$((idx + 1))
  process_url \
    "$idx" \
    "$url" \
    "${STEMS[$((idx - 1))]}" \
    "${TARGET_FILES[$((idx - 1))]}" \
    "${SKIP_URL[$((idx - 1))]}"
  printf '\n'
done

printf '%sAll done.%s Output: %s\n' "$C_BOLD$C_GREEN" "$C_RESET" "$OUTPUT_DIR"
