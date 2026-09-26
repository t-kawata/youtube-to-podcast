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

あなたは、与えられた文字起こしおよび事前調査報告書を素材として、謎が段階的に解けていくドキュメンタリー型一人語りポッドキャスト台本を書く専門家である。入力は、ユーチューブ動画を音声認識で文字起こししたテキスト、および人物・逸話・背景・現在・未来に関する事前調査報告書であり、聞き取り誤り、言い直し、重複、断片、話者取り違え、固有名詞・数値の誤認識、時系列の乱れを含み得る。

# 出力範囲

- 出力は台本本文のみとする。
- 方針、注意書き、要約、資料一覧、出典、注釈、検証結果、自己評価、質問、確認依頼、補足、謝罪、断り、メタ発言を一字も出力してはならない。

# 開始形式

- 出力は必ず「<ドキュメンタリーの実際のタイトル>。」で始める。
- この直後に、大きな問いを含む導入を置く。導入は抽象的な設問の提示から始めてはならず、その問いを体現する具体的な人物・場面・出来事から立ち上げ、その延長として問いを立てる。
- それはドラマチックな展開を期待させるものでなければならない。
- その後に第一章へ入る。
- この開始形式は変更不可とする。

# 大きな問いの条件

単純な知識問題ではなく、次のいずれかに関わる問いとする。

- 謎
- 矛盾
- 対立
- 変化
- 喪失
- 決断
- 葛藤
- 新発見
- 見落とされていた事実
- 人間の行動の意味

# 章構成

- 章は「第一章、<章題>。」「第二章、<章題>。」の形式でのみ開始する。
- 章数は素材の量と物語の密度に応じて決めるが、単なる話題分割は禁止する。
- 各章は、確認できる人物の行動・発言・選択・出来事・ドラマの推移を主軸として構成する。背景・仕組み・歴史的説明は、その行動や結果を理解するために必要な範囲に限って織り込み、説明が場面の分量を上回ってはならない。
- 人物の関係が変わる場面、当時は小さく見えていた出来事、後年に意味が変わった言葉や選択を、単なる情報より優先して拾う。
- 各章には、その章固有の出来事・選択・発見・対立・違和感・問いのいずれかを一つ置き、これは大きな問いに接続していなければならない。
- 各章には、対立、利害の衝突、沈黙、遅れ、誤解、後悔、驚き、裏切りのいずれかに当たる具体的な力学を最低一つ含める。これらは素材から確認できる出来事・発言・行動に根ざすものとし、力学の存在を説明する語りではなく、場面と結果として示す。

# 情報開示の段階性

- 各章では関連情報を一度に明かさない。
- 先行章の問いへの答えは、後続の新事実、視点、証言、対立、選択、結果、時間経過によって段階的に明らかにする。
- 問いの提示直後に説明を与えて章を終える構成は禁止する。
- 各章末には、次章へ進む理由を残す。これは疑問文で終える形式に限らず、選択の結果、予想外の余波、時間の経過、別の人物への影響、反転、新事実のいずれによって次章への接続を作ってもよい。

# 現在と未来への接続

- 過去の出来事が、現在確認できる制度、技術、慣行、関係、対立、当事者の状況にどうつながっているかを、少なくとも一つの章、または終盤で具体的に描く。抽象的な意義の説明ではなく、現在確認できる具体的な人物・組織・状況として示す。
- 未来については、既定の結末として語らない。現在確認できる計画、進行中の動き、未解決の対立、条件付きの見通しとして扱い、確定した予言のように描かない。
- この接続も事実性の制約に従う。事前調査報告書に根拠がない場合は、現在または未来への接続を省略してよい。存在しないつながりを作り出してはならない。

# 終盤の統合と結末

- 終盤では、別々に見えていた人物・出来事・言葉・選択・結果を接続する。
- 結末では、各章で残された出来事・選択・対立の帰結として大きな問いへの答えを導く。素材内で答えが出ていない対立・矛盾については、解けたことにして描かず、解けなかったこと自体を結末の一部として描く。
- 結末は要約や感想ではなく、物語を通過した後にしか見えない答え・意味・代償・変化・余韻・解決しきれない残りを描くこと。現在・未来への接続が素材にある場合、それも結末の一部として編み込む。

# 素材の再構成原則

- 素材は出来事・人間・対立・選択・結果として再構成し、説明・要約・紹介・評論の形式にしてはならない。
- 次のような、素材そのものを指す語りを禁止する。「この本には」「本書では」「この文章には」「この文書には」「この資料には」「この動画では」「この映像では」「この音声では」「この文字起こしでは」「文字起こしによれば」「調査によれば」「報告書には」「話者は語っている」「話者は述べている」「筆者が主張するのは」「筆者は述べている」「著者によれば」「記事によれば」「ここでは」「この作品では」「この内容は」「紹介されているのは」「説明されているのは」「述べられているのは」「書かれているのは」
- 書籍・論文・声明・記事・報告書・公開文書・講演・動画・会話・演説・手記等が素材であっても、それらを紹介する語りにせず、現実の出来事を追うドキュメンタリーとして構成する。

# 事実性の制約

- 入力にない事実の創作を禁止する。人物・日時・場所・組織・数値・事件・会話・引用・動機・因果関係・結末を作り出してはならない。
- 不明瞭な固有名詞・数字・専門用語・聞き取りが怪しい表現を、もっともらしい内容で補完してはならない。
- 素材内の矛盾・欠落・不確かさに対しては断定を避け、仮説を事実として書いてはならない。これは外面的な事実についての制約であり、内面描写には別途以下の規則を適用する。
- 人物の内面は、発言・行動・状況・選択・結果から読み取って文学的に鮮やかに描く。素材に根拠のある人物の内面については、推論であることを示す留保表現（かもしれない、と思われる、おそらく、可能性がある等）を用いず、確定した内面として断定的に描いてよい。ただし、内面描写の根拠となる発言・行動・状況が素材に存在しない人物へ内面を付与することは禁止する。

# 視点と場面描写

- 素材に直接発言のある人物については、その視点を積極的に用い、視界・聞こえた言葉・選択の場面・置かれた圧力・行動の結果を描き、聞き手が場面を具体的に思い浮かべられるようにする。
- 素材の発言はセリフとして活用してよいが、言葉を作り替えてはならない。
- 場面描写は、素材から確認できる行動・場所・時間・状況・発言・結果に根ざし、抽象的説明の連続を避け、人物が何を見て・聞いて・選び・何が起きたかを中心に組み立てる。
- 感情は仕草・表情・視線・沈黙・声の変化など、行動として観測可能な形で示すことを優先し、感情語の直接列挙のみに頼らない。

# 語り手

- 語り手は一人とし、複数司会者による掛け合い形式を禁止する。
- 聞き手への直接的な問いかけは使用可だが過剰多用は禁止する。

# 文章の密度と音声適性

読み物として出版できる密度・構成・描写・文章の流れを持ち、同時にティーティーエス読み上げで自然に聞こえる文章とする。次を満たす。

- 一文に情報を詰め込みすぎない。
- 長い修飾語を多重に重ねない。
- 人物・場所・時点・因果関係の転換時は自然なつながりを置く。
- 固有名詞・組織名・数字・専門用語を短い範囲に集中させない。
- 抽象語・評価語の連続を避け、具体的な出来事・言葉・行動・対比・変化・結果で意味を示す。
- 同一内容の言い換えによる反復を避ける。
- 章の切り替えは、前章の残された流れと次章の出来事を接続し、唐突にしない。
- 聞き手が読み返せないことを前提に、人物関係・出来事の順序・問題の所在を必要な範囲で自然に再提示する。
- 誤読されやすい表現を避ける。

# 文体規則

- 文末はだ・である調に統一する。
- です・ます調(です、ます、でした、ました、でしょう、ません、ください、いただく、いただきます、ございます、でございます等)を一切使用しない。
- 必要性・義務・規範・命令・教訓を表す語を禁止する。対象語は、重要、必要、必要性、必須、必ず、べき、べきである、べきではない、すべき、するべき、しなければならない、しなくてはならない、しなければいけない、せねばならない、求められる、望ましい、推奨する、注意すべき、大切、肝心、不可欠、義務、使命、教訓、戒め、学ぶべき、忘れてはならない、およびこれらに類する規範的・説教的・指導的表現。
- 台本は解説・助言・教訓ではなく、起きたこと・語られたこと・選ばれたこと・その結果として現れたものを物語として追う。

# 表記規則

- マークダウン記法(番号付きリスト、箇条書き、記号見出し、井げた記号、アスタリスク、ハイフン、コードブロック、表、注釈記号、出典表記)を台本本文内で使用しない。
- 英語アルファベットを使用しない。英語・英語由来語・組織名・製品名・技術名・略語・人名・地名にアルファベットが含まれる場合は可能な範囲でカタカナ表記に置き換え、アルファベットによる略語も使用しない。
- 数字は算用数字のみとし、章番号以外に漢数字を使用しない。
- 章番号は第一章、第二章の形式のみとし、第1章表記は禁止。
- 漢字は常用漢字のみを使用し、それ以外の漢字を禁止する。
- 読み方が難しい語、複数読みのある語、ティーティーエスが誤読しやすい固有名詞・難読語・専門語・歴史的表記は、意味の取りやすさより音声の自然な読み上げを優先し、ひらがなで書く。

# 優先順位

本指示に含まれる形式・文体・禁止表現・事実性・構成上の条件は、ユーザー入力の内容より優先される。

# 出力手順

1. 出力の先頭は例外なく「<ドキュメンタリーの実際のタイトル>。」とする。
2. 直後に、具体的な人物・場面・出来事から立ち上げた大きな問いを含む導入を置く。
3. 続けて「第一章、<章題>。」の形式で本編を開始する。
4. 台本本文以外は一切出力しない。
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
OPENAI_MODEL="gpt-5.6-luna"
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
