#!/usr/bin/env bash
# =============================================================================
# download_model_windows.sh
# Cài Ollama trên Windows nếu chưa có, rồi pull model Qwen2.5:3B
#
# Cách dùng trên Windows Git Bash:
#   bash scripts/download_model_windows.sh
#   bash scripts/download_model_windows.sh qwen2.5:3b
# =============================================================================

set -euo pipefail

MODEL="${1:-qwen3:8b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# Đọc .env nếu có
if [ -f ".env" ]; then
    export $(grep -E '^OLLAMA_HOST=' .env | xargs) 2>/dev/null || true
    OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step() { echo -e "\n${BLUE}──── $* ────${NC}"; }

is_windows() {
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*) return 0 ;;
        *) return 1 ;;
    esac
}

step "Kiểm tra môi trường"

if ! is_windows; then
    warn "Script này dành cho Windows Git Bash/MSYS2."
fi

step "Kiểm tra GPU"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
        | awk -F',' '{printf "  GPU: %s | Total: %s | Free: %s\n", $1, $2, $3}'
else
    warn "Không tìm thấy nvidia-smi — Ollama có thể chạy CPU hoặc GPU chưa được cấu hình PATH."
fi

step "Kiểm tra Ollama"

if command -v ollama &>/dev/null; then
    INSTALLED_VER=$(ollama --version 2>/dev/null | grep -oE '[0-9.]+' | head -1 || true)
    log "Ollama đã cài: v${INSTALLED_VER}"
else
    warn "Ollama chưa có trong PATH."

    OLLAMA_EXE="/c/Users/$USERNAME/AppData/Local/Programs/Ollama/ollama.exe"

    if [ -f "$OLLAMA_EXE" ]; then
        log "Tìm thấy Ollama tại: $OLLAMA_EXE"
        export PATH="$PATH:/c/Users/$USERNAME/AppData/Local/Programs/Ollama"
    else
        err "Chưa cài Ollama trên Windows."
        echo ""
        echo "Hãy cài Ollama tại:"
        echo "  https://ollama.com/download/windows"
        echo ""
        echo "Sau khi cài xong, mở lại Git Bash và chạy lại script."
        exit 1
    fi
fi

step "Khởi động Ollama server"

if curl -sf "${OLLAMA_HOST}/api/tags" &>/dev/null; then
    log "Ollama server đang chạy tại ${OLLAMA_HOST}"
else
    log "Đang khởi động Ollama server trên Windows..."

    if command -v powershell.exe &>/dev/null; then
        powershell.exe -NoProfile -Command "Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden"
    else
        nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
    fi

    echo -n "  Đang chờ server khởi động"
    for i in $(seq 1 30); do
        if curl -sf "${OLLAMA_HOST}/api/tags" &>/dev/null; then
            echo ""
            log "Server sẵn sàng sau ${i}s"
            break
        fi

        echo -n "."
        sleep 1

        if [ "$i" -eq 30 ]; then
            echo ""
            err "Ollama server không khởi động được sau 30s."
            err "Hãy thử mở Ollama app thủ công rồi chạy lại script."
            exit 1
        fi
    done
fi

step "Pull model: ${MODEL}"

EXISTING_MODELS=$(curl -sf "${OLLAMA_HOST}/api/tags" | python -c \
    "import sys,json; data=json.load(sys.stdin); print('\n'.join(m.get('name','') for m in data.get('models',[])))" \
    2>/dev/null || echo "")

if echo "${EXISTING_MODELS}" | grep -q "^${MODEL}$"; then
    log "Model '${MODEL}' đã tồn tại — bỏ qua bước pull."
else
    log "Đang pull '${MODEL}' — lần đầu có thể mất vài phút..."
    ollama pull "${MODEL}"
    log "Pull '${MODEL}' thành công!"
fi

step "Test model"

RESP=$(ollama run "${MODEL}" "Xin chào, bạn có hoạt động không?" 2>&1 | head -3 || true)
log "Phản hồi test: ${RESP:0:120}..."

step "Kiểm tra VRAM"

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader \
        | awk -F',' '{printf "  VRAM Used: %s | Free: %s\n", $1, $2}'
else
    warn "Bỏ qua kiểm tra VRAM vì không tìm thấy nvidia-smi."
fi

step "Cài Python dependencies"

if [ -f "pyproject.toml" ] || [ -f "requirements.txt" ]; then
    if command -v python &>/dev/null; then
        python -m pip install requests python-dotenv --quiet
        log "Đã cài: requests, python-dotenv"
    else
        warn "Không tìm thấy python."
    fi
fi

step "Hoàn thành"

log "Mọi thứ đã sẵn sàng!"
echo ""
echo "  Chạy pipeline:"
echo "    python run_context_inference.py"
echo ""
echo "  Tùy chọn:"
echo "    python run_context_inference.py --input ./transcripts --workers 1"
echo ""