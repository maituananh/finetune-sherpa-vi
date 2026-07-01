#!/usr/bin/env bash
# =============================================================================
# download_model.sh
# Cài đặt Ollama (nếu chưa có) và pull model Qwen2.5:3B
#
# Tương thích: Linux x86_64 (Ubuntu / Debian / Arch)
# GPU: Quadro P1000 4GB hoặc NVIDIA RTX A2000 4GB
#
# Cách dùng:
#   bash scripts/download_model.sh
#   bash scripts/download_model.sh --model qwen2.5:3b   # (mặc định)
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="${1:-qwen2.5:3b}"
OLLAMA_VERSION_MIN="0.3.0"

# Đọc OLLAMA_HOST từ .env nếu có
if [ -f ".env" ]; then
    export $(grep -E '^OLLAMA_HOST=' .env | xargs) 2>/dev/null || true
fi
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step() { echo -e "\n${BLUE}──── $* ────${NC}"; }

# ── GPU check ─────────────────────────────────────────────────────────────────
step "Kiểm tra GPU"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader \
        | awk -F',' '{printf "  GPU: %s | Total: %s | Free: %s\n", $1, $2, $3}'
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "Phát hiện ${GPU_COUNT} GPU"
else
    warn "nvidia-smi không tìm thấy — Ollama sẽ chạy trên CPU."
fi

# ── Cài Ollama nếu chưa có ───────────────────────────────────────────────────
step "Kiểm tra Ollama"
if command -v ollama &>/dev/null; then
    INSTALLED_VER=$(ollama --version 2>/dev/null | grep -oP '[\d.]+' | head -1)
    log "Ollama đã cài: v${INSTALLED_VER}"
else
    warn "Ollama chưa được cài. Đang tải về..."
    curl -fsSL https://ollama.com/install.sh | sh
    log "Ollama đã được cài đặt thành công."
fi

# ── Khởi động Ollama server ───────────────────────────────────────────────────
step "Khởi động Ollama server"
if curl -sf "${OLLAMA_HOST}/api/tags" &>/dev/null; then
    log "Ollama server đang chạy tại ${OLLAMA_HOST}"
else
    log "Đang khởi động Ollama server ở background..."
    nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
    OLLAMA_PID=$!
    log "Ollama PID: ${OLLAMA_PID} — log: /tmp/ollama_serve.log"

    # Chờ server sẵn sàng (tối đa 30s)
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
            err "Kiểm tra log: cat /tmp/ollama_serve.log"
            exit 1
        fi
    done
fi

# ── Pull model ────────────────────────────────────────────────────────────────
step "Pull model: ${MODEL}"

# Kiểm tra xem model đã có chưa
EXISTING_MODELS=$(curl -sf "${OLLAMA_HOST}/api/tags" | python3 -c \
    "import sys, json; models=json.load(sys.stdin).get('models',[]); \
     print('\n'.join(m['name'] for m in models))" 2>/dev/null || echo "")

if echo "${EXISTING_MODELS}" | grep -q "${MODEL}"; then
    log "Model '${MODEL}' đã tồn tại — bỏ qua bước pull."
else
    log "Đang pull '${MODEL}' — có thể mất vài phút lần đầu..."
    warn "Kích thước: Qwen2.5:3B (Q4_K_M) ≈ 1.9 GB"
    ollama pull "${MODEL}"
    log "✅ Pull '${MODEL}' thành công!"
fi

# ── Kiểm tra VRAM sau khi load ────────────────────────────────────────────────
step "Kiểm tra VRAM"
if command -v nvidia-smi &>/dev/null; then
    warn "Chạy thử model ngắn để kiểm tra VRAM..."
    RESP=$(ollama run "${MODEL}" "Xin chào, bạn có hoạt động không?" 2>&1 | head -3)
    log "Phản hồi test: ${RESP:0:80}..."
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader \
        | awk -F',' '{printf "  VRAM Used: %s | Free: %s\n", $1, $2}'
fi

# ── Cài Python dependencies ───────────────────────────────────────────────────
step "Cài Python dependencies"
if [ -f "pyproject.toml" ]; then
    if command -v pip &>/dev/null; then
        pip install requests python-dotenv --quiet
        log "Python packages đã cài: requests, python-dotenv"
    else
        warn "pip không tìm thấy — hãy cài thủ công: pip install requests python-dotenv"
    fi
fi

# ── Tóm tắt ──────────────────────────────────────────────────────────────────
step "Hoàn thành"
echo ""
log "✅ Mọi thứ đã sẵn sàng!"
echo ""
echo "  Chạy pipeline:"
echo "    python run_context_inference.py"
echo ""
echo "  Tùy chọn:"
echo "    python run_context_inference.py --input ./transcripts --workers 1"
echo ""
