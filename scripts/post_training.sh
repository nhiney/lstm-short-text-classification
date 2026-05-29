#!/bin/bash
# ============================================================
# post_training.sh — Chạy sau khi copy kết quả từ Colab/Drive về
#
# Usage:
#   bash scripts/post_training.sh
# ============================================================

PYTHON="${1:-python3}"   # Truyền Python path nếu cần: bash scripts/post_training.sh /path/to/python

cd "$(dirname "$0")/.."
echo "Working dir: $(pwd)"
echo ""

# Kiểm tra file bắt buộc
check_file() {
    if [ -f "$1" ]; then
        echo "  ✓ $1"
    else
        echo "  ✗ MISSING: $1"
        MISSING=1
    fi
}

echo "=== Kiểm tra files ==="
MISSING=0
check_file "data/processed/vocabulary.pkl"
check_file "outputs/models/lstm_best.pt"
check_file "outputs/models/dnn_best.pt"
check_file "outputs/models/tfidf.pkl"

if [ "$MISSING" = "1" ]; then
    echo ""
    echo "⚠️  Thiếu file — hãy copy kết quả từ Google Drive về trước!"
    echo "   data/processed/  ← từ Drive: data_processed/"
    echo "   outputs/         ← từ Drive: outputs/"
    exit 1
fi

echo ""
echo "=== Evaluate all models ==="
$PYTHON main.py evaluate

echo ""
echo "=== Generate figures ==="
$PYTHON main.py visualize

echo ""
echo "=== Error analysis ==="
$PYTHON main.py analyze lstm

echo ""
echo "✅ Done! Xem kết quả tại:"
echo "   outputs/reports/comparison.json"
echo "   outputs/figures/*.png"
