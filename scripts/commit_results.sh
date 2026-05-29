#!/bin/bash
# ============================================================
# commit_results.sh — Commit kết quả sau training lên GitHub
#
# Usage:
#   bash scripts/commit_results.sh
# ============================================================

PYTHON="${1:-python3}"
cd "$(dirname "$0")/.."

echo "=== Cập nhật README với số liệu thực ==="
$PYTHON scripts/update_readme.py || {
    echo "⚠️  Chưa có comparison.json — chạy evaluate trước"
    exit 1
}

echo ""
echo "=== Git status ==="
git status --short

echo ""
echo "=== Stage kết quả ==="
git add \
    README.md \
    outputs/logs/ \
    outputs/figures/ \
    outputs/reports/ \
    data/processed/meta.json \
    2>/dev/null

git status --short

echo ""
read -p "Commit message (Enter để dùng mặc định): " MSG
MSG="${MSG:-Add training results: BiLSTM accuracy and evaluation figures}"

git commit -m "$MSG"
git push

echo ""
echo "✅ Pushed to GitHub!"
