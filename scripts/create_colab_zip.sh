#!/bin/bash
# ============================================================
# create_colab_zip.sh
# Tạo file zip để upload lên Google Colab / Drive
#
# Usage:
#   chmod +x scripts/create_colab_zip.sh
#   ./scripts/create_colab_zip.sh
#
# Tạo ra: project.zip (~150KB, không bao gồm model .pt)
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ZIP="$PROJECT_ROOT/project.zip"

echo "Creating Colab zip from: $PROJECT_ROOT"
echo "Output: $OUTPUT_ZIP"

cd "$PROJECT_ROOT"

# Xoá zip cũ nếu có
rm -f "$OUTPUT_ZIP"

# Tạo zip — bỏ qua các file/thư mục nặng hoặc không cần thiết
zip -r project.zip . \
    --exclude "*.git*" \
    --exclude "*__pycache__*" \
    --exclude "*.pyc" \
    --exclude "*.DS_Store" \
    --exclude "outputs/models/*.pt" \
    --exclude "outputs/models/xlmr_tokenizer/*" \
    --exclude "outputs/models/fusion_tokenizer/*" \
    --exclude "project.zip" \
    --exclude "*.ipynb_checkpoints*" \
    --exclude ".venv/*" \
    --exclude "venv/*"

SIZE=$(du -sh "$OUTPUT_ZIP" | cut -f1)
echo ""
echo "✅  Created: project.zip ($SIZE)"
echo ""
echo "Bước tiếp theo:"
echo "  1. Upload project.zip lên Google Drive:"
echo "     MyDrive/lstm-text-classification/project.zip"
echo "  2. Mở notebooks/colab_training.ipynb trong Colab"
echo "  3. Bật GPU: Runtime → Change runtime type → T4 GPU"
echo "  4. Chạy từng cell từ trên xuống"
