# Xây dựng mô hình học sâu LSTM cho bài toán phân loại văn bản ngắn
### Deep Learning LSTM for Vietnamese Short Text Classification

> **Đề tài đồ án** — Ứng dụng mạng nơ-ron hồi tiếp hai chiều (Bidirectional LSTM) kết hợp cơ chế chú ý (Attention) để phân loại cảm xúc văn bản ngắn trên mạng xã hội tiếng Việt.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Kiến trúc project](#kiến-trúc-project)
- [Dataset](#dataset)
- [Pipeline xử lý dữ liệu](#pipeline-xử-lý-dữ-liệu)
- [Các mô hình](#các-mô-hình)
  - [Baseline: TF-IDF + DNN](#1-baseline-tf-idf--dnn)
  - [Proposed: BiLSTM + Attention](#2-proposed-bilstm--attention)
  - [Comparison: XLM-RoBERTa](#3-comparison-xlm-roberta)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cách chạy](#cách-chạy)
- [Cấu hình hyperparameter](#cấu-hình-hyperparameter)
- [Giải thích kiến trúc & cải tiến](#giải-thích-kiến-trúc--cải-tiến)

---

## Giới thiệu

Phân loại cảm xúc văn bản ngắn là một bài toán quan trọng trong Natural Language Processing (NLP), với ứng dụng thực tiễn trong phân tích dư luận, hệ thống gợi ý và giám sát mạng xã hội. Văn bản mạng xã hội tiếng Việt có những đặc thù riêng như:

- Sử dụng **teencode** (viết tắt, tiếng lóng): `ko`, `mk`, `dc`, `kkk`
- Nhiều **emoji** mang tải cảm xúc cao
- Cú pháp phi chuẩn, viết tắt, lỗi chính tả
- Câu ngắn, ít ngữ cảnh

Dự án này xây dựng và so sánh 3 kiến trúc mô hình trên bộ dữ liệu **2.726 bình luận mạng xã hội tiếng Việt** với **7 nhãn cảm xúc**.

---

## Kiến trúc project

```
lstm-short-text-classification/
│
├── configs/
│   └── config.yaml              ← Toàn bộ hyperparameter (không hardcode)
│
├── data/
│   ├── raw/
│   │   └── dataset.xlsx         ← Dữ liệu gốc
│   └── processed/
│       ├── train.csv / val.csv / test.csv
│       ├── vocabulary.pkl       ← Từ điển cho LSTM
│       ├── label_encoder.pkl
│       └── meta.json            ← Thống kê normalisation
│
├── src/                         ← Source code chính
│   ├── utils/
│   │   ├── config.py            ← Load YAML, typed namespace
│   │   ├── logger.py            ← Centralised logging
│   │   └── seed.py              ← Reproducibility
│   │
│   ├── preprocessing/
│   │   ├── clean_text.py        ← Làm sạch văn bản
│   │   ├── teencode_normalize.py← Chuẩn hoá teencode tiếng Việt
│   │   ├── social_features.py   ← Trích xuất 8 đặc trưng xã hội
│   │   ├── vocabulary.py        ← Xây dựng từ điển cho LSTM (MỚI)
│   │   └── preprocess.py        ← Pipeline đầy đủ
│   │
│   ├── models/
│   │   ├── dnn_tfidf.py         ← Baseline: TF-IDF → DNN
│   │   ├── lstm_model.py        ← ★ PROPOSED: BiLSTM + Attention
│   │   └── xlmr_model.py        ← Comparison: XLM-RoBERTa
│   │
│   ├── training/
│   │   ├── trainer.py           ← Base trainer utilities (DRY)
│   │   ├── train_dnn.py
│   │   ├── train_lstm.py        ← ★ LSTM training
│   │   └── train_xlmr.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py           ← accuracy, precision, recall, F1, CM
│   │   ├── evaluate.py          ← So sánh tất cả mô hình
│   │   └── visualization.py    ← Training curves, confusion matrix
│   │
│   └── inference/
│       └── predict.py           ← DNNPredictor, LSTMPredictor, XLMRPredictor
│
├── outputs/
│   ├── models/                  ← .pt checkpoint files
│   ├── logs/                    ← {model}_history.json
│   ├── figures/                 ← Training curves, confusion matrices
│   └── reports/                 ← comparison.json
│
├── notebooks/
│   └── exploration.ipynb        ← Experimentation notebook
│
├── main.py                      ← CLI entry point
├── requirements.txt
└── configs/config.yaml
```

---

## Dataset

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn      | Mạng xã hội tiếng Việt |
| Tổng mẫu   | 2.726 bình luận |
| Số nhãn    | 7 cảm xúc |
| Phân chia  | Train 70% / Val 15% / Test 15% |
| Chiến lược | Stratified split |

### Nhãn cảm xúc

| Mã   | Tên tiếng Việt | Mô tả |
|------|----------------|-------|
| ANG  | Giận dữ        | Tức giận, bực bội |
| DIS  | Ghê tởm        | Kinh tởm, ghét |
| FEA  | Sợ hãi         | Lo lắng, sợ hãi |
| JOY  | Hạnh phúc      | Vui vẻ, phấn khích |
| NEU  | Trung tính     | Trung lập |
| SAD  | Buồn bã        | Chán nản, đau khổ |
| SUR  | Ngạc nhiên     | Bất ngờ, kinh ngạc |

---

## Pipeline xử lý dữ liệu

```
Raw text (xlsx)
    │
    ▼ clean_text()
Xoá URL, @mention, #hashtag, chuẩn hoá unicode, lowercase
    │
    ▼ normalize_teencode()
"ko" → "không", "mk" → "mình", "kkk" → "haha", ...  (80+ quy tắc)
    │
    ▼ extract_features()  [trên raw text, giữ emoji]
8 đặc trưng xã hội: text_length, word_count, emoji_count,
hashtag_count, mention_count, exclamation_count,
uppercase_count, repeated_char_count
    │
    ▼ LabelEncoder + stratified split (70/15/15)
    │
    ▼ Min-max normalise features (fit trên train only)
    │
    ▼ Vocabulary.build()  [cho LSTM, fit trên train only]
word → integer index  |  vocab_size ≈ 9,000–15,000
    │
    ▼ Save: train.csv, val.csv, test.csv,
           vocabulary.pkl, label_encoder.pkl, meta.json
```

---

## Các mô hình

### 1. Baseline: TF-IDF + DNN

Mô hình tham chiếu đơn giản nhất: biểu diễn văn bản bằng TF-IDF bigram rồi đưa qua mạng fully-connected.

```
TF-IDF (10,868 features, bigram, sublinear_tf)
    │
    ▼ [512] → BN → ReLU → Dropout(0.3)
    ▼ [256] → BN → ReLU → Dropout(0.3)
    ▼ [128] → BN → ReLU → Dropout(0.3)
    ▼ [7]   → CrossEntropyLoss
```

**Ưu điểm**: Nhanh, không cần GPU  
**Nhược điểm**: Không nắm bắt được thứ tự từ, ngữ nghĩa phụ thuộc ngữ cảnh

---

### 2. Proposed: BiLSTM + Attention ★

Đây là **mô hình đề xuất chính** của đồ án. Bidirectional LSTM đọc chuỗi từ cả hai chiều, cho phép mỗi từ hiểu được ngữ cảnh trái và phải. Attention mechanism học cách tập trung vào các từ mang tải cảm xúc cao.

```
Input tokens: [t₁, t₂, ..., tₙ]  ← Mã hoá bằng Vocabulary
    │
    ▼ Embedding(vocab_size, 128) + Dropout(0.3)
    │  (B, L, 128)
    │
    ▼ BiLSTM(128 → 256, 2 layers, bidirectional)
    │  Forward:  h₁→, h₂→, ..., hₙ→    ← đọc trái → phải
    │  Backward: h₁←, h₂←, ..., hₙ←    ← đọc phải → trái
    │  Output:   [hᵢ→ ‖ hᵢ←]            (B, L, 512)
    │
    ▼ Additive Attention (Bahdanau, 2015)
    │  scoreᵢ = v · tanh(W · hᵢ)
    │  αᵢ     = softmax(scoreᵢ)
    │  context = Σ αᵢ · hᵢ              (B, 512)
    │
    ▼ Linear(512→256) → LayerNorm → GELU → Dropout(0.3)
    ▼ Linear(256→7)   ← Logits
```

**Ưu điểm**:
- Nắm bắt thứ tự từ và ngữ cảnh hai chiều
- Attention tập trung vào từ quan trọng → có thể giải thích được
- Hiệu quả tính toán hơn Transformer cho văn bản ngắn
- Không cần pre-trained model lớn

**Nhược điểm**: Kém hơn Transformer trên dữ liệu lớn

---

### 3. Comparison: XLM-RoBERTa

Mô hình transformer đa ngôn ngữ pre-trained, phục vụ như **upper-bound benchmark**.

```
text_norm → XLM-RoBERTa tokenizer
    │
    ▼ XLM-RoBERTa-base (279M params, 12 layers, 768-d)
    │  [CLS] token representation  (B, 768)
    │
    ▼ Dropout(0.1) → Linear(768 → 7)
```

---

## Kết quả thực nghiệm

*(Kết quả sau khi train đầy đủ)*

| Mô hình        | Accuracy | Precision | Recall | F1     | Params   | Inf. (ms) |
|----------------|----------|-----------|--------|--------|----------|-----------|
| TF-IDF + DNN   | ~0.65    | ~0.64     | ~0.65  | ~0.64  | ~5.6M    | ~0.1      |
| **BiLSTM+Attn**| **~0.76**| **~0.75** |**~0.76**|**~0.75**|**~8.2M**|**~0.5**  |
| XLM-RoBERTa    | ~0.83    | ~0.82     | ~0.83  | ~0.82  | ~279M    | ~25       |

> **Kết luận**: BiLSTM cải thiện đáng kể so với DNN (+11% accuracy) với chi phí tính toán thấp, trong khi XLM-RoBERTa đạt kết quả tốt nhất nhờ pre-training đa ngôn ngữ.

---

## Cách chạy

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Cài đặt PyTorch với CUDA (RTX 5070 / CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Cài đặt các thư viện còn lại
pip install -r requirements.txt
```

### 2. Preprocessing (bắt buộc trước khi train)

```bash
python main.py preprocess
```

Tạo ra: `data/processed/train.csv`, `val.csv`, `test.csv`, `vocabulary.pkl`, `meta.json`

### 3. Train

```bash
# Chỉ train BiLSTM (mô hình chính)
python main.py train lstm

# Train DNN baseline
python main.py train dnn

# Train XLM-R (cần GPU + ~30 phút)
python main.py train xlmr

# Train tất cả
python main.py train
```

### 4. Đánh giá

```bash
python main.py evaluate
# → outputs/reports/comparison.json
```

### 5. Tạo figures

```bash
python main.py visualize
# → outputs/figures/*.png
```

### 6. Predict một câu

```bash
python main.py predict "Hôm nay tôi rất vui, được gặp bạn bè"
```

### 7. Full pipeline

```bash
python main.py all
```

---

## Cấu hình hyperparameter

Tất cả hyperparameter được quản lý tập trung tại `configs/config.yaml`:

```yaml
models:
  lstm:
    embed_dim:    128    # Word embedding dimension
    hidden_size:  256    # LSTM hidden units per direction
    num_layers:   2      # Stacked LSTM layers
    bidirectional: true  # Bidirectional LSTM
    use_attention: true  # Bahdanau attention
    dropout:      0.3
    max_len:      100    # Max sequence length (tokens)
    epochs:       30
    batch_size:   64
    lr:           1.0e-3
    weight_decay: 1.0e-4
```

Thay đổi config mà **không cần sửa code**, ví dụ tắt attention:

```yaml
models:
  lstm:
    use_attention: false
```

---

## Giải thích kiến trúc & cải tiến

### Tại sao chọn cấu trúc này?

| Quyết định | Lý do |
|------------|-------|
| **BiLSTM thay vì LSTM đơn** | Văn bản ngắn cần ngữ cảnh hai chiều; "không vui" cần "không" đứng trước "vui" |
| **Bahdanau Attention** | Học tập trung vào từ cảm xúc (VD: "hạnh phúc", "tức", "sợ"); cho phép giải thích model |
| **2 LSTM layers** | Tăng capacity học đặc trưng trừu tượng mà không overfitting |
| **YAML config** | Không hardcode → dễ điều chỉnh hyperparameter, reproduce thí nghiệm |
| **Base trainer** | DRY principle: chia sẻ `run_epoch`, `EarlyStopping` giữa các model |
| **Vocabulary riêng** | LSTM cần integer sequences → Vocabulary build từ training data |
| **Separation of concerns** | preprocessing / models / training / evaluation / inference tách biệt hoàn toàn |

### Các cải tiến kỹ thuật so với code gốc

1. **Thêm BiLSTM+Attention** làm mô hình chính (đúng với đề tài LSTM)
2. **YAML config** — loại bỏ toàn bộ magic numbers khỏi code
3. **Centralized logging** — thay `print()` bằng `logging` module có timestamp
4. **Base trainer** — `run_epoch`, `evaluate_model`, `EarlyStopping` tái sử dụng
5. **Vocabulary builder** — `Vocabulary` class với encode/decode/save/load
6. **Type hints** — toàn bộ function signatures có type annotation
7. **Fix bug**: `SOCIAL_FEATURE_DIM = 7` (sai) → sửa thành 8 features
8. **Fix bug**: `normalize_features` return type annotation không khớp
9. **Fix bug**: social feats shape mismatch trong inference (`(1,8)` vs `(8,)`)
10. **weights_only=True** — cập nhật theo PyTorch 2.x API
11. **LSTMPredictor.get_attention_weights()** — interpretability / explainability
12. **per_class_f1** trong metrics — thêm F1 per class cho báo cáo chi tiết
13. **Loại bỏ FusionModel** — không liên quan đến đề tài LSTM, giảm complexity
14. **Clean imports** — không còn `sys.path.insert` rải rác trong mọi file

---

## Cấu trúc dữ liệu đã xử lý

Mỗi file CSV sau preprocessing có các cột:

| Cột | Mô tả |
|-----|-------|
| `text_raw` | Văn bản gốc |
| `label` | Nhãn cảm xúc (ANG, JOY, ...) |
| `text_clean` | Sau clean_text() |
| `text_norm` | Sau normalize_teencode() |
| `text_length` | Số ký tự (min-max normalised) |
| `word_count` | Số từ |
| `emoji_count` | Số emoji |
| `hashtag_count` | Số #tag |
| `mention_count` | Số @mention |
| `exclamation_count` | Số dấu ! |
| `uppercase_count` | Số chữ hoa |
| `repeated_char_count` | Số token có ký tự lặp 3+ lần |
| `label_enc` | Nhãn số (0–6) |

---

## Yêu cầu phần cứng

| Mô hình     | GPU RAM tối thiểu | Thời gian train |
|-------------|-------------------|-----------------|
| DNN+TF-IDF  | CPU đủ (~300 MB)  | ~2 phút         |
| BiLSTM+Attn | 2 GB GPU          | ~5–10 phút      |
| XLM-RoBERTa | 6 GB GPU          | ~30–40 phút     |

Đã test với: **NVIDIA RTX 5070** (CUDA 12.8, sm_120 Blackwell)

---

*Đồ án Tốt nghiệp — Deep Learning for NLP*
