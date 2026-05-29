"""
app.py — Streamlit demo cho dự án LSTM Short Text Classification.

Chạy:
    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LSTM Emotion Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 LSTM Emotion")
    st.caption("Phân loại cảm xúc văn bản ngắn tiếng Việt")
    st.divider()
    page = st.radio(
        "Chọn trang",
        ["🏠 Tổng quan", "🎯 Dự đoán", "📊 Kết quả", "📈 Training curves", "🔬 Ablation study"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Đề tài đồ án · Deep Learning NLP")

# ── Helpers ───────────────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "ANG": "😡", "DIS": "🤢", "FEA": "😨",
    "JOY": "😄", "NEU": "😐", "SAD": "😢", "SUR": "😲",
}
EMOTION_COLOR = {
    "ANG": "#FF4B4B", "DIS": "#8B5A2B", "FEA": "#9B59B6",
    "JOY": "#FFD700",  "NEU": "#95A5A6",  "SAD": "#3498DB", "SUR": "#2ECC71",
}

@st.cache_resource(show_spinner="Đang tải model BiLSTM...")
def load_lstm():
    from src.inference.predict import LSTMPredictor
    return LSTMPredictor()

@st.cache_resource(show_spinner="Đang tải model DNN...")
def load_dnn():
    from src.inference.predict import DNNPredictor
    return DNNPredictor()

@st.cache_resource(show_spinner="Đang tải model XLM-R...")
def load_xlmr():
    from src.inference.predict import XLMRPredictor
    return XLMRPredictor()

def load_comparison():
    p = ROOT / "outputs" / "reports" / "comparison.json"
    return json.load(open(p)) if p.exists() else {}

def load_ablation():
    p = ROOT / "outputs" / "logs" / "ablation_results.json"
    return json.load(open(p)) if p.exists() else {}

def load_history(name):
    p = ROOT / "outputs" / "logs" / f"{name}_history.json"
    return json.load(open(p)) if p.exists() else None

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Tổng quan":
    st.title("Xây dựng mô hình LSTM cho phân loại văn bản ngắn")
    st.caption("Ứng dụng BiLSTM + Bahdanau Attention phân loại 7 cảm xúc trên mạng xã hội tiếng Việt")
    st.divider()

    comp = load_comparison()

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mẫu dữ liệu", "2.726", "7 nhãn cảm xúc")
    c2.metric("BiLSTM Accuracy", f"{comp.get('BiLSTM', {}).get('accuracy', 0):.2%}" if comp else "—", "Mô hình đề xuất")
    c3.metric("XLM-R Accuracy", f"{comp.get('XLM-R', {}).get('accuracy', 0):.2%}" if comp else "—", "Upper bound")
    c4.metric("BiLSTM Params", f"{comp.get('BiLSTM', {}).get('num_params', 0)/1e6:.1f}M" if comp else "—", "Nhẹ hơn XLM-R 95x")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Kiến trúc BiLSTM + Attention")
        st.code("""
Input tokens [t₁, t₂, ..., tₙ]
    │
    ▼ Embedding(vocab, 128) + Dropout
    │
    ▼ BiLSTM(128→256, 2 layers)
    │  Forward  → h₁, h₂, ..., hₙ
    │  Backward ← hₙ, ..., h₂, h₁
    │
    ▼ Bahdanau Attention
    │  α = softmax(v·tanh(W·h))
    │  context = Σ αᵢ·hᵢ
    │
    ▼ Linear→LayerNorm→GELU→Dropout
    │
    ▼ Linear(256→7) — Logits
        """, language="text")

    with col2:
        st.subheader("7 nhãn cảm xúc")
        labels = {
            "😡 ANG": "Giận dữ", "🤢 DIS": "Ghê tởm", "😨 FEA": "Sợ hãi",
            "😄 JOY": "Hạnh phúc", "😐 NEU": "Trung tính",
            "😢 SAD": "Buồn bã",  "😲 SUR": "Ngạc nhiên",
        }
        for emoji_lbl, name in labels.items():
            st.write(f"**{emoji_lbl}** — {name}")

        if comp:
            st.divider()
            st.subheader("So sánh nhanh")
            import pandas as pd
            rows = [{"Model": k, "Accuracy": f"{v['accuracy']:.2%}", "F1": f"{v['f1']:.2%}"}
                    for k, v in comp.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Pipeline
    st.divider()
    st.subheader("Pipeline xử lý")
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, step in zip([p1,p2,p3,p4,p5], [
        ("📥", "Load data", "dataset.xlsx"),
        ("🧹", "Clean text", "URL/emoji/teencode"),
        ("📚", "Vocabulary", "word → index"),
        ("🧠", "BiLSTM", "train model"),
        ("📊", "Evaluate", "acc/F1/CM"),
    ]):
        col.markdown(f"<div style='text-align:center;font-size:2rem'>{step[0]}</div>", unsafe_allow_html=True)
        col.markdown(f"<div style='text-align:center'><b>{step[1]}</b><br><small>{step[2]}</small></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DỰ ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Dự đoán":
    st.title("🎯 Phân tích cảm xúc")

    col_input, col_model = st.columns([3, 1])
    with col_input:
        text = st.text_area(
            "Nhập văn bản tiếng Việt",
            placeholder="Ví dụ: Hôm nay tôi rất vui được gặp bạn bè!!!",
            height=120,
        )
    with col_model:
        st.write("")
        st.write("")
        model_choice = st.selectbox(
            "Chọn model",
            ["BiLSTM + Attention ★", "DNN + TF-IDF", "XLM-RoBERTa"],
        )
        show_attn = st.checkbox("Hiện attention weights", value=True,
                                disabled=("BiLSTM" not in model_choice))

    examples = st.expander("📌 Câu ví dụ")
    with examples:
        ex_cols = st.columns(4)
        sample_texts = [
            "Hôm nay vui lắm, được điểm cao nè!!!",
            "Chán quá không muốn làm gì cả...",
            "Tức quá đi! Sao lại như vậy được???",
            "Ôi trời ơi, không ngờ lại được thế này!",
        ]
        for col, t in zip(ex_cols, sample_texts):
            if col.button(t[:30] + "…", use_container_width=True):
                text = t
                st.rerun()

    if st.button("🔍 Phân tích", type="primary", disabled=not text.strip()):
        with st.spinner("Đang phân tích..."):
            try:
                if "BiLSTM" in model_choice:
                    predictor = load_lstm()
                elif "DNN" in model_choice:
                    predictor = load_dnn()
                else:
                    predictor = load_xlmr()

                result = predictor.predict(text)
                pred   = result["predicted_label"]
                name   = result["predicted_name"]
                conf   = result["confidence"]
                probs  = result["probabilities"]

                # Main result card
                st.divider()
                r1, r2 = st.columns([1, 2])
                with r1:
                    color = EMOTION_COLOR.get(pred, "#888")
                    emoji = EMOTION_EMOJI.get(pred, "")
                    st.markdown(
                        f"""<div style='background:{color}22;border:2px solid {color};
                        border-radius:12px;padding:24px;text-align:center'>
                        <div style='font-size:3rem'>{emoji}</div>
                        <div style='font-size:1.4rem;font-weight:bold;color:{color}'>{name}</div>
                        <div style='font-size:1rem;color:#666'>{pred}</div>
                        <div style='font-size:1.2rem;margin-top:8px'>{conf:.1%}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                with r2:
                    st.write("**Phân phối xác suất**")
                    import pandas as pd
                    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
                    for lbl, p in sorted_probs:
                        emoji = EMOTION_EMOJI.get(lbl, "")
                        color = EMOTION_COLOR.get(lbl, "#888")
                        st.markdown(
                            f"{emoji} **{lbl}** &nbsp;&nbsp;"
                            f"<div style='display:inline-block;background:{color};width:{p*200:.0f}px;"
                            f"height:16px;border-radius:3px;vertical-align:middle'></div>"
                            f"&nbsp; {p:.1%}",
                            unsafe_allow_html=True,
                        )

                # Attention weights
                if show_attn and "BiLSTM" in model_choice and hasattr(predictor, "get_attention_weights"):
                    weights = predictor.get_attention_weights(text)
                    if weights:
                        st.divider()
                        st.write("**Attention weights** — từ nào model tập trung vào")
                        import matplotlib.pyplot as plt
                        import numpy as np

                        tokens = list(weights.keys())
                        scores = np.array(list(weights.values()))
                        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                        fig, ax = plt.subplots(figsize=(min(14, len(tokens)*0.9+2), 2.5))
                        colors  = [plt.cm.YlOrRd(s) for s in scores]
                        ax.bar(range(len(tokens)), scores, color=colors, edgecolor="white", linewidth=0.5)
                        ax.set_xticks(range(len(tokens)))
                        ax.set_xticklabels(tokens, rotation=30, ha="right", fontsize=11)
                        ax.set_ylabel("Attention weight")
                        ax.set_ylim(0, 1.1)
                        ax.spines[["top","right"]].set_visible(False)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            except FileNotFoundError as e:
                st.error(f"Model chưa được train: {e}")
            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Kết quả":
    st.title("📊 Kết quả thực nghiệm")

    comp = load_comparison()
    if not comp:
        st.warning("Chưa có comparison.json — chạy `python main.py evaluate` trước.")
        st.stop()

    # Summary metrics
    cols = st.columns(len(comp))
    for col, (name, m) in zip(cols, comp.items()):
        delta = None
        if name == "BiLSTM":
            delta = "Proposed ★"
        elif name == "XLM-R":
            delta = "Upper bound"
        col.metric(name, f"{m['accuracy']:.2%}", delta)

    st.divider()

    # Full table
    st.subheader("Bảng so sánh đầy đủ")
    import pandas as pd
    rows = []
    for name, m in comp.items():
        rows.append({
            "Model":      name,
            "Accuracy":   f"{m['accuracy']:.4f}",
            "Precision":  f"{m['precision']:.4f}",
            "Recall":     f"{m['recall']:.4f}",
            "F1":         f"{m['f1']:.4f}",
            "Params":     f"{m['num_params']:,}",
            "ms/sample":  f"{m['inference_ms_per_sample']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Per-class F1 of BiLSTM
    pcf1 = comp.get("BiLSTM", {}).get("per_class_f1", {})
    if pcf1:
        st.divider()
        st.subheader("F1 theo từng lớp — BiLSTM+Attention")
        df_pcf1 = pd.DataFrame([
            {"Nhãn": k, "Tên": {"ANG":"Giận dữ","DIS":"Ghê tởm","FEA":"Sợ hãi",
                                  "JOY":"Hạnh phúc","NEU":"Trung tính",
                                  "SAD":"Buồn bã","SUR":"Ngạc nhiên"}.get(k,k),
             "F1": v}
            for k, v in sorted(pcf1.items(), key=lambda x: -x[1])
        ])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 3))
        colors = [EMOTION_COLOR.get(r["Nhãn"], "#888") for _, r in df_pcf1.iterrows()]
        bars   = ax.barh(df_pcf1["Nhãn"], df_pcf1["F1"], color=colors)
        ax.set_xlim(0, 1)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
        ax.set_xlabel("F1 Score")
        ax.spines[["top","right"]].set_visible(False)
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Confusion matrices
    st.divider()
    st.subheader("Confusion matrices")
    cm_files = {
        "BiLSTM": ROOT / "outputs/figures/bilstm_cm.png",
        "DNN+TF-IDF": ROOT / "outputs/figures/dnn_tf_idf_cm.png",
        "XLM-R": ROOT / "outputs/figures/xlm_r_cm.png",
    }
    c1, c2, c3 = st.columns(3)
    for col, (name, path) in zip([c1, c2, c3], cm_files.items()):
        with col:
            st.caption(name)
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Chưa có hình")

    # Model comparison figure
    comp_fig = ROOT / "outputs/figures/model_comparison.png"
    if comp_fig.exists():
        st.divider()
        st.subheader("So sánh tổng thể")
        st.image(str(comp_fig), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training curves":
    st.title("📈 Training curves")

    model_tab = st.tabs(["BiLSTM+Attention ★", "DNN+TF-IDF", "XLM-RoBERTa"])

    for tab, (label, key) in zip(model_tab, [
        ("BiLSTM+Attention", "lstm"),
        ("DNN+TF-IDF", "dnn"),
        ("XLM-RoBERTa", "xlmr"),
    ]):
        with tab:
            hist = load_history(key)
            if not hist:
                st.info(f"Chưa có {key}_history.json")
                continue

            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            epochs = range(1, len(hist["train_loss"]) + 1)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(epochs, hist["train_loss"], "b-o", ms=4, label="Train")
            ax1.plot(epochs, hist["val_loss"],   "r-o", ms=4, label="Val")
            ax1.set_title(f"{label} — Loss")
            ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
            ax1.legend(); ax1.grid(alpha=0.3)

            ax2.plot(epochs, hist["train_acc"], "b-o", ms=4, label="Train")
            ax2.plot(epochs, hist["val_acc"],   "r-o", ms=4, label="Val")
            ax2.set_title(f"{label} — Accuracy")
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
            ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax2.legend(); ax2.grid(alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Epochs", len(hist["train_loss"]))
            c2.metric("Best epoch", hist.get("best_epoch", "—"))
            c3.metric("Test accuracy", f"{hist.get('test_acc', 0):.2%}" if hist.get('test_acc') else "—")
            c4.metric("Test F1", f"{hist.get('test_f1', 0):.2%}" if hist.get('test_f1') else "—")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Ablation study":
    st.title("🔬 Ablation Study")
    st.caption("Đo lường đóng góp của từng component trong BiLSTM+Attention")

    abl = load_ablation()
    if not abl:
        st.warning("Chưa có ablation_results.json")
        st.stop()

    # Summary table
    import pandas as pd
    rows = [
        {
            "Variant": name + (" ★" if "Attention" in name else ""),
            "Bidirectional": "✅" if "BiLSTM" in name else "❌",
            "Attention":     "✅" if "Attention" in name else "❌",
            "Accuracy":      f"{m['accuracy']:.4f}",
            "F1":            f"{m['f1']:.4f}",
            "Params":        f"{m['num_params']:,}",
        }
        for name, m in abl.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Bar chart
    st.divider()
    import matplotlib.pyplot as plt
    import numpy as np

    names  = [n.replace(" (proposed)", "★") for n in abl.keys()]
    accs   = [m["accuracy"] for m in abl.values()]
    f1s    = [m["f1"]       for m in abl.values()]

    x, w = np.arange(len(names)), 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + w/2, f1s,  w, label="F1",       color="#DD8452")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Ablation Study — Contribution of Each Component", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Insight
    st.divider()
    st.subheader("Nhận xét")

    vals = list(abl.values())
    if len(vals) >= 3:
        delta_bi   = vals[1]["accuracy"] - vals[0]["accuracy"]
        delta_attn = vals[2]["accuracy"] - vals[1]["accuracy"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Đóng góp của Bidirectionality",
                f"{delta_bi:+.2%}",
                f"LSTM {vals[0]['accuracy']:.2%} → BiLSTM {vals[1]['accuracy']:.2%}",
                delta_color="normal",
            )
        with col2:
            st.metric(
                "Đóng góp của Attention",
                f"{delta_attn:+.2%}",
                f"BiLSTM {vals[1]['accuracy']:.2%} → +Attn {vals[2]['accuracy']:.2%}",
                delta_color="normal" if delta_attn >= 0 else "inverse",
            )

        st.info(
            "**Phân tích**: Bidirectionality cải thiện đáng kể (+{:.2%}) bằng cách đọc "
            "văn bản theo cả 2 chiều. Attention có đóng góp {}, phù hợp với văn bản "
            "ngắn khi BiLSTM đã nắm bắt đủ ngữ cảnh toàn cục.".format(
                delta_bi,
                "nhỏ" if abs(delta_attn) < 0.01 else f"{delta_attn:+.2%}",
            )
        )
