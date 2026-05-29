"""
app.py — Professional Streamlit demo
Xây dựng mô hình học sâu LSTM cho phân loại văn bản ngắn tiếng Việt
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vietnamese Emotion Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
CLR = {
    "primary":    "#1D4ED8",
    "primary_lt": "#EFF6FF",
    "sidebar":    "#0F172A",
    "surface":    "#FFFFFF",
    "bg":         "#F8FAFC",
    "border":     "#E2E8F0",
    "text":       "#0F172A",
    "muted":      "#64748B",
    "bilstm":     "#1D4ED8",
    "dnn":        "#D97706",
    "xlmr":       "#059669",
    "ANG": "#DC2626", "DIS": "#7C3AED", "FEA": "#9333EA",
    "JOY": "#D97706", "NEU": "#475569",  "SAD": "#2563EB", "SUR": "#059669",
}

EMOTION_EMOJI = {
    "ANG":"😡","DIS":"🤢","FEA":"😨","JOY":"😄","NEU":"😐","SAD":"😢","SUR":"😲",
}
EMOTION_NAME = {
    "ANG":"Giận dữ","DIS":"Ghê tởm","FEA":"Sợ hãi",
    "JOY":"Hạnh phúc","NEU":"Trung tính","SAD":"Buồn bã","SUR":"Ngạc nhiên",
}
MODEL_COLOR = {"BiLSTM": CLR["bilstm"], "DNN+TF-IDF": CLR["dnn"], "XLM-R": CLR["xlmr"]}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stApp {{ font-family: 'Inter', -apple-system, sans-serif !important; }}
.stApp {{ background: {CLR['bg']}; }}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 2rem 2.5rem 4rem !important; max-width: 1200px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {CLR['sidebar']} !important;
    border-right: none !important;
}}
[data-testid="stSidebar"] * {{
    color: #CBD5E1 !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: #94A3B8 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    display: block;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(255,255,255,0.06) !important;
    color: #F1F5F9 !important;
}}
[data-testid="stSidebarContent"] h1,
[data-testid="stSidebarContent"] h2,
[data-testid="stSidebarContent"] h3 {{
    color: #F1F5F9 !important;
}}
[data-testid="stSidebarContent"] hr {{
    border-color: #1E293B !important;
}}

/* ── Cards ── */
.card {{
    background: {CLR['surface']};
    border: 1px solid {CLR['border']};
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}}
.card-sm {{
    background: {CLR['surface']};
    border: 1px solid {CLR['border']};
    border-radius: 10px;
    padding: 1.25rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}}

/* ── Metric card ── */
.metric-box {{
    background: {CLR['surface']};
    border: 1px solid {CLR['border']};
    border-radius: 10px;
    padding: 1.25rem 1rem;
    text-align: center;
}}
.metric-box .val {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {CLR['primary']};
    line-height: 1.2;
}}
.metric-box .lbl {{
    font-size: 0.78rem;
    font-weight: 500;
    color: {CLR['muted']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}}
.metric-box .sub {{
    font-size: 0.72rem;
    color: #94A3B8;
    margin-top: 0.25rem;
}}

/* ── Section header ── */
.section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {CLR['text']};
    margin: 0 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.section-title::after {{
    content: '';
    flex: 1;
    height: 1px;
    background: {CLR['border']};
    margin-left: 0.5rem;
}}

/* ── Page title ── */
.page-header {{
    margin-bottom: 1.75rem;
}}
.page-header h1 {{
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {CLR['text']} !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}}
.page-header p {{
    color: {CLR['muted']};
    font-size: 0.9rem;
    margin: 0.4rem 0 0;
}}

/* ── Result card ── */
.result-card {{
    border-radius: 14px;
    padding: 1.75rem;
    text-align: center;
    border: 2px solid;
}}
.result-card .emoji {{ font-size: 2.75rem; line-height: 1; }}
.result-card .label {{ font-size: 1.4rem; font-weight: 700; margin: 0.5rem 0 0.2rem; }}
.result-card .code  {{ font-size: 0.8rem; opacity: 0.7; font-weight: 500; }}
.result-card .conf  {{ font-size: 1rem; margin-top: 0.75rem; opacity: 0.85; }}

/* ── Probability bar ── */
.prob-row {{ margin: 0.4rem 0; display: flex; align-items: center; gap: 0.75rem; }}
.prob-label {{ width: 2.5rem; font-size: 0.8rem; font-weight: 600; color: {CLR['text']}; }}
.prob-bar-bg {{ flex: 1; background: #F1F5F9; border-radius: 99px; height: 10px; overflow: hidden; }}
.prob-bar-fill {{ height: 100%; border-radius: 99px; transition: width 0.4s ease; }}
.prob-pct {{ width: 3rem; text-align: right; font-size: 0.8rem; color: {CLR['muted']}; font-variant-numeric: tabular-nums; }}

/* ── Table ── */
.styled-table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
.styled-table th {{
    background: {CLR['bg']}; color: {CLR['muted']};
    font-weight: 600; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.05em; padding: 0.75rem 1rem;
    border-bottom: 2px solid {CLR['border']};
    text-align: left;
}}
.styled-table td {{
    padding: 0.875rem 1rem;
    border-bottom: 1px solid {CLR['border']};
    color: {CLR['text']};
}}
.styled-table tr:last-child td {{ border-bottom: none; }}
.styled-table tr:hover td {{ background: {CLR['bg']}; }}
.badge {{
    display: inline-block; padding: 0.2rem 0.6rem;
    border-radius: 99px; font-size: 0.72rem; font-weight: 600;
}}

/* ── Attention token ── */
.token-list {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.75rem; }}
.token {{
    padding: 0.3rem 0.65rem; border-radius: 6px;
    font-size: 0.85rem; font-weight: 500; border: 1px solid rgba(0,0,0,0.08);
}}

/* ── Tab ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {CLR['bg']};
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
    border: 1px solid {CLR['border']};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}}

/* ── Button ── */
.stButton > button {{
    background: {CLR['primary']} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

/* ── Text area ── */
.stTextArea textarea {{
    border-radius: 10px !important;
    border: 1.5px solid {CLR['border']} !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
}}
.stTextArea textarea:focus {{
    border-color: {CLR['primary']} !important;
    box-shadow: 0 0 0 3px rgba(29,78,216,0.08) !important;
}}

/* ── Selectbox ── */
.stSelectbox > div > div {{
    border-radius: 8px !important;
    border-color: {CLR['border']} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.25rem'>
        <div style='font-size:1.5rem; margin-bottom:0.3rem'>🧠</div>
        <div style='color:#F1F5F9; font-size:1rem; font-weight:700; line-height:1.3'>
            Emotion Classifier
        </div>
        <div style='color:#64748B; font-size:0.78rem; margin-top:0.25rem'>
            Vietnamese Social Media NLP
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🏠  Tổng quan", "🎯  Dự đoán", "📊  Kết quả", "📈  Training", "🔬  Ablation"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin-top:auto; padding-top:2rem'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1E293B; border-radius:10px; padding:1rem; margin-top:2rem'>
        <div style='color:#94A3B8; font-size:0.72rem; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.75rem'>
            Về dự án
        </div>
        <div style='color:#CBD5E1; font-size:0.8rem; line-height:1.6'>
            Đề tài tốt nghiệp<br>
            Học sâu cho NLP<br>
            BiLSTM · DNN · XLM-R
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải BiLSTM model…")
def load_lstm():
    from src.inference.predict import LSTMPredictor
    return LSTMPredictor()

@st.cache_resource(show_spinner="Đang tải DNN model…")
def load_dnn():
    from src.inference.predict import DNNPredictor
    return DNNPredictor()

@st.cache_resource(show_spinner="Đang tải XLM-R model…")
def load_xlmr():
    from src.inference.predict import XLMRPredictor
    return XLMRPredictor()

@st.cache_data
def get_comparison():
    p = ROOT / "outputs/reports/comparison.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def get_ablation():
    p = ROOT / "outputs/logs/ablation_results.json"
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def get_history(key):
    p = ROOT / f"outputs/logs/{key}_history.json"
    return json.load(open(p)) if p.exists() else None


# ── Chart helpers ──────────────────────────────────────────────────────────────
def apply_chart_style(fig, ax_list=None):
    fig.patch.set_facecolor("white")
    axes = ax_list or fig.get_axes()
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E2E8F0")
        ax.spines["bottom"].set_color("#E2E8F0")
        ax.tick_params(colors="#64748B", labelsize=9)
        ax.xaxis.label.set_color("#64748B")
        ax.yaxis.label.set_color("#64748B")
        ax.title.set_color("#0F172A")
        ax.grid(axis="y", color="#F1F5F9", linewidth=1)
        ax.set_axisbelow(True)


def metric_card(value, label, sub=""):
    return f"""<div class="metric-box">
        <div class="val">{value}</div>
        <div class="lbl">{label}</div>
        {'<div class="sub">'+sub+'</div>' if sub else ''}
    </div>"""


def section(icon, title):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TỔNG QUAN
# ══════════════════════════════════════════════════════════════════════════════
if "Tổng quan" in page:
    st.markdown("""
    <div class="page-header">
        <h1>Phân loại cảm xúc văn bản ngắn tiếng Việt</h1>
        <p>Bidirectional LSTM với Bahdanau Attention · 7 nhãn cảm xúc · 2.726 mẫu mạng xã hội</p>
    </div>
    """, unsafe_allow_html=True)

    comp = get_comparison()

    # Metric row
    m1, m2, m3, m4, m5 = st.columns(5)
    bilstm_m = comp.get("BiLSTM", {})
    xlmr_m   = comp.get("XLM-R", {})

    cards = [
        ("2.726", "Mẫu dữ liệu", "Social media VN"),
        ("7",     "Nhãn cảm xúc", "ANG DIS FEA JOY…"),
        (f"{bilstm_m.get('accuracy',0):.1%}" if bilstm_m else "—",
         "BiLSTM Accuracy", "Mô hình đề xuất ★"),
        (f"{xlmr_m.get('accuracy',0):.1%}" if xlmr_m else "—",
         "XLM-R Accuracy", "Upper-bound"),
        (f"{bilstm_m.get('num_params',0)/1e6:.1f}M" if bilstm_m else "—",
         "BiLSTM Params", "vs 279M của XLM-R"),
    ]
    for col, (v, l, s) in zip([m1,m2,m3,m4,m5], cards):
        col.markdown(metric_card(v, l, s), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Architecture + Labels
    left, right = st.columns([5, 4], gap="large")

    with left:
        section("🏗️", "Kiến trúc mô hình đề xuất")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.code("""Đầu vào: Văn bản tiếng Việt (sau chuẩn hoá teencode)
    │
    ├─ Embedding  (vocab × 128d)  +  Dropout(0.3)
    │
    ├─ Bidirectional LSTM  ×  2 layers
    │    Forward  →  h₁→  h₂→  …  hₙ→
    │    Backward ←  hₙ←  …   h₂←  h₁←
    │    Output:  [hᵢ→ ‖ hᵢ←]  (512d)
    │
    ├─ Bahdanau Attention
    │    score = v · tanh(W·h)
    │    α     = softmax(score)        [0,1]
    │    context = Σ αᵢ·hᵢ            (512d)
    │
    └─ Linear(512→256) → LayerNorm → GELU
       → Dropout → Linear(256→7) → Logits""", language="text")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        section("🎭", "Nhãn cảm xúc")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for code, name in EMOTION_NAME.items():
            emoji = EMOTION_EMOJI[code]
            color = CLR.get(code, "#888")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.75rem;'
                f'padding:0.5rem 0;border-bottom:1px solid {CLR["border"]}">'
                f'<span style="font-size:1.3rem">{emoji}</span>'
                f'<span class="badge" style="background:{color}18;color:{color}">{code}</span>'
                f'<span style="font-size:0.875rem;color:{CLR["text"]};font-weight:500">{name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline steps
    section("⚙️", "Pipeline xử lý dữ liệu")
    steps = [
        ("📥", "Load data", "dataset.xlsx · 2.726 mẫu"),
        ("🧹", "Clean text", "URL · @mention · #tag"),
        ("🔤", "Teencode", "80+ quy tắc chuẩn hoá"),
        ("📚", "Vocabulary", f"word → index · ~9k tokens"),
        ("🧠", "Train LSTM", "BiLSTM · 2 layers · Attn"),
        ("📊", "Evaluate", "Accuracy · F1 · CM"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(
            f'<div class="card-sm" style="text-align:center">'
            f'<div style="font-size:1.6rem;margin-bottom:0.4rem">{icon}</div>'
            f'<div style="font-weight:600;font-size:0.875rem;color:{CLR["text"]}">{title}</div>'
            f'<div style="font-size:0.75rem;color:{CLR["muted"]};margin-top:0.25rem">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DỰ ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════
elif "Dự đoán" in page:
    st.markdown("""
    <div class="page-header">
        <h1>Phân tích cảm xúc</h1>
        <p>Nhập văn bản tiếng Việt và chọn mô hình để phân tích cảm xúc</p>
    </div>
    """, unsafe_allow_html=True)

    # Input section
    col_a, col_b = st.columns([3, 1], gap="large")
    with col_a:
        text = st.text_area(
            "Văn bản đầu vào",
            placeholder="Nhập câu tiếng Việt bất kỳ… VD: Hôm nay tôi rất vui được gặp bạn bè!!!",
            height=130,
            label_visibility="collapsed",
        )
    with col_b:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        model_key = st.selectbox(
            "Mô hình",
            ["BiLSTM + Attention", "DNN + TF-IDF", "XLM-RoBERTa"],
            label_visibility="visible",
        )
        show_attn = st.toggle("Attention weights", value=True,
                              disabled="BiLSTM" not in model_key)
        predict_btn = st.button("Phân tích →", use_container_width=True)

    # Quick examples
    with st.expander("💡 Câu ví dụ theo từng cảm xúc"):
        ec1, ec2, ec3, ec4 = st.columns(4)
        samples = [
            ("😄 Hạnh phúc", "Hôm nay vui lắm, được điểm cao nè!!!"),
            ("😢 Buồn bã",   "Chán quá không muốn làm gì cả..."),
            ("😡 Giận dữ",   "Tức quá đi! Sao lại như vậy được???"),
            ("😲 Ngạc nhiên","Ôi không ngờ lại được quà như vậy!"),
        ]
        for col, (label, sample) in zip([ec1,ec2,ec3,ec4], samples):
            if col.button(label, use_container_width=True, key=f"ex_{label}"):
                st.session_state["_sample_text"] = sample
                st.rerun()

    if "_sample_text" in st.session_state:
        text = st.session_state.pop("_sample_text")
        st.rerun()

    # Prediction
    if predict_btn and text.strip():
        with st.spinner("Đang phân tích…"):
            try:
                predictor = (load_lstm() if "BiLSTM" in model_key
                             else load_dnn() if "DNN" in model_key
                             else load_xlmr())
                result = predictor.predict(text)

                pred   = result["predicted_label"]
                conf   = result["confidence"]
                probs  = result["probabilities"]
                color  = CLR.get(pred, "#888")
                emoji  = EMOTION_EMOJI.get(pred, "")
                name   = EMOTION_NAME.get(pred, pred)

                st.markdown("<br>", unsafe_allow_html=True)
                r1, r2, r3 = st.columns([2, 3, 3], gap="large")

                # Result card
                with r1:
                    section("✨", "Kết quả")
                    st.markdown(
                        f'<div class="result-card" '
                        f'style="background:{color}0D;border-color:{color}40">'
                        f'<div class="emoji">{emoji}</div>'
                        f'<div class="label" style="color:{color}">{name}</div>'
                        f'<div class="code" style="color:{color}">{pred}</div>'
                        f'<div class="conf">Độ tin cậy: <b>{conf:.1%}</b></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Probability bars
                with r2:
                    section("📊", "Phân phối xác suất")
                    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
                    bars_html = '<div style="margin-top:0.5rem">'
                    for lbl, p in sorted_probs:
                        c = CLR.get(lbl, "#888")
                        bars_html += (
                            f'<div class="prob-row">'
                            f'<span class="prob-label">'
                            f'{EMOTION_EMOJI.get(lbl,"")} {lbl}</span>'
                            f'<div class="prob-bar-bg">'
                            f'<div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{c}"></div>'
                            f'</div>'
                            f'<span class="prob-pct">{p:.1%}</span>'
                            f'</div>'
                        )
                    bars_html += '</div>'
                    st.markdown(bars_html, unsafe_allow_html=True)

                # Preprocessing preview
                with r3:
                    section("🔍", "Tiền xử lý")
                    from src.preprocessing.clean_text import clean_text
                    from src.preprocessing.teencode_normalize import normalize_teencode
                    cleaned = clean_text(text, remove_emoji=False)
                    normed  = normalize_teencode(cleaned)
                    for step, val in [("Gốc", text), ("Làm sạch", cleaned), ("Chuẩn hoá", normed)]:
                        st.markdown(
                            f'<div style="margin-bottom:0.75rem">'
                            f'<div style="font-size:0.72rem;font-weight:600;color:{CLR["muted"]};'
                            f'text-transform:uppercase;letter-spacing:0.05em;'
                            f'margin-bottom:0.25rem">{step}</div>'
                            f'<div style="background:{CLR["bg"]};border:1px solid {CLR["border"]};'
                            f'border-radius:6px;padding:0.5rem 0.75rem;font-size:0.875rem;'
                            f'color:{CLR["text"]};word-break:break-word">{val}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                # Attention heatmap
                if show_attn and "BiLSTM" in model_key and hasattr(predictor, "get_attention_weights"):
                    weights = predictor.get_attention_weights(text)
                    if weights:
                        st.markdown("<br>", unsafe_allow_html=True)
                        section("🎯", "Attention weights — từ nào model tập trung vào")
                        tokens = list(weights.keys())
                        scores = np.array(list(weights.values()))
                        s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                        import matplotlib.colors as mcolors
                        cmap = plt.cm.Blues
                        tokens_html = '<div class="token-list">'
                        for tok, raw_s, s in zip(tokens, scores, s_norm):
                            rgba   = cmap(0.25 + s * 0.65)
                            bg_hex = mcolors.to_hex(rgba)
                            fg     = "white" if s > 0.55 else CLR["text"]
                            tokens_html += (
                                f'<span class="token" '
                                f'style="background:{bg_hex};color:{fg};'
                                f'font-size:{0.8+s*0.25:.2f}rem">'
                                f'{tok}</span>'
                            )
                        tokens_html += '</div>'
                        st.markdown('<div class="card">' + tokens_html + '</div>',
                                    unsafe_allow_html=True)

            except FileNotFoundError as e:
                st.error(f"⚠️ Model chưa được train hoặc không tìm thấy file: `{e}`")
            except Exception as e:
                st.error(f"Lỗi: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — KẾT QUẢ
# ══════════════════════════════════════════════════════════════════════════════
elif "Kết quả" in page:
    st.markdown("""
    <div class="page-header">
        <h1>Kết quả thực nghiệm</h1>
        <p>So sánh hiệu suất 3 mô hình trên tập kiểm tra (test set · 408 mẫu)</p>
    </div>
    """, unsafe_allow_html=True)

    comp = get_comparison()
    if not comp:
        st.warning("Chưa có dữ liệu — chạy `python main.py evaluate`"); st.stop()

    # Metric summary row
    cols = st.columns(len(comp), gap="medium")
    for col, (name, m) in zip(cols, comp.items()):
        c = MODEL_COLOR.get(name, CLR["primary"])
        tag = "Đề xuất ★" if name == "BiLSTM" else ("Upper bound" if name == "XLM-R" else "Baseline")
        col.markdown(
            f'<div class="metric-box" style="border-top:3px solid {c}">'
            f'<div class="val" style="color:{c}">{m["accuracy"]:.2%}</div>'
            f'<div class="lbl">{name}</div>'
            f'<div class="sub">{tag}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📋 Bảng tổng hợp", "📉 Confusion Matrix", "📊 Biểu đồ"])

    # ── Tab 1: Table ──────────────────────────────────────────────────────────
    with tab1:
        rows = ""
        for name, m in comp.items():
            c = MODEL_COLOR.get(name, "#888")
            star = " ★" if name == "BiLSTM" else ""
            p_str = f"{m['num_params']:,}"
            rows += (
                f"<tr>"
                f'<td><span class="badge" style="background:{c}18;color:{c};font-size:0.8rem">'
                f'{name+star}</span></td>'
                f"<td><b>{m['accuracy']:.4f}</b></td>"
                f"<td>{m['precision']:.4f}</td>"
                f"<td>{m['recall']:.4f}</td>"
                f"<td>{m['f1']:.4f}</td>"
                f"<td style='color:{CLR['muted']}'>{p_str}</td>"
                f"<td style='color:{CLR['muted']}'>{m['inference_ms_per_sample']:.2f} ms</td>"
                f"</tr>"
            )
        table_html = (
            f'<div class="card"><table class="styled-table">'
            f"<thead><tr>"
            f"<th>Mô hình</th><th>Accuracy</th><th>Precision</th>"
            f"<th>Recall</th><th>F1</th><th>Params</th><th>Inference</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        # Per-class F1
        pcf1 = comp.get("BiLSTM", {}).get("per_class_f1", {})
        if pcf1:
            st.markdown("<br>", unsafe_allow_html=True)
            section("🎯", "F1 theo từng lớp — BiLSTM+Attention")
            items = sorted(pcf1.items(), key=lambda x: -x[1])
            bars_html = '<div class="card"><div style="display:flex;flex-direction:column;gap:0.6rem">'
            for lbl, f1 in items:
                c = CLR.get(lbl, "#888")
                bars_html += (
                    f'<div style="display:flex;align-items:center;gap:1rem">'
                    f'<span style="width:2.5rem;font-size:0.85rem;font-weight:600">'
                    f'{EMOTION_EMOJI.get(lbl,"")} {lbl}</span>'
                    f'<div style="flex:1;background:#F1F5F9;border-radius:99px;height:12px;overflow:hidden">'
                    f'<div style="width:{f1*100:.1f}%;background:{c};height:100%;border-radius:99px"></div>'
                    f'</div>'
                    f'<span style="width:3.5rem;text-align:right;font-size:0.82rem;'
                    f'color:{CLR["text"]};font-weight:600">{f1:.3f}</span>'
                    f'</div>'
                )
            bars_html += '</div></div>'
            st.markdown(bars_html, unsafe_allow_html=True)

    # ── Tab 2: Confusion Matrix ───────────────────────────────────────────────
    with tab2:
        cm_map = {
            "BiLSTM":     "bilstm_cm.png",
            "DNN+TF-IDF": "dnn_tf_idf_cm.png",
            "XLM-R":      "xlm_r_cm.png",
        }
        cols = st.columns(3, gap="medium")
        for col, (name, fname) in zip(cols, cm_map.items()):
            p = ROOT / "outputs/figures" / fname
            c = MODEL_COLOR.get(name, "#888")
            with col:
                st.markdown(
                    f'<div style="text-align:center;margin-bottom:0.5rem">'
                    f'<span class="badge" style="background:{c}18;color:{c}">{name}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if p.exists():
                    st.image(str(p), use_container_width=True)
                else:
                    st.info("Chưa có hình")

    # ── Tab 3: Charts ──────────────────────────────────────────────────────────
    with tab3:
        p = ROOT / "outputs/figures/model_comparison.png"
        if p.exists():
            st.image(str(p), use_container_width=True)

        # Interactive radar chart với plotly
        if comp:
            section("📡", "Radar chart — so sánh đa chiều")
            cats = ["Accuracy", "Precision", "Recall", "F1"]
            fig_radar = go.Figure()
            for name, m in comp.items():
                vals = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
                c    = MODEL_COLOR.get(name, "#888")
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=cats + [cats[0]],
                    fill="toself",
                    name=name,
                    line_color=c,
                    fillcolor=c,
                    opacity=0.25,
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0.7, 0.85],
                                    tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=11, color="#0F172A")),
                    bgcolor="white",
                ),
                showlegend=True,
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(l=40, r=40, t=30, b=30),
                height=360,
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif "Training" in page:
    st.markdown("""
    <div class="page-header">
        <h1>Training curves</h1>
        <p>Diễn tiến loss và accuracy qua từng epoch trong quá trình huấn luyện</p>
    </div>
    """, unsafe_allow_html=True)

    tab_labels = ["BiLSTM + Attention ★", "DNN + TF-IDF", "XLM-RoBERTa"]
    hist_keys  = ["lstm", "dnn", "xlmr"]
    model_clrs = [CLR["bilstm"], CLR["dnn"], CLR["xlmr"]]
    tabs       = st.tabs(tab_labels)

    for tab, key, mclr in zip(tabs, hist_keys, model_clrs):
        with tab:
            hist = get_history(key)
            if not hist:
                st.info(f"Chưa có {key}_history.json"); continue

            epochs = list(range(1, len(hist["train_loss"]) + 1))

            # Plotly interactive
            fig = go.Figure()
            for y_key, dash, name in [
                ("train_loss", "solid",  "Train loss"),
                ("val_loss",   "dot",    "Val loss"),
            ]:
                fig.add_trace(go.Scatter(
                    x=epochs, y=hist[y_key], name=name,
                    mode="lines+markers",
                    line=dict(width=2.5, dash=dash, color=mclr if "train" in y_key else "#94A3B8"),
                    marker=dict(size=5),
                ))
            fig.update_layout(
                title=dict(text="Loss per Epoch", font=dict(size=13, color="#0F172A")),
                xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F1F5F9",
                           tickfont=dict(size=10), title_font=dict(size=11)),
                yaxis=dict(title="Cross-Entropy Loss", showgrid=True,
                           gridcolor="#F1F5F9", tickfont=dict(size=10),
                           title_font=dict(size=11)),
                paper_bgcolor="white", plot_bgcolor="white",
                height=320, margin=dict(l=50, r=20, t=45, b=40),
                legend=dict(font=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            for y_key, name in [("train_acc","Train acc"), ("val_acc","Val acc")]:
                fig2.add_trace(go.Scatter(
                    x=epochs, y=hist[y_key], name=name,
                    mode="lines+markers",
                    line=dict(width=2.5,
                              color=mclr if "train" in y_key else "#94A3B8"),
                    marker=dict(size=5),
                    hovertemplate="%{y:.2%}",
                ))
            fig2.update_layout(
                title=dict(text="Accuracy per Epoch", font=dict(size=13, color="#0F172A")),
                xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F1F5F9",
                           tickfont=dict(size=10), title_font=dict(size=11)),
                yaxis=dict(title="Accuracy", showgrid=True, gridcolor="#F1F5F9",
                           tickformat=".0%", tickfont=dict(size=10),
                           title_font=dict(size=11)),
                paper_bgcolor="white", plot_bgcolor="white",
                height=320, margin=dict(l=50, r=20, t=45, b=40),
                legend=dict(font=dict(size=10)),
                hovermode="x unified",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(metric_card(len(hist["train_loss"]), "Epochs trained"), unsafe_allow_html=True)
            c2.markdown(metric_card(hist.get("best_epoch","—"), "Best epoch"),  unsafe_allow_html=True)
            c3.markdown(metric_card(f"{hist.get('test_acc',0):.2%}" if hist.get('test_acc') else "—",
                                    "Test accuracy"), unsafe_allow_html=True)
            c4.markdown(metric_card(f"{hist.get('test_f1',0):.2%}" if hist.get('test_f1') else "—",
                                    "Test F1"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABLATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Ablation" in page:
    st.markdown("""
    <div class="page-header">
        <h1>Ablation Study</h1>
        <p>Phân tích đóng góp của từng thành phần trong kiến trúc BiLSTM+Attention</p>
    </div>
    """, unsafe_allow_html=True)

    abl = get_ablation()
    if not abl:
        st.warning("Chưa có ablation_results.json"); st.stop()

    vals = list(abl.values())

    # Contribution metrics
    if len(vals) >= 3:
        delta_bi   = vals[1]["accuracy"] - vals[0]["accuracy"]
        delta_attn = vals[2]["accuracy"] - vals[1]["accuracy"]

        c1, c2, c3 = st.columns(3, gap="medium")
        for col, (title, base, result, delta) in zip([c1,c2,c3], [
            ("LSTM baseline",     None,            vals[0]["accuracy"], None),
            ("+ Bidirectionality", vals[0]["accuracy"], vals[1]["accuracy"], delta_bi),
            ("+ Attention",        vals[1]["accuracy"], vals[2]["accuracy"], delta_attn),
        ]):
            color = (CLR["xlmr"] if delta and delta > 0.005
                     else CLR["dnn"] if delta and delta < -0.002
                     else CLR["primary"])
            col.markdown(
                f'<div class="metric-box" style="border-top:3px solid {color}">'
                f'<div class="val" style="color:{color}">{result:.2%}</div>'
                f'<div class="lbl">{title}</div>'
                f'<div class="sub">{"Δ "+f"{delta:+.2%}" if delta is not None else "Baseline"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Plotly grouped bar
    section("📊", "So sánh Accuracy và F1")
    names_abl = [n.replace(" (proposed)", " ★") for n in abl.keys()]
    accs = [m["accuracy"] for m in abl.values()]
    f1s  = [m["f1"]       for m in abl.values()]
    colors_abl = ["#94A3B8", "#60A5FA", CLR["bilstm"]]

    fig_abl = go.Figure()
    fig_abl.add_trace(go.Bar(
        name="Accuracy", x=names_abl, y=accs,
        marker_color=colors_abl, text=[f"{v:.3f}" for v in accs],
        textposition="outside", offsetgroup=0,
    ))
    fig_abl.add_trace(go.Bar(
        name="F1", x=names_abl, y=f1s,
        marker_color=[c + "99" for c in colors_abl],
        text=[f"{v:.3f}" for v in f1s],
        textposition="outside", offsetgroup=1,
    ))
    fig_abl.update_layout(
        barmode="group",
        yaxis=dict(range=[0.72, 0.82], tickformat=".0%",
                   title="Score", showgrid=True, gridcolor="#F1F5F9",
                   title_font=dict(size=11), tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
        paper_bgcolor="white", plot_bgcolor="white",
        height=380, margin=dict(l=50, r=20, t=20, b=40),
        legend=dict(font=dict(size=11)),
        bargap=0.2, bargroupgap=0.05,
    )
    st.plotly_chart(fig_abl, use_container_width=True)

    # Detail table
    section("📋", "Bảng chi tiết")
    rows = ""
    comps = [("❌","❌"),("✅","❌"),("✅","✅")]
    for (name, m), (bi, attn) in zip(abl.items(), comps):
        star  = " ★" if "Attention" in name else ""
        c     = CLR["bilstm"] if "Attention" in name else CLR["primary"] if "BiLSTM" in name else CLR["muted"]
        rows += (
            f"<tr>"
            f'<td><span class="badge" style="background:{c}18;color:{c}">{name+star}</span></td>'
            f"<td style='text-align:center'>{bi}</td>"
            f"<td style='text-align:center'>{attn}</td>"
            f"<td><b>{m['accuracy']:.4f}</b></td>"
            f"<td>{m['f1']:.4f}</td>"
            f"<td style='color:{CLR['muted']}'>{m['num_params']:,}</td>"
            f"</tr>"
        )
    st.markdown(
        f'<div class="card"><table class="styled-table">'
        f"<thead><tr><th>Variant</th><th>BiDirectional</th><th>Attention</th>"
        f"<th>Accuracy</th><th>F1</th><th>Params</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # Insight box
    if len(vals) >= 3:
        st.markdown("<br>", unsafe_allow_html=True)
        insight = (
            f"**Bidirectionality** cải thiện accuracy **{delta_bi:+.2%}** bằng cách đọc văn bản "
            f"theo cả hai chiều, giúp mỗi từ hiểu ngữ cảnh trái và phải. "
        )
        if abs(delta_attn) < 0.005:
            insight += (
                f"**Attention** có đóng góp không đáng kể ({delta_attn:+.2%}) trên tập dữ liệu này — "
                f"có thể do văn bản ngắn, BiLSTM đã nắm đủ ngữ cảnh toàn cục. "
                f"Tuy nhiên, Attention vẫn mang lại giá trị **interpretability** cho mô hình."
            )
        else:
            insight += f"**Attention** đóng góp thêm **{delta_attn:+.2%}** vào accuracy."

        st.info(f"💡 **Nhận xét**: {insight}")
