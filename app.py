"""
app.py — ViLSTM Emotion · Demo phân loại cảm xúc văn bản ngắn tiếng Việt
Chạy: streamlit run app.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ViLSTM Emotion — Vietnamese NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
GRAD       = "linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%)"
GRAD_SOFT  = "linear-gradient(135deg, #EEF2FF 0%, #F5F3FF 100%)"
INK        = "#0F172A"
MUTED      = "#64748B"
LINE       = "#E2E8F0"
PRIMARY    = "#6366F1"

LABELS = ["ANG", "DIS", "FEA", "JOY", "NEU", "SAD", "SUR"]
EMOTION = {
    "ANG": ("Giận dữ",    "😡", "#EF4444"),
    "DIS": ("Ghê tởm",    "🤢", "#8B5CF6"),
    "FEA": ("Sợ hãi",     "😨", "#A855F7"),
    "JOY": ("Hạnh phúc",  "😄", "#F59E0B"),
    "NEU": ("Trung tính", "😐", "#64748B"),
    "SAD": ("Buồn bã",    "😢", "#3B82F6"),
    "SUR": ("Ngạc nhiên", "😲", "#10B981"),
}
MODEL_CLR = {"DNN+TF-IDF": "#F59E0B", "BiLSTM": "#6366F1", "XLM-R": "#10B981"}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
.stApp {{
    background:
        radial-gradient(1200px 500px at 80% -10%, rgba(139,92,246,0.06), transparent),
        radial-gradient(1000px 400px at -10% 10%, rgba(99,102,241,0.05), transparent),
        #FCFCFD;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none !important; }}
.block-container {{ padding: 1.25rem 2rem 4rem !important; max-width: 1120px; }}

/* ── Top brand bar ── */
.topbar {{ display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1rem; }}
.topbar .logo {{
    width: 40px; height: 40px; border-radius: 12px; background: {GRAD};
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; box-shadow: 0 4px 14px rgba(99,102,241,0.4);
}}
.topbar .brand {{ font-size: 1.1rem; font-weight: 800; color: {INK}; line-height: 1.1; }}
.topbar .tag   {{ font-size: 0.72rem; font-weight: 600; color: {PRIMARY}; }}

/* ── Buttons ── */
.stButton > button {{
    background: {GRAD} !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}}
.stButton > button:disabled {{
    background: #CBD5E1 !important; box-shadow: none !important;
}}
/* secondary (preset) buttons */
[data-testid="column"] .stButton > button[kind="secondary"] {{
    background: #fff !important; color: {INK} !important;
    border: 1.5px solid {LINE} !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    font-weight: 500 !important; font-size: 0.82rem !important;
}}
[data-testid="column"] .stButton > button[kind="secondary"]:hover {{
    border-color: {PRIMARY} !important; color: {PRIMARY} !important;
    transform: translateY(-1px) !important;
}}

/* ── Inputs ── */
.stTextArea textarea {{
    border: 1.5px solid {LINE} !important; border-radius: 12px !important;
    font-size: 0.95rem !important; background: #fff !important; color: {INK} !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
}}
.stTextArea textarea:focus {{
    border-color: {PRIMARY} !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.1) !important;
}}
.stTextArea label, .stSelectbox label {{
    color: {INK} !important; font-weight: 600 !important; font-size: 0.82rem !important;
}}
.stSelectbox > div > div {{
    border: 1.5px solid {LINE} !important; border-radius: 10px !important; background: #fff !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(255,255,255,0.7); backdrop-filter: blur(8px);
    border-radius: 12px !important; padding: 4px !important; gap: 3px !important;
    border: 1px solid {LINE} !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px !important; font-weight: 600 !important;
    font-size: 0.86rem !important; color: {MUTED} !important; padding: 0.45rem 1.1rem !important;
}}
.stTabs [aria-selected="true"] {{
    background: {GRAD} !important; color: #fff !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3) !important;
}}
.stTabs [aria-selected="true"] * {{ color: #fff !important; }}

hr {{ border-color: {LINE} !important; margin: 1.25rem 0 !important; }}
.stInfo {{ border-radius: 12px !important; font-size: 0.875rem !important; }}
[data-testid="stMetricValue"] {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
[data-testid="stMetricLabel"] {{ font-size: 0.74rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED} !important; }}

/* ── Reusable classes ── */
.eyebrow {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {PRIMARY};
}}
.section-eyebrow {{
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {MUTED}; margin: 0 0 0.75rem;
}}
.card {{
    background: rgba(255,255,255,0.85); backdrop-filter: blur(10px);
    border: 1px solid {LINE}; border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04), 0 8px 24px rgba(15,23,42,0.03);
}}
</style>
""", unsafe_allow_html=True)

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải model…")
def get_lstm():
    from src.inference.predict import LSTMPredictor; return LSTMPredictor()

@st.cache_resource(show_spinner="Đang tải model…")
def get_dnn():
    from src.inference.predict import DNNPredictor; return DNNPredictor()

@st.cache_resource(show_spinner="Đang tải model…")
def get_xlmr():
    from src.inference.predict import XLMRPredictor; return XLMRPredictor()

@st.cache_data
def read_json(name):
    p = ROOT / name
    return json.load(open(p)) if p.exists() else {}


def metric_pill(value, label, color=PRIMARY, sub=""):
    return (
        f"<div style='background:#fff;border:1px solid {LINE};border-radius:14px;"
        f"padding:1.15rem 1rem;text-align:center;position:relative;overflow:hidden'>"
        f"<div style='position:absolute;top:0;left:0;right:0;height:3px;background:{color}'></div>"
        f"<div style='font-size:1.55rem;font-weight:800;color:{color};line-height:1.1'>{value}</div>"
        f"<div style='font-size:0.76rem;font-weight:600;color:{INK};margin-top:0.35rem'>{label}</div>"
        f"{'<div style=\"font-size:0.7rem;color:#94A3B8;margin-top:2px\">'+sub+'</div>' if sub else ''}"
        f"</div>"
    )


# ── Top brand bar ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='topbar'>
  <div class='logo'>🧠</div>
  <div>
    <div class='brand'>ViLSTM Emotion</div>
    <div class='tag'>Vietnamese Short-Text Emotion Classification</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Horizontal navigation ─────────────────────────────────────────────────────
NAV_ITEMS = [
    "Tổng quan", "Demo trực tiếp", "Kết quả & Metrics",
    "Training curves", "Ablation study", "Kiến trúc mô hình",
]
NAV_ICONS = ["house", "magic", "bar-chart", "graph-up", "diagram-3", "cpu"]

page = option_menu(
    menu_title=None,
    options=NAV_ITEMS,
    icons=NAV_ICONS,
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {
            "padding": "4px",
            "background-color": "#FFFFFF",
            "border": "1px solid #E2E8F0",
            "border-radius": "14px",
            "box-shadow": "0 1px 3px rgba(15,23,42,0.04)",
            "margin-bottom": "1.5rem",
        },
        "icon": {"font-size": "0.92rem"},
        "nav-link": {
            "font-family": "'Plus Jakarta Sans', sans-serif",
            "font-size": "0.84rem",
            "font-weight": "600",
            "color": "#64748B",
            "padding": "0.5rem 0.9rem",
            "border-radius": "10px",
            "margin": "0 2px",
            "--hover-color": "#F1F5F9",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%)",
            "color": "#FFFFFF",
            "font-weight": "700",
            "box-shadow": "0 2px 8px rgba(99,102,241,0.3)",
        },
    },
)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — TỔNG QUAN  (Hero + Architecture + Metrics)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Tổng quan":
    comp = read_json("outputs/reports/comparison.json")
    bilstm = comp.get("BiLSTM", {})
    xlmr   = comp.get("XLM-R", {})

    # ── Hero ──
    st.markdown(f"""
    <div style='background:{GRAD};border-radius:24px;padding:2.75rem 2.5rem;
                position:relative;overflow:hidden;margin-bottom:1.5rem;
                box-shadow:0 20px 50px rgba(99,102,241,0.25)'>
      <div style='position:absolute;top:-60px;right:-40px;width:280px;height:280px;
                  background:radial-gradient(circle,rgba(255,255,255,0.18),transparent 70%)'></div>
      <div style='position:absolute;bottom:-80px;left:-30px;width:260px;height:260px;
                  background:radial-gradient(circle,rgba(255,255,255,0.1),transparent 70%)'></div>
      <div style='position:relative;z-index:1'>
        <span style='display:inline-block;background:rgba(255,255,255,0.18);
                     color:#fff;font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                     text-transform:uppercase;padding:0.35rem 0.85rem;border-radius:99px;
                     backdrop-filter:blur(8px);margin-bottom:1rem'>
          BiLSTM · Attention · 7-class Ekman
        </span>
        <h1 style='color:#fff;font-size:2.1rem;font-weight:800;margin:0;line-height:1.2;
                   max-width:680px'>
          Phân loại cảm xúc văn bản ngắn tiếng Việt bằng học sâu
        </h1>
        <p style='color:rgba(255,255,255,0.85);font-size:1rem;margin:0.9rem 0 0;max-width:600px;
                  line-height:1.6'>
          Mô hình Bidirectional LSTM kết hợp cơ chế Attention, nhận diện 7 cảm xúc
          từ ngôn ngữ mạng xã hội đầy teencode và emoji.
        </p>
        <div style='display:flex;gap:2rem;margin-top:1.75rem;flex-wrap:wrap'>
          <div>
            <div style='color:#fff;font-size:1.6rem;font-weight:800'>
              {bilstm.get('accuracy',0):.1%}</div>
            <div style='color:rgba(255,255,255,0.7);font-size:0.78rem'>BiLSTM Accuracy</div>
          </div>
          <div>
            <div style='color:#fff;font-size:1.6rem;font-weight:800'>
              {bilstm.get('f1',0):.1%}</div>
            <div style='color:rgba(255,255,255,0.7);font-size:0.78rem'>F1-score</div>
          </div>
          <div>
            <div style='color:#fff;font-size:1.6rem;font-weight:800'>
              {bilstm.get('num_params',0)/1e6:.1f}M</div>
            <div style='color:rgba(255,255,255,0.7);font-size:0.78rem'>Tham số</div>
          </div>
          <div>
            <div style='color:#fff;font-size:1.6rem;font-weight:800'>2.726</div>
            <div style='color:rgba(255,255,255,0.7);font-size:0.78rem'>Mẫu huấn luyện</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Architecture flow ──
    st.markdown("<p class='section-eyebrow'>Kiến trúc pipeline</p>", unsafe_allow_html=True)

    stages = [
        ("📝", "Văn bản thô", "Bình luận MXH<br>teencode · emoji", "#6366F1"),
        ("🧹", "Tiền xử lý", "Clean · chuẩn hoá<br>teencode (80+ rules)", "#7C3AED"),
        ("📚", "Vector hoá", "Word → index<br>vocabulary ~9k", "#8B5CF6"),
        ("🧠", "BiLSTM + Attn", "2 layers · 256d<br>Bahdanau attention", "#A855F7"),
        ("🎯", "7 cảm xúc", "Softmax<br>ANG…SUR", "#EC4899"),
    ]
    arrow = ("<div style='display:flex;align-items:center;justify-content:center;"
             "color:#CBD5E1;font-size:1.3rem;padding:0 0.2rem'>→</div>")
    flow = "<div style='display:flex;align-items:stretch;gap:0.2rem;flex-wrap:nowrap;overflow-x:auto'>"
    for i, (icon, title, desc, c) in enumerate(stages):
        flow += (
            f"<div style='flex:1;min-width:130px;background:#fff;border:1px solid {LINE};"
            f"border-top:3px solid {c};border-radius:14px;padding:1.1rem 0.85rem;text-align:center'>"
            f"<div style='font-size:1.7rem;margin-bottom:0.4rem'>{icon}</div>"
            f"<div style='font-weight:700;font-size:0.86rem;color:{INK}'>{title}</div>"
            f"<div style='font-size:0.72rem;color:{MUTED};margin-top:0.3rem;line-height:1.5'>{desc}</div>"
            f"</div>"
        )
        if i < len(stages) - 1:
            flow += arrow
    flow += "</div>"
    st.markdown(flow, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model comparison snapshot + emotion legend ──
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("<p class='section-eyebrow'>So sánh mô hình</p>", unsafe_allow_html=True)
        if comp:
            for name, m in comp.items():
                c = MODEL_CLR.get(name, PRIMARY)
                star = " ★" if name == "BiLSTM" else ""
                acc = m["accuracy"]
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:1rem;margin-bottom:0.7rem'>"
                    f"<span style='width:110px;font-size:0.84rem;font-weight:600;color:{INK}'>"
                    f"{name}{star}</span>"
                    f"<div style='flex:1;background:#F1F5F9;border-radius:99px;height:12px;overflow:hidden'>"
                    f"<div style='width:{acc*100:.1f}%;height:100%;border-radius:99px;"
                    f"background:linear-gradient(90deg,{c}99,{c})'></div></div>"
                    f"<span style='width:52px;text-align:right;font-size:0.84rem;font-weight:700;"
                    f"color:{INK}'>{acc:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Chạy `python main.py evaluate` để có kết quả.")

    with right:
        st.markdown("<p class='section-eyebrow'>7 nhãn cảm xúc</p>", unsafe_allow_html=True)
        chips = "<div style='display:flex;flex-wrap:wrap;gap:0.5rem'>"
        for code, (name, emoji, c) in EMOTION.items():
            chips += (
                f"<div style='display:flex;align-items:center;gap:0.4rem;"
                f"background:{c}12;border:1px solid {c}25;border-radius:99px;"
                f"padding:0.4rem 0.8rem'>"
                f"<span style='font-size:1rem'>{emoji}</span>"
                f"<span style='font-size:0.8rem;font-weight:600;color:{c}'>{name}</span></div>"
            )
        chips += "</div>"
        st.markdown(chips, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — DEMO TRỰC TIẾP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Demo trực tiếp":
    st.markdown(
        f"<p class='eyebrow'>Live Demo</p>"
        f"<h2 style='margin:0.2rem 0 0.3rem;font-size:1.6rem;font-weight:800;color:{INK}'>"
        f"Thử nghiệm phân tích cảm xúc</h2>"
        f"<p style='color:{MUTED};margin:0 0 1.5rem;font-size:0.92rem'>"
        f"Nhập văn bản tiếng Việt — model nhận diện cảm xúc và giải thích quyết định.</p>",
        unsafe_allow_html=True,
    )

    col_in, col_cfg = st.columns([3, 1], gap="large")
    with col_in:
        text_input = st.text_area(
            "Văn bản đầu vào",
            value=st.session_state.get("demo_text", ""),
            height=130,
            placeholder="VD: Hôm nay mình cực vui vì thi đạt điểm cao, cảm ơn mọi người!!!",
        )
    with col_cfg:
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        model_sel = st.selectbox("Mô hình", ["BiLSTM + Attention", "DNN + TF-IDF", "XLM-RoBERTa"])
        show_attn = st.toggle("Attention weights", value=True, disabled="BiLSTM" not in model_sel)
        btn = st.button("Phân tích cảm xúc →", use_container_width=True, type="primary")

    # ── Preset categories ──
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    preset_groups = {
        "😊 Cảm xúc cơ bản": [
            "Hôm nay thi xong rồi, vui lắm luôn!!!",
            "Mệt và buồn quá, chẳng muốn làm gì cả...",
            "Tức chết đi được, sao lại đối xử như vậy chứ???",
        ],
        "⚡ Teencode nặng": [
            "ko hiểu sao nay mik buồn z, chán ghê",
            "vui vcl các bạn ơi, dc nghỉ học nè kkk",
            "thoy mệt r, ko mún nói chuyện vs ai nữa",
        ],
        "🌐 Trộn ngôn ngữ": [
            "Ôi god ơi không ngờ pass môn luôn, so happy!",
            "Deadline dí mà còn bug, stress cực kỳ",
            "Crush rep tin nhắn rồi, excited quá đi",
        ],
    }
    pcols = st.columns(3, gap="medium")
    for col, (group, samples) in zip(pcols, preset_groups.items()):
        with col:
            st.markdown(
                f"<div style='font-size:0.76rem;font-weight:700;color:{INK};"
                f"margin-bottom:0.5rem'>{group}</div>",
                unsafe_allow_html=True,
            )
            for i, s in enumerate(samples):
                if st.button(s[:34] + ("…" if len(s) > 34 else ""),
                             key=f"p_{group}_{i}", use_container_width=True, type="secondary"):
                    st.session_state["demo_text"] = s
                    st.rerun()

    # ── Prediction ──
    if btn and text_input.strip():
        with st.spinner("Đang phân tích…"):
            try:
                predictor = (get_lstm() if "BiLSTM" in model_sel
                             else get_dnn() if "DNN" in model_sel
                             else get_xlmr())
                res   = predictor.predict(text_input)
                pred  = res["predicted_label"]
                conf  = res["confidence"]
                probs = res["probabilities"]
                ename, eemoji, ecolor = EMOTION[pred]
            except FileNotFoundError:
                st.error("Model chưa được train. Chạy `python main.py train` trước.")
                st.stop()

        st.divider()
        r1, r2, r3 = st.columns([1, 2, 2], gap="large")

        with r1:
            st.markdown(
                f"<div style='background:linear-gradient(160deg,{ecolor}14,{ecolor}06);"
                f"border:1.5px solid {ecolor}30;border-radius:18px;padding:1.6rem;"
                f"text-align:center'>"
                f"<div style='font-size:3rem;line-height:1'>{eemoji}</div>"
                f"<div style='font-size:1.2rem;font-weight:800;color:{ecolor};"
                f"margin:0.6rem 0 0.15rem'>{ename}</div>"
                f"<div style='font-size:0.72rem;color:{ecolor};opacity:.7;font-weight:700;"
                f"letter-spacing:0.08em'>{pred}</div>"
                f"<div style='margin-top:1rem;padding-top:1rem;border-top:1px solid {ecolor}20'>"
                f"<span style='font-size:1.5rem;font-weight:800;color:{INK}'>{conf:.1%}</span>"
                f"<div style='font-size:0.7rem;color:#94A3B8;margin-top:2px'>Độ tin cậy</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown("<p class='section-eyebrow'>Phân phối xác suất</p>", unsafe_allow_html=True)
            for lbl in sorted(probs, key=lambda x: -probs[x]):
                p = probs[lbl]
                _, em, c = EMOTION[lbl]
                is_pred = lbl == pred
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
                    f"<span style='width:30px;font-size:0.8rem;font-weight:"
                    f"{'700' if is_pred else '400'};color:#334155'>{lbl}</span>"
                    f"<div style='flex:1;background:#F1F5F9;border-radius:99px;height:10px;overflow:hidden'>"
                    f"<div style='width:{p*100:.1f}%;height:100%;border-radius:99px;"
                    f"background:linear-gradient(90deg,{c}99,{c})'></div></div>"
                    f"<span style='width:40px;text-align:right;font-size:0.8rem;"
                    f"color:{'#0F172A' if is_pred else '#94A3B8'};"
                    f"font-weight:{'700' if is_pred else '400'}'>{p:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        with r3:
            st.markdown("<p class='section-eyebrow'>Các bước tiền xử lý</p>", unsafe_allow_html=True)
            from src.preprocessing.clean_text import clean_text
            from src.preprocessing.teencode_normalize import normalize_teencode
            cleaned = clean_text(text_input, remove_emoji=False)
            normed  = normalize_teencode(cleaned)
            for step, val in [("Văn bản gốc", text_input),
                              ("Sau làm sạch", cleaned),
                              ("Sau chuẩn hoá teencode", normed)]:
                st.markdown(
                    f"<div style='margin-bottom:0.6rem'>"
                    f"<div style='font-size:0.7rem;font-weight:700;color:#94A3B8;"
                    f"margin-bottom:3px'>{step}</div>"
                    f"<div style='background:#F8FAFC;border:1px solid {LINE};border-radius:9px;"
                    f"padding:0.5rem 0.75rem;font-size:0.85rem;color:#334155;"
                    f"word-break:break-word;line-height:1.5'>"
                    f"{val or '<em style=\"color:#CBD5E1\">trống</em>'}</div></div>",
                    unsafe_allow_html=True,
                )

        # ── Attention token highlight ──
        if (show_attn and "BiLSTM" in model_sel
                and hasattr(predictor, "get_attention_weights")):
            weights = predictor.get_attention_weights(text_input)
            if weights:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<p class='section-eyebrow'>"
                            "Attention — từ nào ảnh hưởng nhiều nhất đến quyết định</p>",
                            unsafe_allow_html=True)
                tokens = list(weights.keys())
                scores = np.array(list(weights.values()))
                s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                import matplotlib.colors as mcolors
                toks_html = ("<div class='card' style='display:flex;flex-wrap:wrap;"
                             "gap:0.45rem;align-items:center'>")
                for tok, s in zip(tokens, s_norm):
                    rgba = mcolors.to_hex(plt.cm.Purples(0.2 + s * 0.7))
                    fg   = "#fff" if s > 0.5 else INK
                    toks_html += (
                        f"<span style='background:{rgba};color:{fg};"
                        f"padding:0.35rem 0.7rem;border-radius:8px;"
                        f"font-size:{0.82 + s*0.22:.2f}rem;font-weight:"
                        f"{600 if s > 0.5 else 500};font-family:\"JetBrains Mono\",monospace'>"
                        f"{tok}</span>"
                    )
                toks_html += "</div>"
                st.markdown(toks_html, unsafe_allow_html=True)
                st.caption("Màu càng đậm = trọng số attention càng cao.")


# ══════════════════════════════════════════════════════════════════════════════
# 3 — KẾT QUẢ & METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Kết quả & Metrics":
    st.markdown(
        f"<p class='eyebrow'>Evaluation</p>"
        f"<h2 style='margin:0.2rem 0 0.3rem;font-size:1.6rem;font-weight:800;color:{INK}'>"
        f"Kết quả thực nghiệm</h2>"
        f"<p style='color:{MUTED};margin:0 0 1.5rem;font-size:0.92rem'>"
        f"Đánh giá trên tập kiểm tra · 408 mẫu · stratified.</p>",
        unsafe_allow_html=True,
    )

    comp = read_json("outputs/reports/comparison.json")
    if not comp:
        st.warning("Chưa có dữ liệu. Chạy `python main.py evaluate`"); st.stop()

    cols = st.columns(len(comp), gap="medium")
    for col, (name, m) in zip(cols, comp.items()):
        c = MODEL_CLR.get(name, PRIMARY)
        tag = "Đề xuất ★" if name == "BiLSTM" else ("Upper bound" if name == "XLM-R" else "Baseline")
        col.markdown(metric_pill(f"{m['accuracy']:.2%}", name, c, tag), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Bảng so sánh", "Confusion matrix", "Radar chart"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        th = ("padding:0.75rem 1rem;text-align:left;color:#64748B;font-size:0.7rem;"
              "font-weight:700;text-transform:uppercase;letter-spacing:0.05em;"
              f"border-bottom:1.5px solid {LINE}")
        head = (f"<div class='card' style='padding:0;overflow:hidden'>"
                f"<table style='width:100%;border-collapse:collapse;font-size:0.875rem'>"
                f"<thead><tr style='background:#F8FAFC'>")
        for h in ["Mô hình","Accuracy","Precision","Recall","F1","Params","Inf."]:
            head += f"<th style='{th}'>{h}</th>"
        head += "</tr></thead><tbody>"
        rows = ""
        for name, m in comp.items():
            c = MODEL_CLR.get(name, "#888")
            star = " ★" if name == "BiLSTM" else ""
            rows += (
                f"<tr style='border-bottom:1px solid #F1F5F9'>"
                f"<td style='padding:0.9rem 1rem'><span style='background:{c}18;color:{c};"
                f"padding:0.2rem 0.7rem;border-radius:99px;font-size:0.78rem;font-weight:700'>"
                f"{name+star}</span></td>"
                f"<td style='padding:0.9rem 1rem;font-weight:700;color:{INK}'>{m['accuracy']:.4f}</td>"
                f"<td style='padding:0.9rem 1rem;color:#334155'>{m['precision']:.4f}</td>"
                f"<td style='padding:0.9rem 1rem;color:#334155'>{m['recall']:.4f}</td>"
                f"<td style='padding:0.9rem 1rem;color:#334155'>{m['f1']:.4f}</td>"
                f"<td style='padding:0.9rem 1rem;color:#94A3B8'>{m['num_params']:,}</td>"
                f"<td style='padding:0.9rem 1rem;color:#94A3B8'>{m['inference_ms_per_sample']:.1f}ms</td>"
                f"</tr>"
            )
        st.markdown(head + rows + "</tbody></table></div>", unsafe_allow_html=True)

        pcf1 = comp.get("BiLSTM", {}).get("per_class_f1", {})
        if pcf1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<p class='section-eyebrow'>F1 theo từng nhãn — BiLSTM</p>",
                        unsafe_allow_html=True)
            bars = "<div class='card' style='display:flex;flex-direction:column;gap:0.65rem'>"
            for lbl, f1 in sorted(pcf1.items(), key=lambda x: -x[1]):
                _, em, c = EMOTION[lbl]
                bars += (
                    f"<div style='display:flex;align-items:center;gap:12px'>"
                    f"<span style='width:90px;font-size:0.82rem;font-weight:600;color:#334155'>"
                    f"{em} {lbl}</span>"
                    f"<div style='flex:1;background:#F1F5F9;border-radius:99px;height:11px'>"
                    f"<div style='width:{f1*100:.1f}%;height:100%;border-radius:99px;"
                    f"background:linear-gradient(90deg,{c}99,{c})'></div></div>"
                    f"<span style='width:44px;text-align:right;font-size:0.82rem;font-weight:700;"
                    f"color:{INK}'>{f1:.3f}</span></div>"
                )
            st.markdown(bars + "</div>", unsafe_allow_html=True)

    with tab2:
        cm = {"BiLSTM":"bilstm_cm.png","DNN+TF-IDF":"dnn_tf_idf_cm.png","XLM-R":"xlm_r_cm.png"}
        cols = st.columns(3, gap="medium")
        for col, (name, fname) in zip(cols, cm.items()):
            c = MODEL_CLR.get(name, "#888")
            path = ROOT / "outputs/figures" / fname
            col.markdown(
                f"<p style='text-align:center;margin-bottom:0.5rem'>"
                f"<span style='background:{c}18;color:{c};padding:0.25rem 0.8rem;"
                f"border-radius:99px;font-size:0.78rem;font-weight:700'>{name}</span></p>",
                unsafe_allow_html=True)
            if path.exists(): col.image(str(path), use_container_width=True)
            else: col.caption("Chưa có hình")

    with tab3:
        metrics = ["Accuracy","Precision","Recall","F1"]
        fig = go.Figure()
        for name, m in comp.items():
            v = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
            c = MODEL_CLR.get(name, "#888")
            fig.add_trace(go.Scatterpolar(
                r=v+[v[0]], theta=metrics+[metrics[0]], fill="toself", name=name,
                line=dict(color=c, width=2.5), fillcolor=c, opacity=0.16,
                marker=dict(size=6, color=c)))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0.72,0.86], tickfont=dict(size=9,color="#94A3B8"),
                                gridcolor=LINE, linecolor=LINE),
                angularaxis=dict(tickfont=dict(size=12,color="#334155"),
                                 gridcolor=LINE, linecolor=LINE),
                bgcolor="white"),
            paper_bgcolor="white", showlegend=True,
            legend=dict(font=dict(size=11), x=1.05, y=0.5),
            margin=dict(l=50,r=120,t=20,b=20), height=380)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Training curves":
    st.markdown(
        f"<p class='eyebrow'>Training</p>"
        f"<h2 style='margin:0.2rem 0 0.3rem;font-size:1.6rem;font-weight:800;color:{INK}'>"
        f"Diễn tiến huấn luyện</h2>"
        f"<p style='color:{MUTED};margin:0 0 1.5rem;font-size:0.92rem'>"
        f"Loss và accuracy qua từng epoch · early stopping theo val loss.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["BiLSTM + Attention ★", "DNN + TF-IDF", "XLM-RoBERTa"])
    cfgs = [("lstm","#6366F1"), ("dnn","#F59E0B"), ("xlmr","#10B981")]

    for tab, (key, mclr) in zip(tabs, cfgs):
        with tab:
            hist = read_json(f"outputs/logs/{key}_history.json")
            if not hist:
                st.info(f"Chưa có {key}_history.json"); continue
            n = len(hist["train_loss"]); ep = list(range(1, n+1))
            cl, cr = st.columns(2, gap="medium")

            def curve(container, ykeys, title, fmt=None):
                fig = go.Figure()
                for yk, nm, dash, color in ykeys:
                    fig.add_trace(go.Scatter(
                        x=ep, y=hist[yk], name=nm, mode="lines+markers",
                        line=dict(width=2.2, color=color, dash=dash),
                        marker=dict(size=5, color=color),
                        hovertemplate=("E%{x}: %{y:.4f}" if not fmt else "E%{x}: %{y:.2%}")
                                       + f"<extra>{nm}</extra>"))
                ya = dict(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(size=10),
                          zeroline=False)
                if fmt: ya["tickformat"] = ".0%"
                fig.update_layout(
                    title=dict(text=title, font=dict(size=12.5,color=INK), x=0),
                    xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F1F5F9",
                               tickfont=dict(size=10), zeroline=False),
                    yaxis=ya, paper_bgcolor="white", plot_bgcolor="white",
                    height=290, margin=dict(l=48,r=15,t=42,b=38),
                    legend=dict(font=dict(size=10), x=0.98 if not fmt else 0.02,
                                y=0.98, xanchor="right" if not fmt else "left",
                                yanchor="top", bgcolor="rgba(255,255,255,0.85)",
                                bordercolor=LINE, borderwidth=1),
                    hovermode="x unified")
                container.plotly_chart(fig, use_container_width=True)

            curve(cl, [("train_loss","Train","solid",mclr),
                       ("val_loss","Validation","dash","#94A3B8")], "Loss")
            curve(cr, [("train_acc","Train","solid",mclr),
                       ("val_acc","Validation","dash","#94A3B8")], "Accuracy", fmt=True)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Epochs", hist.get("best_epoch", n), f"/ {n}")
            c2.metric("Best val loss", f"{min(hist['val_loss']):.4f}")
            c3.metric("Test accuracy", f"{hist['test_acc']:.2%}" if "test_acc" in hist else "—")
            c4.metric("Test F1", f"{hist['test_f1']:.2%}" if "test_f1" in hist else "—")


# ══════════════════════════════════════════════════════════════════════════════
# 5 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ablation study":
    st.markdown(
        f"<p class='eyebrow'>Ablation</p>"
        f"<h2 style='margin:0.2rem 0 0.3rem;font-size:1.6rem;font-weight:800;color:{INK}'>"
        f"Phân tích thành phần</h2>"
        f"<p style='color:{MUTED};margin:0 0 1.5rem;font-size:0.92rem'>"
        f"Tắt từng module để đo đóng góp của Bidirectionality và Attention.</p>",
        unsafe_allow_html=True,
    )

    abl = read_json("outputs/logs/ablation_results.json")
    if not abl:
        st.warning("Chưa có dữ liệu"); st.stop()
    vals = list(abl.values())
    db = vals[1]["accuracy"] - vals[0]["accuracy"]
    da = vals[2]["accuracy"] - vals[1]["accuracy"]

    c0,c1,c2 = st.columns(3, gap="medium")
    for col, title, acc, delta, color in [
        (c0,"LSTM (baseline)",   vals[0]["accuracy"], None, "#94A3B8"),
        (c1,"+ Bidirectionality",vals[1]["accuracy"], db,   "#10B981" if db>0 else "#EF4444"),
        (c2,"+ Attention",       vals[2]["accuracy"], da,   "#10B981" if da>0.002 else "#F59E0B"),
    ]:
        sub = f"Δ {delta:+.2%}" if delta is not None else "Điểm khởi đầu"
        col.markdown(metric_pill(f"{acc:.2%}", title, color, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    names = [n.replace(" (proposed)","").replace(" (uni, no-attn)","") for n in abl]
    accs = [m["accuracy"] for m in vals]; f1s = [m["f1"] for m in vals]
    bclr = ["#CBD5E1", "#A5B4FC", "#6366F1"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=names, y=accs, offsetgroup=0,
        marker_color=bclr, text=[f"{v:.3f}" for v in accs], textposition="outside",
        textfont=dict(size=10.5,color="#334155")))
    fig.add_trace(go.Bar(name="F1", x=names, y=f1s, offsetgroup=1,
        marker_color=[c+"80" for c in bclr], text=[f"{v:.3f}" for v in f1s],
        textposition="outside", textfont=dict(size=10.5,color="#334155")))
    fig.update_layout(barmode="group",
        yaxis=dict(range=[min(accs+f1s)-0.015, max(accs+f1s)+0.025], tickformat=".0%",
                   showgrid=True, gridcolor="#F1F5F9", tickfont=dict(size=10), zeroline=False),
        xaxis=dict(tickfont=dict(size=11), zeroline=False),
        paper_bgcolor="white", plot_bgcolor="white", height=340,
        margin=dict(l=52,r=20,t=20,b=40), bargap=0.25, bargroupgap=0.06,
        legend=dict(font=dict(size=11), bgcolor="white", bordercolor=LINE, borderwidth=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    th = ("padding:0.7rem 1rem;text-align:left;color:#64748B;font-size:0.7rem;font-weight:700;"
          f"text-transform:uppercase;letter-spacing:0.05em;border-bottom:1.5px solid {LINE}")
    rows = ""
    comps = [("❌","❌"),("✅","❌"),("✅","✅")]
    for (name, m),(bi,att) in zip(abl.items(), comps):
        star = " ★" if "Attention" in name else ""
        c = "#6366F1" if "Attention" in name else ("#A5B4FC" if "BiLSTM" in name else "#94A3B8")
        rows += (f"<tr style='border-bottom:1px solid #F1F5F9'>"
                 f"<td style='padding:0.8rem 1rem'><span style='background:{c}18;color:{c};"
                 f"padding:0.2rem 0.7rem;border-radius:99px;font-size:0.78rem;font-weight:700'>"
                 f"{name+star}</span></td>"
                 f"<td style='padding:0.8rem 1rem;text-align:center;font-size:1rem'>{bi}</td>"
                 f"<td style='padding:0.8rem 1rem;text-align:center;font-size:1rem'>{att}</td>"
                 f"<td style='padding:0.8rem 1rem;font-weight:700;color:{INK}'>{m['accuracy']:.4f}</td>"
                 f"<td style='padding:0.8rem 1rem;color:#334155'>{m['f1']:.4f}</td>"
                 f"<td style='padding:0.8rem 1rem;color:#94A3B8'>{m['num_params']:,}</td></tr>")
    st.markdown(
        f"<div class='card' style='padding:0;overflow:hidden'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.875rem'>"
        f"<thead style='background:#F8FAFC'><tr>"
        f"<th style='{th}'>Biến thể</th><th style='{th};text-align:center'>BiDirectional</th>"
        f"<th style='{th};text-align:center'>Attention</th><th style='{th}'>Accuracy</th>"
        f"<th style='{th}'>F1</th><th style='{th}'>Params</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    body = (f"Bidirectionality cải thiện <b>{db:+.2%}</b> accuracy nhờ mã hoá ngữ cảnh hai chiều. ")
    if abs(da) < 0.005:
        body += (f"Attention đóng góp nhỏ ({da:+.2%}) trên dữ liệu ngắn này, "
                 f"nhưng mang lại <b>khả năng giải thích</b> khi trực quan hoá trọng số.")
    else:
        body += f"Attention cải thiện thêm <b>{da:+.2%}</b>."
    st.markdown(
        f"<div style='background:{GRAD_SOFT};border:1px solid #DDD6FE;border-radius:14px;"
        f"padding:1.1rem 1.3rem;font-size:0.88rem;color:#4338CA;line-height:1.6'>"
        f"💡 {body}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6 — KIẾN TRÚC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Kiến trúc mô hình":
    st.markdown(
        f"<p class='eyebrow'>Architecture</p>"
        f"<h2 style='margin:0.2rem 0 0.3rem;font-size:1.6rem;font-weight:800;color:{INK}'>"
        f"Kiến trúc các mô hình</h2>"
        f"<p style='color:{MUTED};margin:0 0 1.5rem;font-size:0.92rem'>"
        f"Baseline → Mô hình đề xuất → So sánh state-of-the-art.</p>",
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["BiLSTM + Attention ★", "DNN + TF-IDF", "XLM-RoBERTa"])
    with t1:
        st.markdown("""
**Mô hình đề xuất** — tối ưu cho văn bản ngắn, ít tài nguyên.

| Thành phần | Chi tiết |
|---|---|
| Embedding | vocab × 128d, padding_idx=0 |
| Encoder | BiLSTM · 2 layers · hidden=256 · dropout=0.3 |
| Attention | Bahdanau additive · score = v·tanh(W·h) |
| Classifier | Linear 512→256 → LayerNorm → GELU → Dropout → Linear 256→7 |
| Tham số | **2.919.047** |
""")
        st.code("""Input: [t₁, t₂, …, tₙ]                 # token indices
   │
Embedding(vocab, 128) + Dropout(0.3)    # (B, L, 128)
   │
BiLSTM(128 → 256, num_layers=2)
   │   Forward : h₁→ h₂→ … hₙ→
   │   Backward: hₙ← … h₂← h₁←
   │   Output  : [hᵢ→ ‖ hᵢ←]            # (B, L, 512)
   │
Attention (Bahdanau)
   │   eᵢ = v · tanh(W · hᵢ)
   │   αᵢ = softmax(e)
   │   c  = Σ αᵢ · hᵢ                    # (B, 512)
   │
Linear(512→256) → LayerNorm → GELU → Dropout(0.3)
Linear(256→7)   →  logits""", language="text")
    with t2:
        st.markdown("""
**Baseline** — biểu diễn thống kê, không nắm thứ tự từ.

| Thành phần | Chi tiết |
|---|---|
| Vectoriser | TF-IDF · 10.868 features · bigram · sublinear_tf |
| MLP | 512 → 256 → 128, mỗi tầng BN + ReLU + Dropout(0.3) |
| Output | Linear(128→7) |
| Tham số | **5.731.847** |
""")
    with t3:
        st.markdown("""
**Upper-bound** — Transformer đa ngôn ngữ pre-trained.

| Thành phần | Chi tiết |
|---|---|
| Backbone | `xlm-roberta-base` · 12 layers · hidden=768 |
| Head | Dropout(0.1) → Linear(768→7) |
| Tham số | **278.049.031** |
""")
        st.info("Pre-train trên 2.5TB văn bản đa ngôn ngữ (gồm tiếng Việt) — "
                "thể hiện giới hạn trên của bài toán với dữ liệu hiện có.")
