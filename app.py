"""
app.py — Demo ứng dụng phân loại cảm xúc văn bản ngắn tiếng Việt
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Classifier — NLP Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background: #F9FAFB; }

/* Ẩn chrome mặc định */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 3rem !important; max-width: 1080px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 2px !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #9CA3AF !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #F3F4F6 !important;
}
[data-testid="stSidebar"] hr { border-color: #1F2937 !important; margin: 1rem 0 !important; }

/* Button primary */
.stButton > button {
    background: #2563EB !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important; letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }
.stButton > button:disabled { background: #93C5FD !important; }

/* Textarea */
.stTextArea textarea {
    border: 1.5px solid #E5E7EB !important;
    border-radius: 9px !important;
    font-size: 0.95rem !important;
    font-family: 'Inter', sans-serif !important;
    background: #fff !important;
    color: #111827 !important;
    transition: border-color 0.15s !important;
}
.stTextArea textarea:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stTextArea label { color: #374151 !important; font-weight: 600 !important; font-size: 0.875rem !important; }

/* Selectbox */
.stSelectbox > div > div {
    border: 1.5px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: #fff !important;
}
.stSelectbox label { color: #374151 !important; font-weight: 600 !important; font-size: 0.875rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #F3F4F6 !important;
    border-radius: 9px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: 1px solid #E5E7EB !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.865rem !important;
    color: #6B7280 !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #111827 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}

/* Divider */
hr { border-color: #E5E7EB !important; margin: 1.25rem 0 !important; }

/* Info / warning boxes */
.stInfo, .stWarning { border-radius: 8px !important; font-size: 0.875rem !important; }

/* Metric */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; color: #6B7280 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABELS = ["ANG","DIS","FEA","JOY","NEU","SAD","SUR"]
EMOTION = {
    "ANG": ("Giận dữ",    "😡", "#DC2626"),
    "DIS": ("Ghê tởm",    "🤢", "#7C3AED"),
    "FEA": ("Sợ hãi",     "😨", "#9333EA"),
    "JOY": ("Hạnh phúc",  "😄", "#D97706"),
    "NEU": ("Trung tính", "😐", "#6B7280"),
    "SAD": ("Buồn bã",    "😢", "#2563EB"),
    "SUR": ("Ngạc nhiên", "😲", "#059669"),
}
MODEL_CLR = {"DNN+TF-IDF":"#D97706","BiLSTM":"#2563EB","XLM-R":"#059669"}

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.25rem 0 1.5rem'>
      <p style='color:#F9FAFB;font-size:1.05rem;font-weight:700;margin:0;line-height:1.3'>
        Vietnamese Emotion Classifier
      </p>
      <p style='color:#4B5563;font-size:0.78rem;margin:0.3rem 0 0'>
        BiLSTM · DNN · XLM-RoBERTa
      </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Demo dự đoán",
        "Kết quả & Metrics",
        "Training curves",
        "Ablation study",
        "Kiến trúc mô hình",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style='font-size:0.78rem;color:#4B5563;line-height:1.8'>
      <b style='color:#9CA3AF;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em'>
        Dataset
      </b><br>
      2.726 bình luận MXH<br>
      7 nhãn cảm xúc<br>
      Train / Val / Test: 70/15/15
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — DEMO DỰ ĐOÁN
# ══════════════════════════════════════════════════════════════════════════════
if page == "Demo dự đoán":
    st.markdown("## Phân tích cảm xúc văn bản")
    st.markdown(
        "<p style='color:#6B7280;margin-top:-0.5rem;margin-bottom:1.5rem;font-size:0.9rem'>"
        "Nhập văn bản tiếng Việt bất kỳ — hệ thống tự động nhận diện cảm xúc.</p>",
        unsafe_allow_html=True,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    col_in, col_cfg = st.columns([3, 1], gap="large")

    with col_in:
        text_input = st.text_area(
            "Văn bản đầu vào",
            value=st.session_state.get("demo_text", ""),
            height=120,
            placeholder="VD: Hôm nay mình cực vui vì thi đạt điểm cao, cảm ơn mọi người!!!",
        )

    with col_cfg:
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        model_sel = st.selectbox("Mô hình", ["BiLSTM + Attention", "DNN + TF-IDF", "XLM-RoBERTa"])
        show_attn = st.toggle("Attention weights", value=True,
                              disabled="BiLSTM" not in model_sel)
        btn = st.button("Phân tích →", use_container_width=True, type="primary")

    # ── Quick examples ────────────────────────────────────────────────────────
    with st.expander("Thử với câu mẫu"):
        ex_rows = [
            [("😄 Vui", "Hôm nay thi xong rồi, vui lắm kkk!!!"),
             ("😢 Buồn", "Mệt và buồn quá, không muốn làm gì cả...")],
            [("😡 Tức", "Tức chết đi được, sao lại phản bội như vậy???"),
             ("😲 Shock", "Trời ơi không ngờ được nhận học bổng, shock thật!")],
            [("😨 Sợ", "Lo lắng quá không biết kết quả ra sao nữa..."),
             ("🤢 Ghê", "Kinh quá, sao có thể làm chuyện đó được chứ")],
        ]
        for row in ex_rows:
            c1, c2 = st.columns(2)
            for col, (lbl, sample) in zip([c1,c2], row):
                if col.button(lbl, key=f"ex_{lbl}", use_container_width=True):
                    st.session_state["demo_text"] = sample
                    st.rerun()

    # ── Prediction ────────────────────────────────────────────────────────────
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

        # Result + probs + preprocessing
        r1, r2, r3 = st.columns([1, 2, 2], gap="large")

        # Card kết quả
        with r1:
            st.markdown(
                f"<div style='background:{ecolor}0D;border:1.5px solid {ecolor}30;"
                f"border-radius:12px;padding:1.5rem;text-align:center'>"
                f"<div style='font-size:2.75rem;line-height:1'>{eemoji}</div>"
                f"<div style='font-size:1.1rem;font-weight:700;color:{ecolor};"
                f"margin:0.6rem 0 0.2rem'>{ename}</div>"
                f"<div style='font-size:0.75rem;color:{ecolor};opacity:.7;"
                f"font-weight:600;letter-spacing:0.05em'>{pred}</div>"
                f"<div style='margin-top:0.9rem;padding-top:0.9rem;"
                f"border-top:1px solid {ecolor}20'>"
                f"<span style='font-size:1.4rem;font-weight:700;color:#111827'>"
                f"{conf:.1%}</span>"
                f"<div style='font-size:0.72rem;color:#9CA3AF;margin-top:2px'>Độ tin cậy</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # Probability bars
        with r2:
            st.markdown(
                "<p style='font-size:0.78rem;font-weight:600;color:#6B7280;"
                "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem'>"
                "Phân phối xác suất</p>",
                unsafe_allow_html=True,
            )
            for lbl in sorted(probs, key=lambda x: -probs[x]):
                p = probs[lbl]
                _, _, c = EMOTION[lbl]
                is_pred = lbl == pred
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:7px'>"
                    f"<span style='width:26px;font-size:0.8rem;font-weight:"
                    f"{'700' if is_pred else '400'};color:#374151'>{lbl}</span>"
                    f"<div style='flex:1;background:#F3F4F6;border-radius:99px;"
                    f"height:9px;overflow:hidden'>"
                    f"<div style='width:{p*100:.1f}%;height:100%;background:{c};"
                    f"border-radius:99px'></div></div>"
                    f"<span style='width:38px;text-align:right;font-size:0.8rem;"
                    f"color:{'#111827' if is_pred else '#9CA3AF'};"
                    f"font-weight:{'600' if is_pred else '400'}'>{p:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Preprocessing
        with r3:
            st.markdown(
                "<p style='font-size:0.78rem;font-weight:600;color:#6B7280;"
                "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem'>"
                "Tiền xử lý</p>",
                unsafe_allow_html=True,
            )
            from src.preprocessing.clean_text import clean_text
            from src.preprocessing.teencode_normalize import normalize_teencode
            cleaned = clean_text(text_input, remove_emoji=False)
            normed  = normalize_teencode(cleaned)
            for step, val in [("Văn bản gốc", text_input),
                               ("Sau làm sạch", cleaned),
                               ("Sau chuẩn hoá", normed)]:
                st.markdown(
                    f"<div style='margin-bottom:0.6rem'>"
                    f"<div style='font-size:0.72rem;font-weight:600;"
                    f"color:#9CA3AF;margin-bottom:3px'>{step}</div>"
                    f"<div style='background:#F9FAFB;border:1px solid #E5E7EB;"
                    f"border-radius:7px;padding:0.45rem 0.7rem;"
                    f"font-size:0.85rem;color:#374151;word-break:break-word;"
                    f"line-height:1.5'>{val or '<em style=\"color:#D1D5DB\">trống</em>'}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        # Attention heatmap
        if (show_attn and "BiLSTM" in model_sel
                and hasattr(predictor, "get_attention_weights")):
            weights = predictor.get_attention_weights(text_input)
            if weights:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-size:0.78rem;font-weight:600;color:#6B7280;"
                    "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem'>"
                    "Attention weights — từ nào model tập trung</p>",
                    unsafe_allow_html=True,
                )
                tokens = list(weights.keys())
                scores = np.array(list(weights.values()))
                s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

                fig, ax = plt.subplots(figsize=(min(14, max(6, len(tokens)*0.85)), 2.2))
                colors  = [plt.cm.Blues(0.3 + s * 0.6) for s in s_norm]
                bars    = ax.bar(range(len(tokens)), s_norm, color=colors,
                                 edgecolor="white", linewidth=0.8, width=0.7)
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=25, ha="right",
                                   fontsize=10.5, fontfamily="monospace")
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_facecolor("white")
                fig.patch.set_facecolor("white")
                fig.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2 — KẾT QUẢ & METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Kết quả & Metrics":
    st.markdown("## Kết quả thực nghiệm")
    st.markdown(
        "<p style='color:#6B7280;margin-top:-0.5rem;margin-bottom:1.5rem;font-size:0.9rem'>"
        "Đánh giá trên tập kiểm tra (test set · 408 mẫu · stratified).</p>",
        unsafe_allow_html=True,
    )

    comp = read_json("outputs/reports/comparison.json")
    if not comp:
        st.warning("Chưa có dữ liệu. Chạy `python main.py evaluate`"); st.stop()

    # Metric row
    cols = st.columns(len(comp), gap="medium")
    for col, (name, m) in zip(cols, comp.items()):
        c = MODEL_CLR.get(name, "#2563EB")
        tag = "Đề xuất" if name == "BiLSTM" else ("Upper bound" if name == "XLM-R" else "Baseline")
        col.markdown(
            f"<div style='background:#fff;border:1px solid #E5E7EB;border-top:3px solid {c};"
            f"border-radius:10px;padding:1.1rem 1rem;text-align:center'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{c}'>{m['accuracy']:.2%}</div>"
            f"<div style='font-size:0.78rem;font-weight:600;color:#374151;margin:0.2rem 0 0.1rem'>{name}</div>"
            f"<div style='font-size:0.72rem;color:#9CA3AF'>{tag}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Bảng so sánh", "Confusion matrix", "Radar chart"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        # Table
        header = (
            "<div style='background:#fff;border:1px solid #E5E7EB;border-radius:10px;"
            "overflow:hidden'><table style='width:100%;border-collapse:collapse;"
            "font-size:0.875rem'>"
            "<thead><tr style='background:#F9FAFB'>"
        )
        for h in ["Mô hình","Accuracy","Precision","Recall","F1","Params","Inf. (ms)"]:
            header += (f"<th style='padding:0.75rem 1rem;text-align:left;"
                       f"color:#6B7280;font-size:0.72rem;font-weight:600;"
                       f"text-transform:uppercase;letter-spacing:0.05em;"
                       f"border-bottom:1px solid #E5E7EB'>{h}</th>")
        header += "</tr></thead><tbody>"

        rows = ""
        for name, m in comp.items():
            c = MODEL_CLR.get(name, "#888")
            star = " ★" if name == "BiLSTM" else ""
            rows += (
                f"<tr style='border-bottom:1px solid #F3F4F6'>"
                f"<td style='padding:0.875rem 1rem'>"
                f"<span style='background:{c}18;color:{c};padding:0.2rem 0.65rem;"
                f"border-radius:99px;font-size:0.78rem;font-weight:600'>{name+star}</span></td>"
                f"<td style='padding:0.875rem 1rem;font-weight:600;color:#111827'>"
                f"{m['accuracy']:.4f}</td>"
                f"<td style='padding:0.875rem 1rem;color:#374151'>{m['precision']:.4f}</td>"
                f"<td style='padding:0.875rem 1rem;color:#374151'>{m['recall']:.4f}</td>"
                f"<td style='padding:0.875rem 1rem;color:#374151'>{m['f1']:.4f}</td>"
                f"<td style='padding:0.875rem 1rem;color:#9CA3AF'>{m['num_params']:,}</td>"
                f"<td style='padding:0.875rem 1rem;color:#9CA3AF'>"
                f"{m['inference_ms_per_sample']:.2f} ms</td>"
                f"</tr>"
            )
        st.markdown(header + rows + "</tbody></table></div>", unsafe_allow_html=True)

        # Per-class F1 BiLSTM
        pcf1 = comp.get("BiLSTM", {}).get("per_class_f1", {})
        if pcf1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.78rem;font-weight:600;color:#6B7280;"
                "text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem'>"
                "F1 theo từng nhãn — BiLSTM</p>",
                unsafe_allow_html=True,
            )
            items = sorted(pcf1.items(), key=lambda x: -x[1])
            bars_html = (
                "<div style='background:#fff;border:1px solid #E5E7EB;"
                "border-radius:10px;padding:1.25rem;display:flex;"
                "flex-direction:column;gap:0.6rem'>"
            )
            for lbl, f1 in items:
                _, emoji, c = EMOTION[lbl]
                bars_html += (
                    f"<div style='display:flex;align-items:center;gap:12px'>"
                    f"<span style='width:80px;font-size:0.82rem;font-weight:500;"
                    f"color:#374151;flex-shrink:0'>{emoji} {lbl}</span>"
                    f"<div style='flex:1;background:#F3F4F6;border-radius:99px;height:10px'>"
                    f"<div style='width:{f1*100:.1f}%;background:{c};"
                    f"height:100%;border-radius:99px'></div></div>"
                    f"<span style='width:42px;text-align:right;font-size:0.82rem;"
                    f"font-weight:600;color:#111827'>{f1:.3f}</span>"
                    f"</div>"
                )
            st.markdown(bars_html + "</div>", unsafe_allow_html=True)

    with tab2:
        cm_files = {
            "BiLSTM":     ROOT/"outputs/figures/bilstm_cm.png",
            "DNN+TF-IDF": ROOT/"outputs/figures/dnn_tf_idf_cm.png",
            "XLM-R":      ROOT/"outputs/figures/xlm_r_cm.png",
        }
        c1, c2, c3 = st.columns(3, gap="medium")
        for col, (name, path) in zip([c1,c2,c3], cm_files.items()):
            c = MODEL_CLR.get(name, "#888")
            col.markdown(
                f"<p style='text-align:center;margin-bottom:0.5rem'>"
                f"<span style='background:{c}18;color:{c};padding:0.25rem 0.75rem;"
                f"border-radius:99px;font-size:0.78rem;font-weight:600'>{name}</span></p>",
                unsafe_allow_html=True,
            )
            if path.exists():
                col.image(str(path), use_container_width=True)
            else:
                col.caption("Chưa có hình")

    with tab3:
        metrics = ["Accuracy","Precision","Recall","F1"]
        fig_r = go.Figure()
        for name, m in comp.items():
            vals = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
            c = MODEL_CLR.get(name, "#888")
            fig_r.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=metrics + [metrics[0]],
                fill="toself", name=name,
                line=dict(color=c, width=2.5),
                fillcolor=c, opacity=0.18,
                marker=dict(size=6, color=c),
            ))
        fig_r.update_layout(
            polar=dict(
                radialaxis=dict(range=[0.72, 0.86], showticklabels=True,
                                tickfont=dict(size=9, color="#9CA3AF"),
                                gridcolor="#E5E7EB", linecolor="#E5E7EB"),
                angularaxis=dict(tickfont=dict(size=11.5, color="#374151"),
                                 gridcolor="#E5E7EB", linecolor="#E5E7EB"),
                bgcolor="white",
            ),
            paper_bgcolor="white", showlegend=True,
            legend=dict(font=dict(size=11), bgcolor="white", bordercolor="#E5E7EB",
                        borderwidth=1, x=1.05, y=0.5),
            margin=dict(l=50, r=120, t=20, b=20), height=380,
        )
        st.plotly_chart(fig_r, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Training curves":
    st.markdown("## Training curves")
    st.markdown(
        "<p style='color:#6B7280;margin-top:-0.5rem;margin-bottom:1.5rem;font-size:0.9rem'>"
        "Diễn tiến loss và accuracy qua từng epoch. Điểm đứt quãng = early stopping.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["BiLSTM + Attention ★", "DNN + TF-IDF", "XLM-RoBERTa"])
    configs = [("lstm","BiLSTM","#2563EB"), ("dnn","DNN+TF-IDF","#D97706"), ("xlmr","XLM-R","#059669")]

    for tab, (key, label, mclr) in zip(tabs, configs):
        with tab:
            hist = read_json(f"outputs/logs/{key}_history.json")
            if not hist:
                st.info(f"Chưa có {key}_history.json"); continue

            n = len(hist["train_loss"])
            ep = list(range(1, n+1))

            col_l, col_r = st.columns(2, gap="medium")

            # Loss chart
            with col_l:
                fig = go.Figure()
                for key2, name, dash, color in [
                    ("train_loss","Train","solid", mclr),
                    ("val_loss",  "Validation","dash","#9CA3AF"),
                ]:
                    fig.add_trace(go.Scatter(
                        x=ep, y=hist[key2], name=name,
                        mode="lines+markers",
                        line=dict(width=2, color=color, dash=dash),
                        marker=dict(size=5, color=color),
                        hovertemplate="Epoch %{x}: %{y:.4f}<extra>" + name + "</extra>",
                    ))
                fig.update_layout(
                    title=dict(text="Loss", font=dict(size=12.5, color="#111827"),
                               x=0, xanchor="left"),
                    xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F3F4F6",
                               tickfont=dict(size=10), title_font=dict(size=10.5, color="#6B7280"),
                               zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor="#F3F4F6",
                               tickfont=dict(size=10),
                               title_font=dict(size=10.5, color="#6B7280"), zeroline=False),
                    paper_bgcolor="white", plot_bgcolor="white",
                    height=280, margin=dict(l=45, r=15, t=45, b=40),
                    legend=dict(font=dict(size=10), x=0.98, y=0.98,
                                xanchor="right", yanchor="top",
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="#E5E7EB", borderwidth=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Accuracy chart
            with col_r:
                fig2 = go.Figure()
                for key2, name, dash, color in [
                    ("train_acc","Train","solid", mclr),
                    ("val_acc",  "Validation","dash","#9CA3AF"),
                ]:
                    fig2.add_trace(go.Scatter(
                        x=ep, y=hist[key2], name=name,
                        mode="lines+markers",
                        line=dict(width=2, color=color, dash=dash),
                        marker=dict(size=5, color=color),
                        hovertemplate="Epoch %{x}: %{y:.2%}<extra>" + name + "</extra>",
                    ))
                fig2.update_layout(
                    title=dict(text="Accuracy", font=dict(size=12.5, color="#111827"),
                               x=0, xanchor="left"),
                    xaxis=dict(title="Epoch", showgrid=True, gridcolor="#F3F4F6",
                               tickfont=dict(size=10), title_font=dict(size=10.5, color="#6B7280"),
                               zeroline=False),
                    yaxis=dict(tickformat=".0%", showgrid=True, gridcolor="#F3F4F6",
                               tickfont=dict(size=10),
                               title_font=dict(size=10.5, color="#6B7280"), zeroline=False),
                    paper_bgcolor="white", plot_bgcolor="white",
                    height=280, margin=dict(l=50, r=15, t=45, b=40),
                    legend=dict(font=dict(size=10), x=0.02, y=0.98,
                                xanchor="left", yanchor="top",
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="#E5E7EB", borderwidth=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Stats row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Epochs", hist.get("best_epoch", n),
                      f"/ {n} tổng" if hist.get("best_epoch") else "")
            c2.metric("Best val loss", f"{min(hist['val_loss']):.4f}")
            c3.metric("Test accuracy",
                      f"{hist['test_acc']:.2%}" if "test_acc" in hist else "—")
            c4.metric("Test F1",
                      f"{hist['test_f1']:.2%}" if "test_f1" in hist else "—")


# ══════════════════════════════════════════════════════════════════════════════
# 4 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Ablation study":
    st.markdown("## Ablation Study")
    st.markdown(
        "<p style='color:#6B7280;margin-top:-0.5rem;margin-bottom:1.5rem;font-size:0.9rem'>"
        "Tắt từng thành phần để đo lường đóng góp thực sự của Bidirectionality và Attention.</p>",
        unsafe_allow_html=True,
    )

    abl = read_json("outputs/logs/ablation_results.json")
    if not abl:
        st.warning("Chưa có dữ liệu"); st.stop()

    vals = list(abl.values())
    db   = vals[1]["accuracy"] - vals[0]["accuracy"]  # Bidirectional delta
    da   = vals[2]["accuracy"] - vals[1]["accuracy"]  # Attention delta

    # Contribution cards
    c0, c1, c2 = st.columns(3, gap="medium")
    for col, title, acc, delta, color in [
        (c0, "LSTM (baseline)",      vals[0]["accuracy"], None,    "#6B7280"),
        (c1, "+ Bidirectionality",   vals[1]["accuracy"], db,      "#059669" if db > 0 else "#DC2626"),
        (c2, "+ Attention",          vals[2]["accuracy"], da,      "#059669" if da > 0.002 else "#D97706"),
    ]:
        col.markdown(
            f"<div style='background:#fff;border:1px solid #E5E7EB;border-top:3px solid {color};"
            f"border-radius:10px;padding:1.2rem;text-align:center'>"
            f"<div style='font-size:1.6rem;font-weight:700;color:{color}'>{acc:.2%}</div>"
            f"<div style='font-size:0.8rem;font-weight:600;color:#374151;margin:0.25rem 0 0.15rem'>"
            f"{title}</div>"
            f"<div style='font-size:0.78rem;color:#9CA3AF'>"
            f"{'Δ ' + f'{delta:+.2%}' if delta is not None else 'Điểm khởi đầu'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    names_a = [n.replace(" (proposed)","").replace(" (uni, no-attn)","") for n in abl]
    accs_a  = [m["accuracy"] for m in vals]
    f1s_a   = [m["f1"]       for m in vals]
    bar_clr = ["#D1D5DB", "#93C5FD", "#2563EB"]

    fig_a = go.Figure()
    fig_a.add_trace(go.Bar(
        name="Accuracy", x=names_a, y=accs_a, offsetgroup=0,
        marker_color=bar_clr, text=[f"{v:.3f}" for v in accs_a],
        textposition="outside", textfont=dict(size=10.5, color="#374151"),
    ))
    fig_a.add_trace(go.Bar(
        name="F1 (weighted)", x=names_a, y=f1s_a, offsetgroup=1,
        marker_color=[c + "80" for c in bar_clr],
        text=[f"{v:.3f}" for v in f1s_a],
        textposition="outside", textfont=dict(size=10.5, color="#374151"),
    ))
    fig_a.update_layout(
        barmode="group",
        yaxis=dict(
            range=[min(accs_a+f1s_a) - 0.015, max(accs_a+f1s_a) + 0.025],
            tickformat=".0%", showgrid=True, gridcolor="#F3F4F6",
            tickfont=dict(size=10), zeroline=False,
        ),
        xaxis=dict(tickfont=dict(size=11), zeroline=False),
        paper_bgcolor="white", plot_bgcolor="white",
        height=340, margin=dict(l=55, r=20, t=20, b=40),
        bargap=0.25, bargroupgap=0.06,
        legend=dict(font=dict(size=11), bgcolor="white",
                    bordercolor="#E5E7EB", borderwidth=1),
    )
    st.plotly_chart(fig_a, use_container_width=True)

    # Detail table
    st.markdown("<br>", unsafe_allow_html=True)
    rows_t = ""
    comps_t = [("❌","❌"), ("✅","❌"), ("✅","✅")]
    for (name, m), (bi, att) in zip(abl.items(), comps_t):
        star = " ★" if "Attention" in name else ""
        c = "#2563EB" if "Attention" in name else ("#60A5FA" if "BiLSTM" in name else "#9CA3AF")
        rows_t += (
            f"<tr style='border-bottom:1px solid #F3F4F6'>"
            f"<td style='padding:0.8rem 1rem'>"
            f"<span style='background:{c}18;color:{c};padding:0.2rem 0.65rem;"
            f"border-radius:99px;font-size:0.78rem;font-weight:600'>{name+star}</span></td>"
            f"<td style='padding:0.8rem 1rem;text-align:center;font-size:1rem'>{bi}</td>"
            f"<td style='padding:0.8rem 1rem;text-align:center;font-size:1rem'>{att}</td>"
            f"<td style='padding:0.8rem 1rem;font-weight:600;color:#111827'>{m['accuracy']:.4f}</td>"
            f"<td style='padding:0.8rem 1rem;color:#374151'>{m['f1']:.4f}</td>"
            f"<td style='padding:0.8rem 1rem;color:#9CA3AF'>{m['num_params']:,}</td>"
            f"</tr>"
        )
    th_style = "padding:0.65rem 1rem;text-align:left;color:#6B7280;font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;border-bottom:1.5px solid #E5E7EB;"
    st.markdown(
        f"<div style='background:#fff;border:1px solid #E5E7EB;border-radius:10px;overflow:hidden'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.875rem'>"
        f"<thead style='background:#F9FAFB'><tr>"
        f"<th style='{th_style}'>Biến thể</th>"
        f"<th style='{th_style};text-align:center'>Bidirectional</th>"
        f"<th style='{th_style};text-align:center'>Attention</th>"
        f"<th style='{th_style}'>Accuracy</th>"
        f"<th style='{th_style}'>F1</th>"
        f"<th style='{th_style}'>Params</th>"
        f"</tr></thead><tbody>{rows_t}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    # Insight
    st.markdown("<br>", unsafe_allow_html=True)
    insight_body = (
        f"Bidirectionality cải thiện **{db:+.2%}** accuracy bằng cách mã hoá ngữ cảnh "
        f"hai chiều — mỗi token nhận thông tin từ cả trái lẫn phải. "
    )
    if abs(da) < 0.005:
        insight_body += (
            f"Attention cho đóng góp nhỏ ({da:+.2%}) trên tập dữ liệu ngắn này, "
            f"nhưng vẫn mang lại **khả năng giải thích** khi trực quan hoá trọng số."
        )
    else:
        insight_body += f"Attention tiếp tục cải thiện thêm **{da:+.2%}**."

    st.info("💡 " + insight_body)


# ══════════════════════════════════════════════════════════════════════════════
# 5 — KIẾN TRÚC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Kiến trúc mô hình":
    st.markdown("## Kiến trúc mô hình")
    st.markdown(
        "<p style='color:#6B7280;margin-top:-0.5rem;margin-bottom:1.5rem;font-size:0.9rem'>"
        "Chi tiết 3 mô hình: Baseline → Proposed → State-of-the-art comparison.</p>",
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["BiLSTM + Attention ★", "DNN + TF-IDF", "XLM-RoBERTa"])

    with t1:
        st.markdown("""
**Mô hình đề xuất** — phù hợp với đặc trưng văn bản ngắn, ít tài nguyên.

| Thành phần | Chi tiết |
|---|---|
| Embedding | vocab × 128d, padding_idx=0 |
| Encoder | BiLSTM · 2 layers · hidden=256 · dropout=0.3 |
| Attention | Bahdanau additive · score = v·tanh(W·h) |
| Classifier | Linear 512→256 → LayerNorm → GELU → Dropout → Linear 256→7 |
| Tham số | **2.919.047** |
| Huấn luyện | AdamW · lr=1e-3 · CosineAnnealing · patience=5 |
""")
        st.code("""
Input: [t₁, t₂, …, tₙ]           # token indices
    │
Embedding(vocab, 128) + Dropout(0.3)
    │  shape: (B, L, 128)
    │
BiLSTM(128 → 256, num_layers=2)
    │  Forward : h₁→ h₂→ … hₙ→
    │  Backward: hₙ← … h₂← h₁←
    │  Output  : [hᵢ→ ‖ hᵢ←]   shape: (B, L, 512)
    │
Attention (Bahdanau)
    │  eᵢ = v · tanh(W · hᵢ)    score
    │  αᵢ = softmax(e)            weight
    │  c  = Σ αᵢ · hᵢ            context  (B, 512)
    │
Linear(512→256) → LayerNorm → GELU → Dropout(0.3)
    │
Linear(256→7)   →  logits  →  CrossEntropyLoss
""", language="text")

    with t2:
        st.markdown("""
**Baseline** — biểu diễn thống kê, không nắm thứ tự từ.

| Thành phần | Chi tiết |
|---|---|
| Vectoriser | TF-IDF · 10.868 features · bigram · sublinear_tf |
| Layer 1 | Linear(10868→512) → BN → ReLU → Dropout(0.3) |
| Layer 2 | Linear(512→256) → BN → ReLU → Dropout(0.3) |
| Layer 3 | Linear(256→128) → BN → ReLU → Dropout(0.3) |
| Output | Linear(128→7) |
| Tham số | **5.731.847** |
| Huấn luyện | AdamW · lr=1e-3 · CosineAnnealing · patience=5 |
""")

    with t3:
        st.markdown("""
**Upper-bound** — mô hình Transformer đa ngôn ngữ pre-trained.

| Thành phần | Chi tiết |
|---|---|
| Backbone | `xlm-roberta-base` · 12 layers · hidden=768 |
| Head | Dropout(0.1) → Linear(768→7) |
| Tham số | **278.049.031** |
| Fine-tuning | AdamW · lr=2e-5 · warmup 10% · patience=5 |
""")
        st.info(
            "XLM-RoBERTa được pre-train trên 2.5TB văn bản đa ngôn ngữ, "
            "bao gồm tiếng Việt. Đây là **upper bound** — cho thấy giới hạn trên "
            "của bài toán với dữ liệu hiện có."
        )
