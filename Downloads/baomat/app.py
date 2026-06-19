import streamlit as st
import pandas as pd
import numpy as np
import time
import hashlib
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DDoS Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; margin: 4px 0; }
.metric-label { font-size: 0.82rem; color: #8b949e; text-transform: uppercase; letter-spacing: .05em; }
.alert-danger {
    background: #2d1b1b; border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 8px; padding: 16px 20px; margin: 8px 0;
    color: #f85149; font-weight: 600;
}
.alert-safe {
    background: #1b2d1b; border: 1px solid #3fb950;
    border-left: 4px solid #3fb950;
    border-radius: 8px; padding: 16px 20px; margin: 8px 0;
    color: #3fb950; font-weight: 600;
}
.ip-badge {
    display: inline-block;
    background: #21262d; border: 1px solid #30363d;
    border-radius: 6px; padding: 2px 10px;
    font-family: monospace; font-size: 0.85rem; color: #e6edf3;
    margin: 2px 4px;
}
.ip-blocked {
    background: #2d1b1b; border-color: #f85149; color: #f85149;
}
h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Danh sách đặc trưng chuẩn ─────────────────────────────────────────────────
FEATURE_NAMES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]
FEATURE_NAMES = [x.strip() for x in FEATURE_NAMES]
LABEL_COL = 'Label'


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất đúng bộ FEATURE_NAMES từ một DataFrame log mạng bất kỳ."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    X = pd.DataFrame(0.0, index=df.index, columns=FEATURE_NAMES)
    col_mapping_strict = {c.lower().replace(" ", ""): c for c in df.columns}

    for col in FEATURE_NAMES:
        col_clean = col.lower().replace(" ", "")
        if col_clean in col_mapping_strict:
            actual_col = col_mapping_strict[col_clean]
            X[col] = pd.to_numeric(df[actual_col], errors='coerce').fillna(0.0)
        else:
            fuzzy_matched = [c for c in df.columns if col.lower() in c.lower() or c.lower() in col.lower()]
            if fuzzy_matched:
                X[col] = pd.to_numeric(df[fuzzy_matched[0]], errors='coerce').fillna(0.0)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)
    X = X.clip(-1e15, 1e15).astype(np.float64)
    return X


@st.cache_resource(show_spinner=False)
def train_model_from_df(file_hash: str, df: pd.DataFrame):
    """Train Random Forest từ DataFrame log mạng thật (phải có cột Label)."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    label_col_actual = [c for c in df.columns if c.lower() == 'label']
    if not label_col_actual:
        raise ValueError("Không tìm thấy cột 'Label' trong file dữ liệu! Vui lòng kiểm tra lại file CSV.")

    raw_labels = df[label_col_actual[0]].fillna('BENIGN').astype(str).str.strip().str.upper()
    y = np.where(raw_labels == 'BENIGN', 0, 1).astype(int)

    X = extract_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120, max_depth=18, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
        "n_benign": int((y == 0).sum()),
        "n_attack": int((y == 1).sum()),
    }
    return model, metrics


def fake_ip():
    import random
    return f"{random.randint(1,254)}.{random.randint(0,254)}.{random.randint(0,254)}.{random.randint(1,254)}"


def sample_row_from_training(X_train: pd.DataFrame, y_train: np.ndarray, want_ddos: bool, rng):
    """Lấy ngẫu nhiên một dòng dữ liệu THẬT từ tập train (không sinh giả)
    để mô phỏng giám sát thời gian thực dựa trên dữ liệu đã train."""
    idx_pool = np.where(y_train == (1 if want_ddos else 0))[0]
    if len(idx_pool) == 0:
        idx_pool = np.arange(len(y_train))
    i = rng.choice(idx_pool)
    return X_train.iloc[i].values.astype(np.float64)


# ─── Sidebar: Upload & Train ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ DDoS Detection")
    st.markdown("**AI-Powered Network Guardian**")
    st.divider()
    page = st.radio(
        "Chọn chức năng",
        ["🏠 Tổng quan", "📁 Phân tích file CSV", "🔴 Giám sát thời gian thực"],
        label_visibility="collapsed",
    )
    st.divider()

    st.markdown("### 🧠 Dữ liệu & Huấn luyện")
    train_file = st.file_uploader(
        "Upload file CSV để TRAIN (phải có cột 'Label')",
        type=["csv"],
        key="train_uploader",
    )

    if train_file is not None:
        file_bytes = train_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        same_file = st.session_state.get("trained_file_hash") == file_hash
        if same_file and "trained_model" in st.session_state:
            st.success("✓ File này đã được train trong phiên hiện tại.")

        if st.button("🚀 Train mô hình từ file này", use_container_width=True):
            try:
                with st.spinner("⚙️ Đang đọc dữ liệu và huấn luyện Random Forest..."):
                    train_file.seek(0)
                    train_df = pd.read_csv(train_file)
                    model, metrics = train_model_from_df(file_hash, train_df)

                    # Lưu lại train/test thật để dùng cho mô phỏng real-time
                    train_df.columns = train_df.columns.str.strip()
                    label_col_actual = [c for c in train_df.columns if c.lower() == 'label']
                    raw_labels = train_df[label_col_actual[0]].fillna('BENIGN').astype(str).str.strip().str.upper()
                    y_full = np.where(raw_labels == 'BENIGN', 0, 1).astype(int)
                    X_full = extract_features(train_df)

                    ip_col_candidates = [c for c in train_df.columns if 'ip' in c.lower() and 'src' in c.lower()] \
                        or [c for c in train_df.columns if c.lower() in ('source ip', 'src ip', 'srcip')]

                    X_tr, _, y_tr, _ = train_test_split(
                        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
                    )

                    st.session_state["trained_model"] = model
                    st.session_state["trained_metrics"] = metrics
                    st.session_state["trained_file_hash"] = file_hash
                    st.session_state["X_train_real"] = X_tr.reset_index(drop=True)
                    st.session_state["y_train_real"] = y_tr
                    st.session_state["train_filename"] = train_file.name

                st.success(f"✅ Train xong! Accuracy: {metrics['accuracy']*100:.2f}%")
            except Exception as e:
                st.error(f"❌ Lỗi khi train: {e}")

    st.divider()
    if "trained_model" in st.session_state:
        st.success("Mô hình đã sẵn sàng ✓")
        st.caption(f"Nguồn dữ liệu train: `{st.session_state.get('train_filename', '—')}`")
        m = st.session_state["trained_metrics"]
        st.caption(f"Train: {m['n_train']:,} dòng · Test (cùng file): {m['n_test']:,} dòng")
        st.caption(f"BENIGN: {m['n_benign']:,} · Attack: {m['n_attack']:,}")

        st.divider()
        st.markdown("### 🧪 Đánh giá trên file CSV KHÁC")
        st.caption("Test mô hình trên dữ liệu chưa từng thấy để có số liệu thực tế, tránh data leakage.")
        eval_file = st.file_uploader(
            "Upload file CSV khác để đánh giá (cần cột 'Label')",
            type=["csv"],
            key="eval_uploader",
        )
        if eval_file is not None and st.button("📊 Đánh giá trên file này", use_container_width=True):
            try:
                with st.spinner("⚙️ Đang đánh giá trên dữ liệu mới..."):
                    eval_df = pd.read_csv(eval_file)
                    eval_df.columns = eval_df.columns.str.strip()
                    label_col_actual = [c for c in eval_df.columns if c.lower() == 'label']
                    if not label_col_actual:
                        st.error("❌ File này không có cột 'Label' nên không thể đánh giá.")
                    else:
                        raw_labels_e = eval_df[label_col_actual[0]].fillna('BENIGN').astype(str).str.strip().str.upper()
                        y_e = np.where(raw_labels_e == 'BENIGN', 0, 1).astype(int)
                        X_e = extract_features(eval_df)
                        y_pred_e = model.predict(X_e.values)

                        ext_metrics = {
                            "accuracy": accuracy_score(y_e, y_pred_e),
                            "precision": precision_score(y_e, y_pred_e, zero_division=0),
                            "recall": recall_score(y_e, y_pred_e, zero_division=0),
                            "f1": f1_score(y_e, y_pred_e, zero_division=0),
                            "cm": confusion_matrix(y_e, y_pred_e).tolist(),
                            "n_total": int(len(y_e)),
                            "n_benign": int((y_e == 0).sum()),
                            "n_attack": int((y_e == 1).sum()),
                        }
                        st.session_state["external_eval_metrics"] = ext_metrics
                        st.session_state["external_eval_filename"] = eval_file.name
                st.success(f"✅ Đánh giá xong trên `{eval_file.name}`!")
            except Exception as e:
                st.error(f"❌ Lỗi khi đánh giá: {e}")

        if "external_eval_metrics" in st.session_state:
            em = st.session_state["external_eval_metrics"]
            st.caption(f"Kết quả trên `{st.session_state['external_eval_filename']}` ({em['n_total']:,} dòng):")
            st.caption(f"Acc: {em['accuracy']*100:.2f}% · Prec: {em['precision']*100:.2f}% · Rec: {em['recall']*100:.2f}% · F1: {em['f1']*100:.2f}%")
    else:
        st.warning("⚠️ Chưa có mô hình. Hãy upload file CSV thật và bấm **Train mô hình**.")

    st.divider()
    st.markdown("**Mô hình**")
    st.markdown("`Random Forest · 120 cây`")
    st.markdown("**Phiên bản**")
    st.markdown("`v1.1.0`")

model = st.session_state.get("trained_model")
eval_metrics = st.session_state.get("trained_metrics")

if model is None:
    st.markdown("# 🛡️ Hệ thống Phát hiện Tấn công DDoS")
    st.markdown("Phân tích gói tin mạng theo thời gian thực · Phân loại bằng Machine Learning")
    st.divider()
    st.info(
        "👈 Vui lòng upload **file CSV dữ liệu mạng thật** (có cột `Label`, ví dụ định dạng "
        "CICIDS2017) ở thanh bên trái, sau đó bấm **Train mô hình từ file này** để bắt đầu."
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Tổng quan
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Tổng quan":
    st.markdown("# 🛡️ Hệ thống Phát hiện Tấn công DDoS")
    st.markdown("Phân tích gói tin mạng theo thời gian thực · Phân loại bằng Machine Learning")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    metrics_display = [
        (c1, "Accuracy",  f"{eval_metrics['accuracy']*100:.2f}%",  "#58a6ff"),
        (c2, "Precision", f"{eval_metrics['precision']*100:.2f}%", "#3fb950"),
        (c3, "Recall",    f"{eval_metrics['recall']*100:.2f}%",    "#d2a8ff"),
        (c4, "F1-Score",  f"{eval_metrics['f1']*100:.2f}%",        "#ffa657"),
    ]
    for col, label, val, color in metrics_display:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔄 Luồng hoạt động hệ thống")
    col_arch = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
    steps = [("📄", "File CSV thật", "Dữ liệu mạng"), ("⚙️", "Trích xuất", f"{len(FEATURE_NAMES)} đặc trưng"), ("🤖", "Mô hình AI", "Random Forest"), ("🚦", "Kết quả", "BENIGN / DDoS")]
    arrows = [1, 3, 5]
    si = 0
    for i, col in enumerate(col_arch):
        if i in arrows:
            col.markdown("<div style='text-align:center;font-size:1.8rem;color:#8b949e;padding-top:30px'>→</div>", unsafe_allow_html=True)
        else:
            icon, title, sub = steps[si]; si += 1
            col.markdown(f"""
            <div class="metric-card" style="padding:20px 10px">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="font-weight:600;color:#e6edf3;margin:6px 0 2px">{title}</div>
                <div style="font-size:0.8rem;color:#8b949e">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Ma trận nhầm lẫn (Confusion Matrix) trên tập test — cùng file train")
    cm = np.array(eval_metrics["cm"])
    fig_cm = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Dự đoán", y="Thực tế", color="Số lượng"),
        x=["BENIGN", "DDoS"], y=["BENIGN", "DDoS"],
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    if "external_eval_metrics" in st.session_state:
        em = st.session_state["external_eval_metrics"]
        st.divider()
        st.markdown(f"### 🧪 Đánh giá trên file KHÁC: `{st.session_state['external_eval_filename']}`")
        st.caption("Số liệu này phản ánh thực tế hơn vì mô hình chưa từng thấy dữ liệu này khi train.")

        e1, e2, e3, e4 = st.columns(4)
        ext_display = [
            (e1, "Accuracy",  f"{em['accuracy']*100:.2f}%",  "#58a6ff"),
            (e2, "Precision", f"{em['precision']*100:.2f}%", "#3fb950"),
            (e3, "Recall",    f"{em['recall']*100:.2f}%",    "#d2a8ff"),
            (e4, "F1-Score",  f"{em['f1']*100:.2f}%",        "#ffa657"),
        ]
        for col, label, val, color in ext_display:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color}">{val}</div>
                </div>""", unsafe_allow_html=True)

        cm_e = np.array(em["cm"])
        fig_cm_e = px.imshow(
            cm_e, text_auto=True,
            labels=dict(x="Dự đoán", y="Thực tế", color="Số lượng"),
            x=["BENIGN", "DDoS"], y=["BENIGN", "DDoS"],
            color_continuous_scale="Reds",
        )
        fig_cm_e.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=350, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm_e, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Phân tích file CSV
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Phân tích file CSV":
    st.markdown("# 📁 Phân tích file log mạng")
    st.markdown("Upload file CSV chứa log lưu lượng mạng thật để mô hình đã train phân tích hàng loạt")

    uploaded = st.file_uploader("Chọn file CSV cần phân tích", type=["csv"], key="analysis_uploader")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.session_state["csv_df"] = df

    df = st.session_state.get("csv_df", None)

    if df is None:
        st.info("👆 Hãy upload một file CSV log mạng để phân tích.")
    else:
        st.divider()
        df.columns = df.columns.str.strip()
        st.markdown(f"**{len(df):,} dòng** đã tải · {df.shape[1]} cột")

        X_batch = extract_features(df)
        X_raw = X_batch.values.astype(np.float64)

        preds = model.predict(X_raw)
        probs = model.predict_proba(X_raw)[:, 1]

        df["Prediction"] = np.where(preds == 1, "🚨 DDoS", "✅ BENIGN")
        df["DDoS Probability"] = (probs * 100).round(1)

        actual_label_col = [c for c in df.columns if c.lower() == 'label']
        if actual_label_col:
            real_labels = df[actual_label_col[0]].astype(str).str.strip().str.upper()
            real_ddos_count = (~real_labels.str.contains('BENIGN')).sum()
            real_benign_count = real_labels.str.contains('BENIGN').sum()
            st.info(f"📋 Nhãn thật trong file: **{real_benign_count:,} BENIGN** · **{real_ddos_count:,} DDoS/Attack**")

        n_ddos = int(preds.sum())
        n_normal = len(preds) - n_ddos

        s1, s2, s3 = st.columns(3)
        s1.markdown(f'<div class="metric-card"><div class="metric-label">Tổng gói tin</div><div class="metric-value" style="color:#58a6ff">{len(preds):,}</div></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="metric-card"><div class="metric-label">Bình thường</div><div class="metric-value" style="color:#3fb950">{n_normal:,}</div></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="metric-card"><div class="metric-label">DDoS phát hiện</div><div class="metric-value" style="color:#f85149">{n_ddos:,}</div></div>', unsafe_allow_html=True)

        st.divider()
        ch1, ch2 = st.columns(2)
        with ch1:
            st.markdown("**Phân bố kết quả**")
            fig_pie = go.Figure(go.Pie(labels=["BENIGN", "DDoS"], values=[n_normal, n_ddos], marker_colors=["#3fb950", "#f85149"], hole=0.5))
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=280, margin=dict(t=10,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            st.markdown("**Phân phối xác suất DDoS**")
            fig_hist = px.histogram(x=probs * 100, nbins=40, labels={"x": "DDoS Probability (%)"}, color_discrete_sequence=["#d2a8ff"])
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=280, margin=dict(t=10,b=10), yaxis=dict(gridcolor="#30363d"), xaxis=dict(gridcolor="#30363d"))
            st.plotly_chart(fig_hist, use_container_width=True)

        ip_col_candidates = [c for c in df.columns if c.lower() in
                              ('source ip', 'src ip', 'srcip', 'source_ip')]
        if ip_col_candidates and n_ddos > 0:
            ip_col = ip_col_candidates[0]
            st.markdown("**🚫 IP bị chặn (phát hiện DDoS)**")
            blocked = df[preds == 1][ip_col].value_counts().head(10)
            badges = "".join(f'<span class="ip-badge ip-blocked">⛔ {ip} ({cnt})</span>' for ip, cnt in blocked.items())
            st.markdown(badges, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Chi tiết từng dòng**")

        filter_choice = st.radio(
            "Lọc theo kết quả dự đoán",
            ["Tất cả", "✅ BENIGN", "🚨 DDoS"],
            horizontal=True,
            key="prediction_filter",
        )

        show_cols = (ip_col_candidates[:1] if ip_col_candidates else []) + ["Prediction", "DDoS Probability"]
        show_cols += [c for c in FEATURE_NAMES if c in df.columns][:6]

        filtered_df = df
        if filter_choice == "✅ BENIGN":
            filtered_df = df[df["Prediction"] == "✅ BENIGN"]
        elif filter_choice == "🚨 DDoS":
            filtered_df = df[df["Prediction"] == "🚨 DDoS"]

        st.caption(f"Hiển thị {len(filtered_df):,} / {len(df):,} dòng")
        st.dataframe(
            filtered_df[show_cols].head(500), # Lấy 500 dòng đầu tiên theo đúng thứ tự file CSV
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Giám sát thời gian thực (phát lại dữ liệu THẬT từ tập train)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔴 Giám sát thời gian thực":
    st.markdown("# 🔴 Giám sát thời gian thực")
    st.markdown("Phát lại các dòng dữ liệu THẬT từ tập train — AI phân loại mỗi 0.4 giây")

    if "X_train_real" not in st.session_state:
        st.warning("⚠️ Chưa có dữ liệu train thật trong phiên này. Hãy train mô hình ở sidebar trước.")
        st.stop()

    X_train_real = st.session_state["X_train_real"]
    y_train_real = st.session_state["y_train_real"]

    if "rt_history" not in st.session_state: st.session_state.rt_history = []
    if "rt_blocked_ips" not in st.session_state: st.session_state.rt_blocked_ips = {}
    if "rt_running" not in st.session_state: st.session_state.rt_running = False

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 3])
    if ctrl1.button("▶ Bắt đầu", use_container_width=True, type="primary"): st.session_state.rt_running = True
    if ctrl2.button("⏹ Dừng", use_container_width=True): st.session_state.rt_running = False
    if ctrl3.button("🗑️ Xóa lịch sử", use_container_width=True):
        st.session_state.rt_history = []
        st.session_state.rt_blocked_ips = {}

    attack_mode = st.toggle("🔥 Kích hoạt chế độ tấn công (ưu tiên lấy mẫu DDoS thật từ tập train)", value=False)
    ddos_ratio = 0.75 if attack_mode else 0.12

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    ph_total = m1.empty(); ph_ddos = m2.empty(); ph_safe = m3.empty(); ph_rate = m4.empty()
    chart_ph = st.empty(); log_ph = st.empty()

    def render_metrics():
        hist = st.session_state.rt_history
        total = len(hist)
        ddos_n = sum(1 for h in hist if h["pred"] == 1)
        safe_n = total - ddos_n
        rate = f"{ddos_n/total*100:.1f}%" if total else "0%"
        ph_total.markdown(f'<div class="metric-card"><div class="metric-label">Tổng gói tin</div><div class="metric-value" style="color:#58a6ff">{total}</div></div>', unsafe_allow_html=True)
        ph_ddos.markdown(f'<div class="metric-card"><div class="metric-label">DDoS</div><div class="metric-value" style="color:#f85149">{ddos_n}</div></div>', unsafe_allow_html=True)
        ph_safe.markdown(f'<div class="metric-card"><div class="metric-label">An toàn</div><div class="metric-value" style="color:#3fb950">{safe_n}</div></div>', unsafe_allow_html=True)
        ph_rate.markdown(f'<div class="metric-card"><div class="metric-label">Tỉ lệ DDoS</div><div class="metric-value" style="color:#ffa657">{rate}</div></div>', unsafe_allow_html=True)

    def render_chart():
        hist = st.session_state.rt_history[-60:]
        if not hist: return
        times = [h["t"] for h in hist]
        vals  = [h["prob"] * 100 for h in hist]
        colors = ["#f85149" if h["pred"] == 1 else "#3fb950" for h in hist]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=vals, mode="lines+markers", line=dict(color="#58a6ff", width=1.5), marker=dict(color=colors, size=7)))
        fig.add_hline(y=50, line_dash="dash", line_color="#ffa657", annotation_text="Ngưỡng 50%")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=260, margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(range=[-5, 105], title="Xác suất DDoS (%)", gridcolor="#30363d"), xaxis=dict(gridcolor="#30363d", tickangle=-30), showlegend=False)
        chart_ph.plotly_chart(fig, use_container_width=True)

    def render_log():
        hist = st.session_state.rt_history[-8:][::-1]
        lines = []
        for h in hist:
            icon = "🚨" if h["pred"] == 1 else "✅"
            cls  = "alert-danger" if h["pred"] == 1 else "alert-safe"
            lines.append(f'<div class="{cls}" style="padding:8px 14px;margin:3px 0">{icon} [{h["t"]}] <span class="ip-badge">IP {h["ip"]}</span> → {"DDoS" if h["pred"] == 1 else "BENIGN"} ({h["prob"]*100:.1f}%)</div>')
        log_ph.markdown("".join(lines), unsafe_allow_html=True)

    render_metrics(); render_chart()

    if st.session_state.rt_running:
        rng = np.random.default_rng()
        for _ in range(40):
            if not st.session_state.rt_running: break
            want_ddos = rng.random() < ddos_ratio
            v = sample_row_from_training(X_train_real, y_train_real, want_ddos, rng)

            feat = v.reshape(1, -1)
            pred = int(model.predict(feat)[0])
            prob = float(model.predict_proba(feat)[0][1])
            ip = fake_ip()
            ts = datetime.now().strftime("%H:%M:%S")

            if pred == 1: st.session_state.rt_blocked_ips[ip] = ts
            st.session_state.rt_history.append({"t": ts, "ip": ip, "pred": pred, "prob": prob})

            if len(st.session_state.rt_history) > 200:
                st.session_state.rt_history = st.session_state.rt_history[-200:]

            render_metrics(); render_chart(); render_log()
            time.sleep(0.4)
        st.rerun()
    else:
        render_log()
        if st.session_state.rt_blocked_ips:
            st.divider()
            st.markdown("**🚫 Danh sách IP bị chặn trong phiên này**")
            badges = "".join(f'<span class="ip-badge ip-blocked">⛔ {ip} ({ts})</span>' for ip, ts in list(st.session_state.rt_blocked_ips.items())[-20:])
            st.markdown(badges, unsafe_allow_html=True)