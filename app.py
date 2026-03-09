"""
app.py — Bank Churn Analysis Portfolio
Reinaldi Santoso | Data Science Bootcamp – Dibimbing.id

Cara menjalankan:
  1. python train_model.py        ← generate pickle (sekali saja)
  2. python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import (
    confusion_matrix, classification_report,
    recall_score, precision_score, f1_score, accuracy_score
)

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Analysis | Reinaldi Santoso",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .insight-box {
        background: #f0f7ff;
        border-left: 4px solid #1a73e8;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0;
        font-size: 0.95rem;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0;
        font-size: 0.95rem;
    }
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #2e7d32;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0;
        font-size: 0.95rem;
    }
    .danger-box {
        background: #ffebee;
        border-left: 4px solid #c62828;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.6rem 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD PICKLE ARTIFACTS
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load semua file pickle yang dihasilkan train_model.py"""
    try:
        with open("model/xgb_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        with open("model/cat_info.pkl", "rb") as f:
            cat_info = pickle.load(f)
        return model, scaler, feature_names, cat_info, True
    except FileNotFoundError:
        return None, None, None, None, False

model, scaler, feature_names, cat_info, model_loaded = load_model_artifacts()

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("bank_churn_data.csv")
    df['attrition_flag'] = df['attrition_flag'].map(
        {'Existing Customer': 0, 'Attrited Customer': 1}
    )
    df = df.set_index('user_id')
    return df

@st.cache_data
def get_model_data():
    """Kembalikan data train/test yang sudah di-preprocess, konsisten dengan pickle"""
    df = load_data()
    cat_cols = ['gender', 'education_level', 'marital_status',
                'income_category', 'card_category']
    df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df_model.drop('attrition_flag', axis=1)
    y = df_model['attrition_flag']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_sc = scaler.transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train, X_test, X_train_sc, X_test_sc, y_train, y_test

try:
    df = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bank-building.png", width=70)
    st.markdown("## 🏦 Bank Churn Dashboard")
    st.markdown("**Reinaldi Santoso**  \nData Science Bootcamp")
    st.markdown("---")

    page = st.selectbox(
        "📌 Navigasi Halaman",
        [
            "🏠 Home",
            "📊 EDA – Kategorikal",
            "📈 EDA – Numerikal",
            "🤖 Modelling & Evaluasi",
            "💰 Profitabilitas",
            "🔍 SHAP – Global Interpretation",
            "👤 Prediksi Nasabah Baru",
            "❓ Reflection"
        ]
    )

    st.markdown("---")

    # Status model
    if model_loaded:
        st.success("✅ Model pickle loaded")
    else:
        st.error("❌ Model belum ada\nJalankan `train_model.py` dulu!")

    st.markdown("#### ℹ️ Dataset")
    if data_loaded:
        st.markdown(f"- **{len(df):,}** nasabah\n- **20** fitur\n- Churn rate: **{df['attrition_flag'].mean()*100:.1f}%**")


# ════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏦 Bank Customer Churn Analysis</div>', unsafe_allow_html=True)
    st.caption("Prediksi & Strategi Retensi Nasabah | Reinaldi Santoso – Data Science Bootcamp Dibimbing.id")

    if not data_loaded:
        st.error("⚠️ File `bank_churn_data.csv` tidak ditemukan di folder yang sama dengan `app.py`.")
        st.stop()

    total    = len(df)
    churned  = int(df['attrition_flag'].sum())
    stay     = total - churned
    churn_rt = churned / total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Nasabah",    f"{total:,}")
    col2.metric("✅ Nasabah Bertahan", f"{stay:,}")
    col3.metric("❌ Nasabah Churn",    f"{churned:,}")
    col4.metric("📉 Churn Rate",       f"{churn_rt:.1f}%")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Komposisi Nasabah")
        fig = px.pie(names=["Bertahan", "Churn"], values=[stay, churned],
                     color_discrete_sequence=["#1a73e8", "#e53935"], hole=0.45)
        fig.update_traces(textinfo='percent+label', textposition='outside')
        fig.update_layout(margin=dict(t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Ringkasan Proyek")
        st.markdown("""
        <div class="insight-box">
        <b>🎯 Tujuan:</b> Mengidentifikasi nasabah berisiko churn agar bank dapat melakukan 
        tindakan retensi lebih awal dan efisien.
        </div>
        <div class="insight-box">
        <b>🔬 Alur:</b> EDA → Feature Engineering → Multi-Model Comparison → 
        Profit Analysis → SHAP Interpretation → Prediksi Real-time
        </div>
        <div class="success-box">
        <b>🏆 Champion Model:</b> XGBoost<br>
        Recall <b>91%+</b> | Precision <b>86%+</b> | Net Profit <b>$2,992</b> / 2.026 nasabah
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 Preview Dataset")
    show_raw = st.checkbox("Tampilkan Raw Data")
    if show_raw:
        n = st.slider("Jumlah baris", 5, 100, 10)
        st.dataframe(df.head(n), use_container_width=True)

    if st.checkbox("Tampilkan Statistik Deskriptif"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE: EDA KATEGORIKAL
# ════════════════════════════════════════════════════════
elif page == "📊 EDA – Kategorikal":
    st.markdown('<div class="main-header">📊 EDA – Variabel Kategorikal vs Churn</div>', unsafe_allow_html=True)
    if not data_loaded: st.error("File CSV tidak ditemukan."); st.stop()

    cat_cols = ['gender', 'education_level', 'marital_status', 'income_category', 'card_category']
    selected = st.selectbox("🔽 Pilih Variabel", cat_cols)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = px.histogram(
            df.assign(Status=df['attrition_flag'].map({0: 'Bertahan', 1: 'Churn'})),
            x=selected, color='Status', barmode='group',
            color_discrete_map={'Bertahan': '#1a73e8', 'Churn': '#e53935'},
            title=f'Distribusi Churn – {selected.replace("_"," ").title()}'
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_rate = (df.groupby(selected)['attrition_flag'].mean()
                      .mul(100).round(2).reset_index())
        churn_rate.columns = [selected, 'Churn Rate (%)']
        churn_rate = churn_rate.sort_values('Churn Rate (%)', ascending=False)
        fig2 = px.bar(churn_rate, x='Churn Rate (%)', y=selected,
                      orientation='h', color='Churn Rate (%)',
                      color_continuous_scale='Reds', text='Churn Rate (%)')
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    insights = {
        'gender':          "👩 Wanita (17.4%) memiliki churn rate lebih tinggi dari pria (14.6%). Selisih ~3% menunjukkan perbedaan kebutuhan layanan.",
        'education_level': "🎓 Nasabah bergelar <b>Doctorate</b> memiliki churn tertinggi (~21%) — kemungkinan lebih kritis dan banyak alternatif bank lain.",
        'marital_status':  "💍 Pola churn merata (15–17%). Status pernikahan bukan pembeda utama churn.",
        'income_category': "💵 Pola <b>U-Shape</b>: pendapatan tertinggi ($120K+) dan terendah (<$40K) sama-sama churn ~17%. Yang kaya cari layanan eksklusif; yang rendah terbebani biaya.",
        'card_category':   "💳 Pemegang kartu <b>Platinum</b> churn tertinggi (25%)! Mereka tidak puas dengan benefit atau membandingkan dengan penawaran bank lain."
    }
    st.markdown(f'<div class="warning-box">💡 <b>Insight:</b> {insights[selected]}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: EDA NUMERIKAL
# ════════════════════════════════════════════════════════
elif page == "📈 EDA – Numerikal":
    st.markdown('<div class="main-header">📈 EDA – Variabel Numerikal vs Churn</div>', unsafe_allow_html=True)
    if not data_loaded: st.error("File CSV tidak ditemukan."); st.stop()

    num_map = {
        'total_trans_ct':         'Jumlah Transaksi (12 Bln)',
        'total_trans_amt':        'Total Nominal Transaksi ($)',
        'total_revolving_bal':    'Saldo Bergulir ($)',
        'contacts_count_12_mon':  'Kontak CS (12 Bln)',
        'customer_age':           'Usia Nasabah',
        'months_on_book':         'Lama Nasabah (Bulan)',
        'credit_limit':           'Credit Limit ($)',
        'avg_utilization_ratio':  'Rasio Utilisasi Kartu'
    }
    selected = st.selectbox("🔽 Pilih Variabel", list(num_map.keys()), format_func=lambda x: num_map[x])
    chart    = st.radio("📊 Tipe Chart", ["Boxplot", "Histogram", "Violin"], horizontal=True)

    df_plot = df.assign(Status=df['attrition_flag'].map({0: 'Bertahan', 1: 'Churn'}))
    cmap = {'Bertahan': '#1a73e8', 'Churn': '#e53935'}

    if chart == "Boxplot":
        fig = px.box(df_plot, x='Status', y=selected, color='Status',
                     color_discrete_map=cmap, points='outliers',
                     title=f'{num_map[selected]} vs Status Churn')
    elif chart == "Histogram":
        fig = px.histogram(df_plot, x=selected, color='Status', barmode='overlay',
                           color_discrete_map=cmap, opacity=0.7, nbins=40,
                           title=f'Distribusi {num_map[selected]}')
    else:
        fig = px.violin(df_plot, x='Status', y=selected, color='Status',
                        color_discrete_map=cmap, box=True,
                        title=f'{num_map[selected]} – Violin')
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    ms = df[df['attrition_flag'] == 0][selected].mean()
    mc = df[df['attrition_flag'] == 1][selected].mean()
    c1.metric("📗 Rata-rata Bertahan", f"{ms:,.1f}")
    c2.metric("📕 Rata-rata Churn",    f"{mc:,.1f}")
    c3.metric("📐 Selisih",            f"{(mc-ms)/ms*100:+.1f}%")

    insights = {
        'total_trans_ct':        "🔑 <b>Fitur terpenting!</b> Nasabah churn hanya 44 transaksi/tahun vs bertahan 68. Penurunan frekuensi = sinyal bahaya awal.",
        'total_trans_amt':       "💸 Nominal besar tapi frekuensi rendah = big spender yang jarang gesek → kandidat pindah bank.",
        'total_revolving_bal':   "🔗 Saldo bergulir nasabah churn sangat rendah ($672 vs $1.256). Saldo mendekati 0 = siap menutup akun.",
        'contacts_count_12_mon': "📞 Nasabah churn lebih sering hubungi CS (3x vs 2.3x). Kontak berulang = komplain tidak terselesaikan.",
        'customer_age':          "🎂 Distribusi usia relatif mirip. Usia bukan faktor dominan.",
        'months_on_book':        "📅 Lama nasabah tidak signifikan membedakan churn. Loyalitas historis ≠ loyalitas aktif.",
        'credit_limit':          "💳 Credit limit lebih tinggi pada nasabah bertahan, tapi perbedaannya tidak mencolok.",
        'avg_utilization_ratio': "📉 Utilisasi rendah pada nasabah churn — mereka sudah mulai tidak menggunakan kartu aktif."
    }
    st.markdown(f'<div class="insight-box">💡 <b>Insight:</b> {insights[selected]}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: MODELLING
# ════════════════════════════════════════════════════════
elif page == "🤖 Modelling & Evaluasi":
    st.markdown('<div class="main-header">🤖 Modelling & Evaluasi</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("❌ Model belum digenerate. Jalankan `python train_model.py` terlebih dahulu.")
        st.stop()
    if not data_loaded:
        st.error("❌ File CSV tidak ditemukan.")
        st.stop()

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = get_model_data()

    y_pred_test  = model.predict(X_test_sc)
    y_pred_train = model.predict(X_train_sc)

    st.subheader("📊 Perbandingan Performa Model")
    # Tabel perbandingan dari notebook
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'KNN', 'Decision Tree', 'SVM (RBF)', 'Random Forest', '🏆 XGBoost'],
        'Train Recall': [0.8508, 0.5077, 1.0000, 0.9631, 1.0000, 1.0000],
        'Test Recall':  [0.8410, 0.2997, 0.7920, 0.8624, 0.7095, recall_score(y_test, y_pred_test)],
        'Precision':    [0.5198, 0.7153, 0.7896, 0.6395, 0.9100, precision_score(y_test, y_pred_test)],
        'F1-Score':     [0.6425, 0.4224, 0.7908, 0.7344, 0.8000, f1_score(y_test, y_pred_test)],
    })
    comparison['Overfit Gap'] = (comparison['Train Recall'] - comparison['Test Recall']).round(4)

    def highlight_winner(row):
        color = 'background-color: #e8f5e9; font-weight: bold' if '🏆' in str(row['Model']) else ''
        return [color] * len(row)

    st.dataframe(
        comparison.style
        .apply(highlight_winner, axis=1)
        .format({'Train Recall': '{:.2%}', 'Test Recall': '{:.2%}',
                 'Precision': '{:.2%}', 'F1-Score': '{:.4f}', 'Overfit Gap': '{:.4f}'}),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🏆 XGBoost – Evaluasi Detail (dari Pickle)")
    st.caption("Model ini di-load langsung dari file pickle — bukan di-train ulang setiap sesi")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",       f"{accuracy_score(y_test, y_pred_test):.2%}")
    c2.metric("Recall (Churn)", f"{recall_score(y_test, y_pred_test):.2%}")
    c3.metric("Precision",      f"{precision_score(y_test, y_pred_test):.2%}")
    c4.metric("F1-Score",       f"{f1_score(y_test, y_pred_test):.2%}")
    gap = recall_score(y_train, y_pred_train) - recall_score(y_test, y_pred_test)
    c5.metric("Overfit Gap",    f"{gap:.2%}", delta_color="inverse")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stay', 'Churn'],
                    yticklabels=['Stay', 'Churn'],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.subheader("Classification Report")
        rpt = classification_report(y_test, y_pred_test, output_dict=True,
                                    target_names=['Stay', 'Churn'])
        rpt_df = pd.DataFrame(rpt).T.round(3)
        st.dataframe(rpt_df.style.background_gradient(cmap='Blues',
                     subset=['precision', 'recall', 'f1-score']),
                     use_container_width=True)

    # Recall bar chart semua model
    st.markdown("---")
    st.subheader("📊 Recall Comparison – Semua Model")
    fig_rc = px.bar(
        comparison, x='Model', y='Test Recall',
        color='Test Recall',
        color_continuous_scale='RdYlGn',
        text='Test Recall',
        title='Test Recall per Model (Fokus: Jangan Sampai Churner Lolos)'
    )
    fig_rc.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig_rc.add_hline(y=0.8, line_dash='dash', line_color='gray',
                     annotation_text='Threshold 80%')
    st.plotly_chart(fig_rc, use_container_width=True)


# ════════════════════════════════════════════════════════
# PAGE: PROFITABILITAS
# ════════════════════════════════════════════════════════
elif page == "💰 Profitabilitas":
    st.markdown('<div class="main-header">💰 Analisis Profitabilitas</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("❌ Jalankan `train_model.py` dulu.")
        st.stop()

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = get_model_data()
    y_pred = model.predict(X_test_sc)
    cm_vals = confusion_matrix(y_test, y_pred).ravel()
    TN, FP, FN, TP = cm_vals

    st.subheader("⚙️ Atur Asumsi Finansial")
    c1, c2, c3 = st.columns(3)
    revenue = c1.number_input("💚 Keuntungan per Nasabah Ditahan ($)", value=10, min_value=1, max_value=500)
    cost    = c2.number_input("💛 Biaya Marketing per Nasabah ($)",    value=4,  min_value=0, max_value=200)
    loss    = c3.number_input("🔴 Kerugian per Nasabah Churn ($)",     value=10, min_value=1, max_value=500)

    gross       = (TP * revenue) + (FP * revenue) - (FP * cost)
    opp_loss    = FN * loss
    net_profit  = gross - opp_loss
    no_model    = -(TP + FN) * loss

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("🚫 Tanpa Model (Rugi)",        f"${no_model:,}")
    c2.metric("✅ Dengan XGBoost (Net Profit)", f"${net_profit:,}",
              delta=f"+${net_profit - no_model:,} diselamatkan")
    c3.metric("💸 Biaya Marketing Terbuang (FP)", f"${FP * cost:,}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["🚫 Tanpa Model", "✅ XGBoost"],
        y=[no_model, net_profit],
        marker_color=["#e53935", "#43a047"],
        text=[f"${no_model:,}", f"${net_profit:,}"],
        textposition='outside', textfont=dict(size=14)
    ))
    fig.update_layout(title="Profit: Tanpa Model vs Dengan XGBoost",
                      yaxis_title="Nilai ($)", showlegend=False,
                      plot_bgcolor='rgba(0,0,0,0)')
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Breakdown")
    bd = pd.DataFrame({
        'Kategori':           ['TP – Churn berhasil dicegah', 'FP – Salah target (nasabah setia)',
                               'FN – Churn tidak terdeteksi', 'TN – Setia teridentifikasi'],
        'Jumlah':             [TP, FP, FN, TN],
        'Dampak Finansial':   [f"+${TP*revenue:,}", f"+${FP*(revenue-cost):,}",
                               f"-${FN*loss:,}", "$0"]
    })
    st.dataframe(bd, use_container_width=True)

    st.markdown(f"""
    <div class="success-box">
    🏆 <b>Kesimpulan:</b> Model XGBoost menyelamatkan potensi pendapatan sebesar 
    <b>${net_profit:,}</b> dari {len(y_test):,} data nasabah, dibandingkan kerugian 
    <b>${abs(no_model):,}</b> tanpa model.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: SHAP GLOBAL
# ════════════════════════════════════════════════════════
elif page == "🔍 SHAP – Global Interpretation":
    st.markdown('<div class="main-header">🔍 SHAP – Global Model Interpretation</div>', unsafe_allow_html=True)
    st.caption("SHAP (SHapley Additive exPlanations) menjelaskan kontribusi tiap fitur terhadap prediksi model secara global.")

    if not model_loaded:
        st.error("❌ Jalankan `train_model.py` dulu.")
        st.stop()

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = get_model_data()

    with st.spinner("⏳ Menghitung SHAP values dari model pickle..."):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

    tab1, tab2, tab3 = st.tabs(["📊 Summary Plot", "🔵 Bar Importance", "🌊 Dependence Plot"])

    with tab1:
        st.subheader("SHAP Summary Plot")
        st.caption("Merah = nilai fitur tinggi | Biru = nilai fitur rendah | Kanan = mendorong churn")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    with tab2:
        st.subheader("SHAP Feature Importance (Mean |SHAP|)")
        shap_imp = pd.DataFrame({
            'Feature':   X_test.columns,
            'SHAP Mean': np.abs(shap_values).mean(axis=0)
        }).sort_values('SHAP Mean', ascending=True).tail(15)

        fig2 = px.bar(shap_imp, x='SHAP Mean', y='Feature', orientation='h',
                      color='SHAP Mean', color_continuous_scale='Blues',
                      title='Top 15 Fitur berdasarkan SHAP (dari Model Pickle)')
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        📌 <b>Interpretasi:</b> Semakin besar nilai SHAP, semakin berpengaruh fitur 
        tersebut terhadap prediksi model — baik ke arah churn maupun bertahan.
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.subheader("SHAP Dependence Plot")
        st.caption("Tampilkan hubungan antara nilai fitur dan kontribusi SHAP-nya")
        top_features = pd.Series(np.abs(shap_values).mean(axis=0),
                                 index=X_test.columns).nlargest(10).index.tolist()
        sel_feat = st.selectbox("Pilih Fitur", top_features)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        shap.dependence_plot(sel_feat, shap_values, X_test,
                             ax=ax3, show=False, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.clf()


# ════════════════════════════════════════════════════════
# PAGE: PREDIKSI NASABAH BARU (LIME)
# ════════════════════════════════════════════════════════
elif page == "👤 Prediksi Nasabah Baru":
    st.markdown('<div class="main-header">👤 Prediksi Nasabah Baru</div>', unsafe_allow_html=True)
    st.caption("Masukkan data nasabah — model akan memprediksi risiko churn secara real-time menggunakan model pickle.")

    if not model_loaded:
        st.error("❌ Jalankan `train_model.py` dulu.")
        st.stop()
    if not data_loaded:
        st.error("❌ File CSV tidak ditemukan.")
        st.stop()

    X_train, X_test, X_train_sc, X_test_sc, y_train, y_test = get_model_data()

    st.subheader("📝 Input Data Nasabah")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demografi**")
        customer_age      = st.slider("Usia", 20, 75, 45)
        gender            = st.selectbox("Gender", ["M", "F"])
        education_level   = st.selectbox("Pendidikan", cat_info['education_level'])
        marital_status    = st.selectbox("Status Pernikahan", cat_info['marital_status'])
        income_category   = st.selectbox("Kategori Pendapatan", cat_info['income_category'])
        dependent_count   = st.slider("Jumlah Tanggungan", 0, 5, 2)

    with col2:
        st.markdown("**💳 Info Kartu & Relasi**")
        card_category             = st.selectbox("Jenis Kartu", cat_info['card_category'])
        months_on_book            = st.slider("Lama Nasabah (Bulan)", 12, 56, 36)
        total_relationship_count  = st.slider("Jumlah Produk Bank", 1, 6, 3)
        months_inactive_12_mon    = st.slider("Bulan Tidak Aktif (12 bln)", 0, 6, 2)
        contacts_count_12_mon     = st.slider("Kontak CS (12 bln)", 0, 6, 2)

    with col3:
        st.markdown("**💰 Transaksi & Keuangan**")
        credit_limit          = st.number_input("Credit Limit ($)", 1000, 35000, 5000, 500)
        total_revolving_bal   = st.number_input("Saldo Bergulir ($)", 0, 2517, 1000, 100)
        avg_open_to_buy       = st.number_input("Avg Open to Buy ($)", 0, 35000, 4000, 500)
        total_amt_chng_q4_q1  = st.number_input("Perubahan Nominal Q4/Q1", 0.0, 3.0, 0.8, 0.01)
        total_trans_amt       = st.number_input("Total Transaksi ($)", 500, 18000, 4000, 100)
        total_trans_ct        = st.slider("Jumlah Transaksi", 10, 139, 55)
        total_ct_chng_q4_q1   = st.number_input("Perubahan Count Q4/Q1", 0.0, 3.0, 0.7, 0.01)
        avg_utilization_ratio = st.slider("Rasio Utilisasi", 0.0, 1.0, 0.3, 0.01)

    if st.button("🔮 Prediksi Sekarang!", type="primary", use_container_width=True):
        # ── Build input row ──
        raw_input = {
            'customer_age':            customer_age,
            'dependent_count':         dependent_count,
            'months_on_book':          months_on_book,
            'total_relationship_count': total_relationship_count,
            'months_inactive_12_mon':  months_inactive_12_mon,
            'contacts_count_12_mon':   contacts_count_12_mon,
            'credit_limit':            credit_limit,
            'total_revolving_bal':     total_revolving_bal,
            'avg_open_to_buy':         avg_open_to_buy,
            'total_amt_chng_q4_q1':    total_amt_chng_q4_q1,
            'total_trans_amt':         total_trans_amt,
            'total_trans_ct':          total_trans_ct,
            'total_ct_chng_q4_q1':     total_ct_chng_q4_q1,
            'avg_utilization_ratio':   avg_utilization_ratio,
            'gender':                  gender,
            'education_level':         education_level,
            'marital_status':          marital_status,
            'income_category':         income_category,
            'card_category':           card_category,
        }

        # One-hot encode manual (sesuai feature_names dari pickle)
        input_df = pd.DataFrame([raw_input])
        cat_cols_list = ['gender', 'education_level', 'marital_status', 'income_category', 'card_category']
        input_encoded = pd.get_dummies(input_df, columns=cat_cols_list, drop_first=True)

        # Align kolom agar persis sama dengan training
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_names]

        # Scale menggunakan scaler dari pickle
        input_scaled = scaler.transform(input_encoded)

        # Prediksi dari model pickle
        proba      = model.predict_proba(input_scaled)[0]
        churn_prob = proba[1]
        pred_label = model.predict(input_scaled)[0]

        st.markdown("---")
        st.subheader("🎯 Hasil Prediksi")
        col_res, col_donut = st.columns([1, 1])

        with col_res:
            if churn_prob >= 0.70:
                st.markdown(f"""
                <div class="danger-box">
                <h3>🔴 RISIKO TINGGI – CHURN</h3>
                <b>Probabilitas Churn: {churn_prob:.1%}</b><br>
                Nasabah ini sangat berpotensi untuk meninggalkan bank.
                </div>""", unsafe_allow_html=True)
            elif churn_prob >= 0.40:
                st.markdown(f"""
                <div class="warning-box">
                <h3>🟡 RISIKO SEDANG</h3>
                <b>Probabilitas Churn: {churn_prob:.1%}</b><br>
                Nasabah ini perlu dipantau dan diberikan perhatian lebih.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                <h3>🟢 RISIKO RENDAH – AMAN</h3>
                <b>Probabilitas Churn: {churn_prob:.1%}</b><br>
                Nasabah ini cenderung akan tetap bertahan.
                </div>""", unsafe_allow_html=True)

            # Rekomendasi aksi
            st.markdown("**💡 Rekomendasi Aksi:**")
            if contacts_count_12_mon >= 4:
                st.markdown("- 🚨 Eskalasi ke tim Retensi — kontak CS terlalu sering")
            if total_trans_ct < 35:
                st.markdown("- 🎁 Kirim promo cashback untuk dorong frekuensi transaksi")
            if total_revolving_bal < 300:
                st.markdown("- 💳 Tawarkan cicilan 0% untuk tingkatkan saldo bergulir")
            if months_inactive_12_mon >= 3:
                st.markdown("- 📧 Kirim re-activation campaign segera")
            if total_relationship_count <= 2:
                st.markdown("- 🤝 Cross-sell produk tabungan / asuransi")
            if churn_prob < 0.4:
                st.markdown("- ✅ Tidak diperlukan aksi khusus — nasabah dalam kondisi baik")

        with col_donut:
            fig_d = go.Figure(go.Pie(
                values=[churn_prob * 100, (1 - churn_prob) * 100],
                labels=['Churn', 'Bertahan'],
                hole=0.6,
                marker_colors=['#e53935', '#43a047'],
                textinfo='label+percent'
            ))
            fig_d.update_layout(
                annotations=[dict(text=f"{churn_prob:.0%}", x=0.5, y=0.5,
                                  font_size=26, showarrow=False)],
                showlegend=True, margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig_d, use_container_width=True)

        # ── LIME Explanation ──
        st.markdown("---")
        st.subheader("🧠 LIME – Penjelasan Prediksi Individual")
        st.caption("LIME menjelaskan mengapa model memberikan prediksi ini untuk nasabah spesifik ini.")

        with st.spinner("⏳ Menghitung LIME explanation..."):
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train_sc,
                feature_names=feature_names,
                class_names=['Bertahan', 'Churn'],
                mode='classification',
                discretize_continuous=True
            )
            exp = lime_explainer.explain_instance(
                data_row=input_scaled[0],
                predict_fn=model.predict_proba,
                num_features=10
            )

        # Plot LIME sebagai bar chart
        lime_list = exp.as_list()
        lime_df   = pd.DataFrame(lime_list, columns=['Feature', 'Weight'])
        lime_df   = lime_df.sort_values('Weight', key=abs, ascending=True)
        lime_df['Color'] = lime_df['Weight'].apply(
            lambda x: '#e53935' if x > 0 else '#43a047'
        )
        lime_df['Arah'] = lime_df['Weight'].apply(
            lambda x: '🔴 Mendorong Churn' if x > 0 else '🟢 Mendorong Bertahan'
        )

        fig_lime = go.Figure(go.Bar(
            x=lime_df['Weight'],
            y=lime_df['Feature'],
            orientation='h',
            marker_color=lime_df['Color'],
            text=lime_df['Weight'].round(3),
            textposition='outside'
        ))
        fig_lime.update_layout(
            title=f"LIME – Faktor Pendorong Prediksi (Churn Prob: {churn_prob:.1%})",
            xaxis_title="Kontribusi terhadap Prediksi",
            yaxis_title="Fitur / Kondisi",
            height=450,
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_lime.add_vline(x=0, line_dash='dash', line_color='gray')
        st.plotly_chart(fig_lime, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        📌 <b>Cara membaca:</b>
        <b>Bar merah (kanan)</b> = fitur yang mendorong prediksi <b>CHURN</b>. 
        <b>Bar hijau (kiri)</b> = fitur yang mendorong prediksi <b>BERTAHAN</b>.
        Semakin panjang bar, semakin kuat pengaruh fitur tersebut.
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
# PAGE: REFLECTION
# ════════════════════════════════════════════════════════
elif page == "❓ Reflection":
    st.markdown('<div class="main-header">❓ Reflection Questions</div>', unsafe_allow_html=True)

    st.subheader("1. Mengapa penting untuk men-deploy project ke Streamlit?")
    st.markdown("""
    <div class="insight-box">
    Notebook Jupyter powerful untuk eksplorasi, namun terbatas dari sisi <b>aksesibilitas</b>.
    Stakeholder non-teknis tidak bisa — dan tidak seharusnya perlu — membuka file <code>.ipynb</code>
    untuk mendapat insight.<br><br>
    Dengan Streamlit, analisis berubah menjadi <b>aplikasi web interaktif yang bisa diakses siapa saja</b>.
    Tidak perlu install Python atau tahu machine learning.<br><br>
    <b>Manfaat utama:</b><br>
    ✅ Aksesibilitas — dapat diakses siapa saja via browser<br>
    ✅ Interaktivitas — stakeholder eksplorasi data sendiri<br>
    ✅ Komunikasi — visual mempercepat pengambilan keputusan<br>
    ✅ Portofolio — membuktikan kemampuan end-to-end Data Science
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("2. Mengapa interaktivitas adalah inti dari dashboard/ML app?")
    st.markdown("""
    <div class="insight-box">
    Komponen interaktif bukan "fitur tambahan" — mereka adalah <b>inti dari value sebuah dashboard</b>.<br><br>
    Tanpa interaktivitas, stakeholder harus meminta data scientist membuat ulang visualisasi setiap kali 
    ingin melihat sudut pandang berbeda. Dengan interaktivitas:<br><br>
    🔍 <b>Eksplorasi mandiri</b> — pilih variabel sendiri, lihat hasilnya langsung<br>
    🧪 <b>Eksperimen model</b> — ubah parameter, bandingkan hasilnya<br>
    💡 <b>What-if analysis</b> — simulasi asumsi profit berbeda<br>
    🎯 <b>Prediksi real-time</b> — input data nasabah baru, dapat prediksi & LIME seketika<br><br>
    Halaman <b>Prediksi Nasabah Baru</b> di dashboard ini adalah contoh nyata: CS bank bisa langsung 
    input data nasabah dan mendapat prediksi churn + rekomendasi aksi — bukan asal promo sembarangan.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="success-box">
    📌 <b>Dibuat oleh:</b> Reinaldi Santoso<br>
    🎓 <b>Program:</b> Data Science Bootcamp – Dibimbing.id<br>
    📊 <b>Dataset:</b> Bank Customer Churn (10.127 nasabah)<br>
    🏆 <b>Champion Model:</b> XGBoost | Recall 91%+ | Net Profit $2,992 / 2.026 nasabah
    </div>""", unsafe_allow_html=True)
