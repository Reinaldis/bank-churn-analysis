# 🏦 Bank Customer Churn Analysis – Streamlit App

> **Reinaldi Santoso** | Data Science Bootcamp – Dibimbing.id

---

## 📌 Deskripsi

Dashboard interaktif untuk analisis, prediksi, dan interpretasi **churn nasabah bank** menggunakan Machine Learning. Model disimpan sebagai **file pickle** sehingga prediksi berjalan real-time tanpa training ulang setiap sesi.

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/USERNAME/bank-churn-streamlit.git
cd bank-churn-streamlit

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate model pickle (WAJIB dijalankan sekali)
python train_model.py

# 4. Jalankan Streamlit app
python -m streamlit run app.py
```

> ⚠️ Pastikan `bank_churn_data.csv` ada di folder yang sama dengan `app.py`

---

## 🗂️ Struktur Project

```
bank-churn-streamlit/
│
├── app.py                  ← Aplikasi Streamlit utama
├── train_model.py          ← Script generate pickle (jalankan sekali)
├── bank_churn_data.csv     ← Dataset (tidak di-upload ke repo)
├── requirements.txt        ← Dependencies
├── README.md               ← Dokumentasi ini
│
└── model/                  ← Folder otomatis dibuat oleh train_model.py
    ├── xgb_model.pkl       ← Model XGBoost
    ├── scaler.pkl          ← StandardScaler
    ├── feature_names.pkl   ← Nama fitur (untuk konsistensi encoding)
    └── cat_info.pkl        ← Info kategori (untuk UI input)
```

---

## 🎯 Fitur Halaman

| Halaman | Deskripsi |
|---|---|
| 🏠 Home | KPI metrics, pie chart, preview dataset |
| 📊 EDA – Kategorikal | Bar chart interaktif per variabel kategori |
| 📈 EDA – Numerikal | Boxplot / Histogram / Violin interaktif |
| 🤖 Modelling & Evaluasi | Tabel perbandingan 6 model, confusion matrix dari **pickle** |
| 💰 Profitabilitas | Simulasi profit dengan asumsi finansial yang bisa diubah |
| 🔍 SHAP – Global | Summary plot, bar importance, dependence plot dari **model pickle** |
| 👤 Prediksi Nasabah Baru | Input data nasabah → prediksi real-time + **LIME explanation** |
| ❓ Reflection | Jawaban pertanyaan refleksi |

---

## 📊 Hasil Model

| Model | Recall | Precision | F1-Score |
|---|---|---|---|
| Logistic Regression | 84.1% | 51.9% | 0.64 |
| KNN | 29.9% | 71.5% | 0.42 |
| Decision Tree | 79.2% | 78.9% | 0.79 |
| SVM (RBF) | 86.2% | 63.9% | 0.73 |
| Random Forest | 71.0% | 91.0% | 0.80 |
| **XGBoost ✅** | **91%+** | **86%+** | **0.89** |

---

## 🔍 Key Insights

- `total_trans_ct` (frekuensi transaksi) adalah faktor paling dominan (SHAP tertinggi)
- Nasabah yang sering kontak CS (>3x/tahun) adalah **red flag** utama
- Saldo bergulir mendekati 0 = sinyal nasabah siap menutup akun
- Faktor demografis (usia, gender) **tidak signifikan** → fokus ke perilaku

---

## 🧰 Tech Stack

`Python` · `Streamlit` · `XGBoost` · `SHAP` · `LIME` · `Plotly` · `Scikit-learn` · `Pandas`

---

*Data Science Bootcamp – Dibimbing.id*
