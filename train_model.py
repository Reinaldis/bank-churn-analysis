"""
train_model.py
==============
Jalankan script ini SEKALI untuk menghasilkan file model pickle.
Perintah: python train_model.py

Output yang dihasilkan:
  - model/xgb_model.pkl       → Model XGBoost
  - model/scaler.pkl          → StandardScaler
  - model/feature_names.pkl   → Nama kolom fitur
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from xgboost import XGBClassifier

# ── Buat folder model ──
os.makedirs("model", exist_ok=True)

print("=" * 50)
print("  BANK CHURN – MODEL TRAINING & SAVING")
print("=" * 50)

# ── 1. Load Data ──
print("\n[1/5] Loading data...")
df = pd.read_csv("bank_churn_data.csv")
df['attrition_flag'] = df['attrition_flag'].map(
    {'Existing Customer': 0, 'Attrited Customer': 1}
)
df = df.set_index('user_id')
print(f"      Data shape: {df.shape}")

# ── 2. Feature Engineering ──
print("[2/5] Feature engineering...")
cat_cols = ['gender', 'education_level', 'marital_status',
            'income_category', 'card_category']
df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_model.drop('attrition_flag', axis=1)
y = df_model['attrition_flag']

# Simpan nama fitur agar konsisten saat prediksi
feature_names = X.columns.tolist()

# ── 3. Split & Scale ──
print("[3/5] Splitting & scaling data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"      Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── 4. Train XGBoost ──
print("[4/5] Training XGBoost model...")
ratio = (len(y_train) - y_train.sum()) / y_train.sum()
model = XGBClassifier(
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric='logloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train_sc, y_train,
          eval_set=[(X_test_sc, y_test)],
          verbose=False)

y_pred = model.predict(X_test_sc)

print("\n      === EVALUATION RESULTS ===")
print(f"      Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"      Precision : {precision_score(y_test, y_pred):.4f}")
print(f"      F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))

# ── 5. Save Pickle ──
print("[5/5] Saving model artifacts to /model folder...")

with open("model/xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

# Simpan juga info kolom kategori untuk encoding konsisten
cat_info = {col: sorted(df[col].unique().tolist()) for col in cat_cols}
with open("model/cat_info.pkl", "wb") as f:
    pickle.dump(cat_info, f)

print("\n✅ Semua file berhasil disimpan:")
for fname in os.listdir("model"):
    size = os.path.getsize(f"model/{fname}")
    print(f"   model/{fname}  ({size/1024:.1f} KB)")

print("\n🚀 Sekarang jalankan: python -m streamlit run app.py")
print("=" * 50)
