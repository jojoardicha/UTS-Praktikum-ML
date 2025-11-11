# ============================================================
# 1. IMPORT LIBRARY
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
sns.set()

# ============================================================
# 2. LOAD DATASET
# ============================================================
df = pd.read_excel("DP.Proyek1.xlsx")

# Normalisasi nama kolom
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# ============================================================
# 3. CLEANING & KONVERSI NUMERIK
# ============================================================
df["Jumlah_Investasi_num"] = pd.to_numeric(
    df["Jumlah_Investasi"].astype(str).str.replace(r'[^0-9.]', '', regex=True),
    errors="coerce"
)

# Konversi numeric lain
for col in ["luas_tanah", "TKI"]:
    df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")

# ============================================================
# 4. PILIH FITUR & TARGET
# ============================================================
target_col = "Uraian_Jenis_Proyek"

feature_cols = [
    "Jumlah_Investasi_num",
    "luas_tanah_num",
    "TKI_num",
    "kecamatan_usaha",
    "Kbli",
    "Uraian_Jenis_Perusahaan"
]

# Ambil dataset untuk modelling
df_model = df.dropna(subset=[target_col]).copy()

# Hapus kelas dengan jumlah < 2 (Naïve Bayes butuh minimal 2 sampel)
valid_class = df_model[target_col].value_counts()
valid_class = valid_class[valid_class >= 2].index.tolist()
df_model = df_model[df_model[target_col].isin(valid_class)]

# Hapus baris tanpa fitur apapun
df_model = df_model[df_model[feature_cols].notnull().any(axis=1)]

X = df_model[feature_cols].copy()
y = df_model[target_col].copy()

# ============================================================
# 5. PREPROCESSING
# ============================================================
numeric_features = [c for c in feature_cols if c.endswith("_num")]
categorical_features = [c for c in feature_cols if c not in numeric_features]

# Konversi kategori ke string
for col in categorical_features:
    X[col] = X[col].astype(str).fillna("missing").str.strip()

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
except:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

# ============================================================
# 7. PIPELINE MODEL NAIVE BAYES
# ============================================================
model = Pipeline(steps=[
    ("pre", preprocessor),
    ("clf", GaussianNB())
])

model.fit(X_train, y_train)

# ============================================================
# 8. PREDIKSI & EVALUASI
# ============================================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Model Naive Bayes :", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix Naïve Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 10. VISUALISASI DATASET
# ============================================================

# Grafik 1: Jumlah Proyek per Jenis
plt.figure(figsize=(6,4))
df["Uraian_Jenis_Proyek"].value_counts().plot(kind="bar")
plt.title("Jumlah Proyek per Jenis")
plt.xlabel("Jenis Proyek")
plt.ylabel("Jumlah")
plt.show()

# Grafik 2: Top 10 Kecamatan berdasarkan Investasi
invest_kec = df.groupby("kecamatan_usaha")["Jumlah_Investasi_num"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,4))
invest_kec.plot(kind="bar")
plt.title("Top 10 Kecamatan dengan Investasi Tertinggi")
plt.ylabel("Total Investasi")
plt.show()

# Grafik 3: Histogram (log) Investasi
plt.figure(figsize=(6,4))
df["Jumlah_Investasi_num"].dropna().apply(np.log1p).hist(bins=30)
plt.title("Distribusi (log) Jumlah Investasi")
plt.xlabel("log(Investasi)")
plt.show()

# ============================================================
# 11. CETAK RINGKASAN DATA
# ============================================================
print("\nTotal Proyek :", len(df))
print("Total Investasi :", df["Jumlah_Investasi_num"].sum())
print("\nTop Kecamatan per Investasi :")
print(invest_kec)
