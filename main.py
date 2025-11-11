# ============================================================
# 1. IMPORT LIBRARY
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Konversi jumlah investasi menjadi angka
df["Jumlah_Investasi_num"] = pd.to_numeric(
    df["Jumlah_Investasi"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
    errors="coerce"
)

# Kolom numerik lain
for col in ["luas_tanah", "TKI"]:
    df[col + "_num"] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 4. PREPARE FITUR DAN TARGET
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

# Buang target yang kosong
df_model = df.dropna(subset=[target_col]).copy()

# Hapus kategori target yang hanya muncul sekali (Naïve Bayes butuh >1)
valid_classes = df_model[target_col].value_counts()
valid_classes = valid_classes[valid_classes > 1].index.tolist()
df_model = df_model[df_model[target_col].isin(valid_classes)]

# Buang baris tanpa fitur valid
df_model = df_model[df_model[feature_cols].notnull().any(axis=1)]

X = df_model[feature_cols].copy()
y = df_model[target_col].copy()


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_features = [c for c in feature_cols if c.endswith("_num")]
categorical_features = [c for c in feature_cols if c not in numeric_features]

# pastikan kolom kategori selalu string
for col in categorical_features:
    X[col] = X[col].astype(str).fillna("missing").str.strip()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 6. SPLIT DATA
# ============================================================

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
except:
    # fallback jika stratify gagal
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )


# ============================================================
# 7. MODEL NAÏVE BAYES
# ============================================================

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("nb", GaussianNB())
])

model.fit(X_train, y_train)


# ============================================================
# 8. PREDIKSI & EVALUASI
# ============================================================

y_pred = model.predict(X_test)

print("\n=== AKURASI MODEL ===")
print("Akurasi:", accuracy_score(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================================
# 9. VISUALISASI DATASET
# ============================================================

# Grafik 1 — Jumlah Proyek per Jenis
plt.figure(figsize=(7,4))
df["Uraian_Jenis_Proyek"].value_counts().plot(kind="bar")
plt.title("Jumlah Proyek per Jenis")
plt.xlabel("Jenis Proyek")
plt.ylabel("Jumlah")
plt.show()

# Grafik 2 — Investasi per Kecamatan (10 Terbesar)
top_kec = df.groupby("kecamatan_usaha")["Jumlah_Investasi_num"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,4))
top_kec.plot(kind="bar")
plt.title("Top 10 Kecamatan Berdasarkan Total Investasi")
plt.ylabel("Total Investasi")
plt.show()

# Grafik 3 — Histogram Log Investasi
plt.figure(figsize=(7,4))
df["Jumlah_Investasi_num"].dropna().apply(np.log1p).hist(bins=30)
plt.title("Distribusi Log(Jumlah Investasi)")
plt.xlabel("log(Investasi)")
plt.ylabel("Frekuensi")
plt.show()


# ============================================================
# 10. RINGKASAN DATA
# ============================================================

print("\n=== RINGKASAN DATASET ===")
print("Total Proyek:", len(df))
print("Total Investasi:", df["Jumlah_Investasi_num"].sum())
print("\nTop Kecamatan Investasi:\n", top_kec)
