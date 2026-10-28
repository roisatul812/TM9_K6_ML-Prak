from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


def find_file(filename):
    """Cari file data di beberapa lokasi umum, jadi tidak error walau struktur folder berbeda."""
    kandidat = [
        BASE_DIR.parent / "dataset",
        BASE_DIR / "dataset",
        BASE_DIR,
        BASE_DIR.parent,
    ]
    for folder in kandidat:
        path = folder / filename
        if path.exists():
            return path
    lokasi = "\n".join(f" - {folder / filename}" for folder in kandidat)
    raise FileNotFoundError(
        f"File '{filename}' tidak ditemukan. Sudah dicari di:\n{lokasi}\n"
        "Letakkan file CSV di salah satu lokasi tersebut."
    )


DATA_PATH = find_file("heart.csv")
# Folder output ada sejajar dengan folder dataset (satu tingkat di atas code).
if DATA_PATH.parent.name == "dataset":
    ROOT_DIR = DATA_PATH.parent.parent
else:
    ROOT_DIR = BASE_DIR
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "HeartDisease"

print("=== DATASET INFO ===")
print("File data     :", DATA_PATH)
print("Nama dataset :", "Heart Disease Dataset")
print("Jumlah baris :", df.shape[0])
print("Jumlah kolom :", df.shape[1])
print("\n=== 5 DATA PERTAMA ===")
print(df.head())
print("\n=== TIPE DATA ===")
print(df.dtypes)
print("\n=== STATISTIK DESKRIPTIF (KOLOM NUMERIK) ===")
print(df.describe(include="number").transpose())
print("\n=== STATISTIK DESKRIPTIF (KOLOM KATEGORIKAL) ===")
print(df.select_dtypes(exclude="number").describe().transpose())
print("\n=== MISSING VALUE ===")
print(df.isnull().sum())
print("\n=== DUPLICATE ===")
print("Jumlah duplicate:", int(df.duplicated().sum()))

# Hapus baris duplikat persis jika ada.
df = df.drop_duplicates().copy()

# Fitur numerik dan kategorikal ditentukan manual.
# FastingBS berisi 0/1 sehingga diperlakukan sebagai kategorikal.
numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
categorical_features = ["Sex", "ChestPainType", "FastingBS",
                        "RestingECG", "ExerciseAngina", "ST_Slope"]
feature_columns = numeric_features + categorical_features

# Missing value tersamar: nilai 0 yang tidak valid secara medis.
print("\n=== NILAI 0 TIDAK VALID ===")
for col in ["Cholesterol", "RestingBP"]:
    print(f"{col} = 0 : {int((df[col] == 0).sum())}")

for col in ["Cholesterol", "RestingBP"]:
    median_valid = df.loc[df[col] > 0, col].median()
    df[col] = df[col].replace(0, median_valid)
    print(f"{col}: nilai 0 diganti median valid = {median_valid}")

# Deteksi outlier dengan IQR (dilaporkan, tidak dicapping/dihapus).
outlier_rows = []
for col in numeric_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_rows.append({
        "Feature": col, "Q1": q1, "Q3": q3, "IQR": iqr,
        "Lower Bound": lower, "Upper Bound": upper, "Outlier Count": count
    })
outlier_df = pd.DataFrame(outlier_rows)
print("\n=== OUTLIER (IQR) ===")
print(outlier_df[["Feature", "Lower Bound", "Upper Bound", "Outlier Count"]])
outlier_df.to_csv(OUTPUT_DIR / "02_heart_outlier_results.csv", index=False)

# Kategorikal tetap kategorikal (string), tanpa one-hot dan tanpa scaling.
# Scaling fitur numerik dilakukan pada tahap K-Prototypes.
prepared = df[feature_columns].copy()
for col in categorical_features:
    prepared[col] = prepared[col].astype(str)
# HeartDisease disimpan hanya untuk interpretasi, TIDAK dipakai clustering.
prepared[TARGET] = df[TARGET].values
prepared.to_csv(OUTPUT_DIR / "03_heart_dataset_preprocessed.csv", index=False)

with open(OUTPUT_DIR / "01_heart_dataset_overview.txt", "w", encoding="utf-8") as f:
    f.write("RINGKASAN DATASET HEART DISEASE\n")
    f.write("=" * 40 + "\n")
    f.write(f"Ukuran setelah penghapusan duplikat: {df.shape[0]} baris x {df.shape[1]} kolom\n")
    f.write(f"Total missing value: {int(df.isna().sum().sum())}\n")
    f.write(f"Fitur numerik: {numeric_features}\n")
    f.write(f"Fitur kategorikal: {categorical_features}\n")
    f.write(f"Kolom {TARGET} tidak digunakan untuk clustering (hanya interpretasi).\n")
    f.write("Dataset disimpan tanpa scaling dan tanpa one-hot encoding agar dapat digunakan K-Prototypes.\n")

print("\nFitur numerik:", numeric_features)
print("Fitur kategorikal:", categorical_features)
print("\nFile tersimpan:")
print(OUTPUT_DIR / "02_heart_outlier_results.csv")
print(OUTPUT_DIR / "01_heart_dataset_overview.txt")
print(OUTPUT_DIR / "03_heart_dataset_preprocessed.csv")
print("\n=== PREPROCESSING HEART DISEASE SELESAI ===")