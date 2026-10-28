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


DATA_PATH = find_file("Mall_Customers.csv")
# Folder output ada sejajar dengan folder dataset (satu tingkat di atas code).
if DATA_PATH.parent.name == "dataset":
    ROOT_DIR = DATA_PATH.parent.parent
else:
    ROOT_DIR = BASE_DIR
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=== DATASET INFO ===")
print("File data     :", DATA_PATH)
print("Nama dataset :", "Mall Customers")
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

# CustomerID hanya identifier, bukan karakteristik pelanggan untuk clustering.
identifier_columns = ["CustomerID"]
numeric_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
categorical_features = ["Gender"]
feature_columns = numeric_features + categorical_features

# Isi missing value hanya jika memang ada.
for col in numeric_features:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
for col in categorical_features:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

# Deteksi outlier dengan IQR. Untuk K-Prototypes, outlier dilaporkan dahulu,
# tidak otomatis dicapping/dihapus agar struktur data asli tidak berubah tanpa alasan.
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
outlier_df.to_csv(OUTPUT_DIR / "02_mall_outlier_results.csv", index=False)

# Simpan fitur campuran apa adanya (tanpa scaling dan tanpa encoding).
# Scaling fitur numerik dilakukan pada tahap K-Prototypes.
prepared = df[feature_columns].copy()
prepared["Gender"] = prepared["Gender"].astype(str)
prepared.to_csv(OUTPUT_DIR / "03_mall_dataset_preprocessed.csv", index=False)

# Simpan ringkasan data dan definisi fitur.
with open(OUTPUT_DIR / "01_mall_dataset_overview.txt", "w", encoding="utf-8") as f:
    f.write("RINGKASAN DATASET MALL CUSTOMERS\n")
    f.write("=" * 40 + "\n")
    f.write(f"Ukuran setelah penghapusan duplikat: {df.shape[0]} baris x {df.shape[1]} kolom\n")
    f.write(f"Total missing value setelah preprocessing: {int(df.isna().sum().sum())}\n")
    f.write(f"Kolom identifier yang tidak digunakan: {identifier_columns}\n")
    f.write(f"Fitur numerik: {numeric_features}\n")
    f.write(f"Fitur kategorikal: {categorical_features}\n")
    f.write("Dataset disimpan tanpa scaling dan tanpa encoding kategorikal agar dapat digunakan K-Prototypes.\n")

print("\nFitur numerik:", numeric_features)
print("Fitur kategorikal:", categorical_features)
print("Identifier dikeluarkan:", identifier_columns)
print("\nFile tersimpan:")
print(OUTPUT_DIR / "02_mall_outlier_results.csv")
print(OUTPUT_DIR / "01_mall_dataset_overview.txt")
print(OUTPUT_DIR / "03_mall_dataset_preprocessed.csv")
print("\n=== PREPROCESSING MALL CUSTOMERS SELESAI ===")