"""
Implementasi Algoritma K-Prototypes — Heart Disease Dataset
Tugas Pertemuan 9 (Clustering) - Person 2 (ISA)
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

# Konfigurasi path direktori
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent if BASE_DIR.name == "code" else BASE_DIR
OUTPUT_DIR = ROOT_DIR / "output" / "heart"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT_DIR / "output" / "03_heart_dataset_preprocessed.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT_DIR / "03_heart_dataset_preprocessed.csv"


def run():
    print("=== K-PROTOTYPES: HEART DISEASE DATASET ===")

    # 1. Membaca dataset hasil preprocessing Person 1
    df = pd.read_csv(DATA_PATH)
    target_column = "HeartDisease"

    # Memisahkan fitur numerik dan kategorikal
    numeric_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    categorical_features = [
        "Sex", "ChestPainType", "FastingBS",
        "RestingECG", "ExerciseAngina", "ST_Slope"
    ]

    for col in categorical_features:
        df[col] = df[col].astype(str)

    print(f"Data diproses      : {df.shape[0]} baris x {len(numeric_features) + len(categorical_features)} fitur")
    print(f"Fitur numerik      : {numeric_features}")
    print(f"Fitur kategorikal  : {categorical_features}")

    # Standardisasi fitur numerik agar skala nilai seimbang
    scaler = StandardScaler()
    scaled_num = pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features)
    X = pd.concat([scaled_num, df[categorical_features]], axis=1)

    # Menentukan indeks fitur kategorikal untuk model
    cat_indices = [X.columns.get_loc(col) for col in categorical_features]

    # 2. Penentuan jumlah cluster K dengan metode elbow (K=2..6)
    k_range = list(range(2, 7))
    elbow_records = []

    for k in k_range:
        kp = KPrototypes(n_clusters=k, init="Huang", n_init=10, max_iter=300, random_state=42, n_jobs=1)
        kp.fit(X, categorical=cat_indices)
        elbow_records.append({"K": k, "Cost": round(float(kp.cost_), 2), "Iterasi": int(kp.n_iter_)})

    df_elbow = pd.DataFrame(elbow_records)
    df_elbow["Delta_Cost"] = [0.0] + [round(df_elbow.loc[i - 1, "Cost"] - df_elbow.loc[i, "Cost"], 2) for i in range(1, len(df_elbow))]
    df_elbow["Penurunan(%)"] = [0.0] + [round((df_elbow.loc[i, "Delta_Cost"] / df_elbow.loc[i - 1, "Cost"]) * 100, 2) for i in range(1, len(df_elbow))]

    print("\nHasil Pemilihan K (Elbow):")
    print(df_elbow.to_string(index=False))

    # Menyimpan grafik elbow
    plt.figure(figsize=(8, 5))
    plt.plot(df_elbow["K"], df_elbow["Cost"], marker="o", color="#1f77b4", linewidth=2)
    plt.scatter([3], [df_elbow.loc[df_elbow["K"] == 3, "Cost"].values[0]], color="red", s=150, zorder=5)
    plt.title("Metode Elbow K-Prototypes — Heart Disease")
    plt.xlabel("Jumlah Cluster (K)")
    plt.ylabel("Cost")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "heart_elbow_curve.png", dpi=200)
    plt.close()

    # 3. Melatih model final K-Prototypes dengan K=3
    k_optimal = 3
    model = KPrototypes(n_clusters=k_optimal, init="Huang", n_init=10, max_iter=300, random_state=42, n_jobs=1)
    labels = model.fit_predict(X, categorical=cat_indices)

    print(f"\nModel Final K={k_optimal}:")
    print(f"Cost akhir         : {float(model.cost_):,.2f}")
    print(f"Iterasi konvergen  : {int(model.n_iter_)}")

    # Rekonstruksi prototipe dalam skala asli
    centroids_num = scaler.inverse_transform(model.cluster_centroids_[:, :len(numeric_features)].astype(float))
    centroids_cat = model.cluster_centroids_[:, len(numeric_features):]

    df_proto = pd.DataFrame(centroids_num, columns=numeric_features)
    for idx, col in enumerate(categorical_features):
        df_proto[col] = centroids_cat[:, idx]
    df_proto.index = [f"Cluster_{i}" for i in range(k_optimal)]
    for col in numeric_features:
        df_proto[col] = df_proto[col].round(2)

    # 4. Menghitung jumlah anggota setiap cluster
    counts = pd.Series(labels).value_counts().sort_index()
    df_counts = pd.DataFrame({
        "Cluster": [f"Cluster_{i}" for i in range(k_optimal)],
        "Jumlah_Anggota": counts.values,
        "Persentase(%)": (counts.values / len(labels) * 100).round(2)
    })

    print("\nJumlah Anggota Tiap Cluster:")
    print(df_counts.to_string(index=False))

    # 5. Menyimpan hasil clustering
    df_result = df.copy()
    df_result["Cluster_KPrototypes"] = labels
    df_result["Cluster_Label"] = [f"Cluster_{i}" for i in labels]

    df_result.to_csv(OUTPUT_DIR / "heart_clustered.csv", index=False)
    df_counts.to_csv(OUTPUT_DIR / "heart_cluster_counts.csv", index=False)
    df_elbow.to_csv(OUTPUT_DIR / "heart_k_selection.csv", index=False)
    df_proto.to_csv(OUTPUT_DIR / "heart_prototypes.csv")

    summary_text = (
        "RINGKASAN MODEL K-PROTOTYPES — HEART DISEASE\n"
        "===========================================\n"
        f"Jumlah observasi : {len(df)}\n"
        f"Fitur numerik    : {numeric_features}\n"
        f"Fitur kategorikal: {categorical_features}\n"
        f"K terpilih       : {k_optimal} (Elbow point)\n"
        f"Cost akhir       : {float(model.cost_):,.2f}\n"
        f"Iterasi          : {int(model.n_iter_)}\n\n"
        "Distribusi Anggota:\n"
        f"{df_counts.to_string(index=False)}\n\n"
        "Prototipe Cluster (Skala Asli):\n"
        f"{df_proto.to_string()}\n"
    )
    with open(OUTPUT_DIR / "heart_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\nFile output tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
