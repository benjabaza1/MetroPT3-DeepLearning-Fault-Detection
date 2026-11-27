
from pathlib import Path
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ==========================================
# 1. CONFIGURACION
# ==========================================

BASE = Path(__file__).resolve().parent.parent

DATOS = BASE / "01_datos" / "procesados"
RESULTADOS = BASE / "06_resultados"

MODELOS = RESULTADOS / "modelos_entrenados"

METRICAS = (
    RESULTADOS / "metricas" / "explicabilidad"
)

GRAFICOS = (
    RESULTADOS / "graficos" / "explicabilidad"
)

METRICAS.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

SENSORES = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Motor_current", "COMP", "DV_eletric",
    "Towers", "MPG", "LPS",
    "Pressure_switch", "Oil_level",
    "Caudal_impulses"
]

MODELOS_SELECCIONADOS = {
    "MLP": (
        MODELOS / "experimentos_mlp"
        / "MLP_04_Tanh.keras"
    ),
    "CNN_1D": (
        MODELOS / "experimentos_cnn"
        / "CNN_05_Tanh.keras"
    ),
    "LSTM": (
        MODELOS / "experimentos_lstm"
        / "LSTM_01_Base.keras"
    )
}

REPETICIONES = 3
SEMILLA = 42

# ==========================================
# 2. CARGAR VALIDACION
# ==========================================

print("\n=== EXPLICABILIDAD METROPT-3 ===")

X_val = np.load(
    DATOS / "X_validation.npy"
)

y_val = np.load(
    DATOS / "y_validation.npy"
)

assert X_val.shape == (3581, 60, 15)
assert len(y_val) == len(X_val)

print("Ventanas:", len(X_val))
print("Sensores:", len(SENSORES))

# ==========================================
# 3. PREPARAR ENTRADAS
# ==========================================

def preparar_entrada(X, familia):

    if familia == "MLP":
        return X.reshape(len(X), 900)

    return X

def predecir(modelo, X, familia):

    entrada = preparar_entrada(X, familia)

    return modelo.predict(
        entrada,
        batch_size=128,
        verbose=0
    ).ravel()

# ==========================================
# 4. PERMUTACION DE SENSORES
# ==========================================

resultados = []

for familia, ruta in MODELOS_SELECCIONADOS.items():

    print(f"\n=== MODELO {familia} ===")

    tf.keras.backend.clear_session()

    modelo = tf.keras.models.load_model(
        ruta,
        compile=False
    )

    prob_original = predecir(
        modelo, X_val, familia
    )

    ap_original = average_precision_score(
        y_val, prob_original
    )

    print("AP original:", round(ap_original, 6))

    rng = np.random.default_rng(SEMILLA)

    for indice, sensor in enumerate(SENSORES):

        importancias = []

        for repeticion in range(REPETICIONES):

            X_modificado = X_val.copy()

            # Intercambiamos un sensor entre ventanas.
            # Conservamos sus 60 instantes internos.
            permutacion = rng.permutation(len(X_val))

            X_modificado[:, :, indice] = (
                X_val[permutacion, :, indice]
            )

            prob = predecir(
                modelo,
                X_modificado,
                familia
            )

            ap_modificado = average_precision_score(
                y_val, prob
            )

            importancia = (
                ap_original - ap_modificado
            )

            importancias.append(importancia)

        resultado = {
            "modelo": familia,
            "sensor": sensor,
            "ap_original": float(ap_original),
            "importancia_media": float(
                np.mean(importancias)
            ),
            "desviacion": float(
                np.std(importancias)
            )
        }

        resultados.append(resultado)

        print(
            f"{sensor:20s} "
            f"Importancia: "
            f"{resultado['importancia_media']:.6f}"
        )

    del modelo

# ==========================================
# 5. GUARDAR RESULTADOS
# ==========================================

tabla = pd.DataFrame(resultados)

tabla.to_csv(
    METRICAS / "importancia_sensores.csv",
    index=False
)

# ==========================================
# 6. GRAFICOS POR MODELO
# ==========================================

for familia in MODELOS_SELECCIONADOS:

    subset = tabla[
        tabla["modelo"] == familia
    ].sort_values("importancia_media")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        subset["sensor"],
        subset["importancia_media"]
    )

    ax.axvline(0, linewidth=1)

    ax.set_title(
        f"Importancia por permutacion - {familia}"
    )

    ax.set_xlabel(
        "Disminucion de Average Precision"
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        GRAFICOS / f"importancia_{familia}.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

# ==========================================
# 7. COMPARACION FINAL
# ==========================================

print("\n=== TOP 5 POR ARQUITECTURA ===")

for familia in MODELOS_SELECCIONADOS:

    subset = (
        tabla[tabla["modelo"] == familia]
        .sort_values(
            "importancia_media",
            ascending=False
        )
        .head(5)
    )

    print(f"\n{familia}")
    print(
        subset[
            ["sensor", "importancia_media"]
        ].to_string(index=False)
    )

print("\nExplicabilidad finalizada.")
print("No se utilizaron datos de TEST.")
