
from pathlib import Path
import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. RUTAS
# ==========================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "01_datos" / "procesados"
RESULTADOS = BASE / "06_resultados"

MODELOS = RESULTADOS / "modelos_entrenados"
METRICAS = RESULTADOS / "metricas"
GRAFICOS = RESULTADOS / "graficos"

SALIDA = METRICAS / "evaluacion_final"
SALIDA_GRAFICOS = GRAFICOS / "evaluacion_final"

SALIDA.mkdir(parents=True, exist_ok=True)
SALIDA_GRAFICOS.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. SELECCION CONGELADA DE MODELOS
# ==========================================

# MLP: configuracion ganadora en validacion
with open(
    METRICAS / "experimentos_mlp" / "mejor_mlp.json",
    encoding="utf-8"
) as f:
    mejor_mlp = json.load(f)

# CNN: elegir la mejor por Average Precision
tabla_cnn = pd.read_csv(
    METRICAS / "experimentos_cnn" / "comparacion_cnn.csv"
)

mejor_cnn = (
    tabla_cnn.sort_values(
        "average_precision", ascending=False
    ).iloc[0].to_dict()
)

# LSTM: configuracion ganadora en validacion
with open(
    METRICAS / "experimentos_lstm" / "mejor_lstm.json",
    encoding="utf-8"
) as f:
    mejor_lstm = json.load(f)

seleccion = [
    ("MLP", mejor_mlp, "experimentos_mlp"),
    ("CNN_1D", mejor_cnn, "experimentos_cnn"),
    ("LSTM", mejor_lstm, "experimentos_lstm")
]

print("\n=== MODELOS SELECCIONADOS ===")

for familia, config, carpeta in seleccion:
    print(
        familia,
        config["modelo"],
        "Umbral:",
        round(float(config["umbral_ajustado"]), 5)
    )

# ==========================================
# 3. CARGAR TEST UNA SOLA VEZ
# ==========================================

X_test = np.load(DATOS / "X_test.npy")
y_test = np.load(DATOS / "y_test.npy")

assert X_test.shape == (7335, 60, 15)
assert y_test.shape == (7335,)

print("\n=== CONJUNTO TEST ===")
print("Ventanas:", len(y_test))
print("Normales:", int((y_test == 0).sum()))
print("Fallas:", int((y_test == 1).sum()))

# ==========================================
# 4. EVALUAR LOS TRES MODELOS
# ==========================================

resultados = []

for familia, config, carpeta in seleccion:

    nombre = config["modelo"]
    umbral = float(config["umbral_ajustado"])

    ruta_modelo = (
        MODELOS / carpeta / f"{nombre}.keras"
    )

    if not ruta_modelo.exists():
        raise FileNotFoundError(ruta_modelo)

    print(f"\n=== EVALUANDO {familia}: {nombre} ===")

    tf.keras.backend.clear_session()

    modelo = tf.keras.models.load_model(
        ruta_modelo,
        compile=False
    )

    # MLP recibe 900 caracteristicas.
    # CNN y LSTM reciben secuencias 60 x 15.
    if familia == "MLP":
        entradas = X_test.reshape(-1, 900)
    else:
        entradas = X_test

    probabilidades = modelo.predict(
        entradas,
        batch_size=128,
        verbose=0
    ).ravel()

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    ).ravel()

    resultado = {
        "familia": familia,
        "modelo": nombre,
        "umbral_validacion": umbral,
        "parametros": int(modelo.count_params()),
        "average_precision_test": float(
            average_precision_score(
                y_test, probabilidades
            )
        ),
        "precision_test": float(
            precision_score(
                y_test, predicciones,
                zero_division=0
            )
        ),
        "recall_test": float(
            recall_score(
                y_test, predicciones,
                zero_division=0
            )
        ),
        "f1_test": float(
            f1_score(
                y_test, predicciones,
                zero_division=0
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_test, predicciones
            )
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    resultados.append(resultado)

    pd.DataFrame({
        "real": y_test,
        "puntuacion": probabilidades,
        "prediccion": predicciones
    }).to_csv(
        SALIDA / f"predicciones_{familia}.csv",
        index=False
    )

    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        matriz,
        display_labels=["Normal", "Falla"]
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(f"Matriz de confusion - {nombre}")
    plt.tight_layout()

    plt.savefig(
        SALIDA_GRAFICOS / f"matriz_{familia}.png",
        dpi=180
    )

    plt.close(fig)

    print("Average Precision:", round(
        resultado["average_precision_test"], 4
    ))
    print("Precision:", round(
        resultado["precision_test"], 4
    ))
    print("Recall:", round(
        resultado["recall_test"], 4
    ))
    print("F1:", round(
        resultado["f1_test"], 4
    ))
    print("Matriz:", matriz.tolist())

# ==========================================
# 5. COMPARACION FINAL
# ==========================================

tabla = pd.DataFrame(resultados)

tabla.to_csv(
    SALIDA / "comparacion_final_test.csv",
    index=False
)

print("\n=== COMPARACION FINAL TEST ===")

columnas = [
    "familia",
    "modelo",
    "parametros",
    "average_precision_test",
    "precision_test",
    "recall_test",
    "f1_test",
    "FP",
    "FN"
]

print(tabla[columnas].to_string(index=False))

# Guardar un registro de los umbrales utilizados
with open(
    SALIDA / "seleccion_congelada.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        [
            {
                "familia": familia,
                "modelo": config["modelo"],
                "umbral": float(
                    config["umbral_ajustado"]
                )
            }
            for familia, config, _ in seleccion
        ],
        f,
        indent=4
    )

print("\nEvaluacion final completada.")
print("No se modificaron los modelos ni umbrales.")
