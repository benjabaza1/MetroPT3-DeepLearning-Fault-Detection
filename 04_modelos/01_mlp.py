
from pathlib import Path
import os
import json
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)

# ==================================================
# 1. CONFIGURACION GENERAL
# ==================================================

BASE = Path(__file__).resolve().parent.parent

DATOS = BASE / "01_datos" / "procesados"
RESULTADOS = BASE / "06_resultados"

CARPETA_MODELO = RESULTADOS / "modelos_entrenados"
CARPETA_METRICAS = RESULTADOS / "metricas"
CARPETA_GRAFICOS = RESULTADOS / "graficos"

for carpeta in [
    CARPETA_MODELO,
    CARPETA_METRICAS,
    CARPETA_GRAFICOS
]:
    carpeta.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.keras.utils.set_random_seed(SEMILLA)

# ==================================================
# 2. CARGAR DATOS
# ==================================================

print("\n=== MODELO MLP - METROPT-3 ===")

X_train = np.load(DATOS / "X_train.npy")
y_train = np.load(DATOS / "y_train.npy")

X_val = np.load(DATOS / "X_validation.npy")
y_val = np.load(DATOS / "y_validation.npy")

print("\nDimensiones originales:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

# Flatten: 60 x 15 = 900 caracteristicas
X_train = X_train.reshape(len(X_train), -1)
X_val = X_val.reshape(len(X_val), -1)

print("\nDimensiones para la MLP:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

assert X_train.shape[1] == 900
assert X_val.shape[1] == 900

# ==================================================
# 3. PESOS DE CLASE
# ==================================================

clases = np.array([0, 1])

pesos = compute_class_weight(
    class_weight="balanced",
    classes=clases,
    y=y_train
)

class_weight = {
    int(clase): float(peso)
    for clase, peso in zip(clases, pesos)
}

print("\n=== PESOS DE CLASE ===")
print(class_weight)

# ==================================================
# 4. CONSTRUIR ARQUITECTURA MLP
# ==================================================

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(900,)),

    tf.keras.layers.Dense(
        64,
        activation="relu",
        name="capa_oculta_1"
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu",
        name="capa_oculta_2"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="probabilidad_falla"
    )
], name="MLP_MetroPT3")

# ==================================================
# 5. COMPILAR MODELO
# ==================================================

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="recall"
        ),
        tf.keras.metrics.AUC(
            curve="PR",
            name="pr_auc"
        )
    ]
)

print("\n=== ARQUITECTURA ===")
modelo.summary()

# ==================================================
# 6. CALLBACKS
# ==================================================

ruta_modelo = CARPETA_MODELO / "mlp_base.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        patience=7,
        mode="max",
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ruta_modelo),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True
    ),

    tf.keras.callbacks.CSVLogger(
        str(CARPETA_METRICAS / "mlp_historial.csv")
    )
]

# ==================================================
# 7. ENTRENAMIENTO
# ==================================================

print("\n=== INICIANDO ENTRENAMIENTO ===")

historial = modelo.fit(
    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCAS,
    batch_size=BATCH_SIZE,

    class_weight=class_weight,
    callbacks=callbacks,

    verbose=1
)

# ==================================================
# 8. PREDICCIONES DE VALIDACION
# ==================================================

print("\n=== EVALUACION EN VALIDACION ===")

probabilidades = modelo.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=0
).ravel()

# Evaluacion inicial con umbral 0.5
predicciones_05 = (
    probabilidades >= 0.5
).astype(int)

print("\n=== RESULTADOS UMBRAL 0.5 ===")
print(
    classification_report(
        y_val,
        predicciones_05,
        zero_division=0
    )
)

print("\nMatriz de confusion:")
print(confusion_matrix(y_val, predicciones_05))

# ==================================================
# 9. AJUSTAR UMBRAL EN VALIDACION
# ==================================================

precision, recall, umbrales = precision_recall_curve(
    y_val,
    probabilidades
)

# El ultimo elemento de precision/recall
# no corresponde a un umbral.
f1 = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

mejor_indice = int(np.argmax(f1))
mejor_umbral = float(umbrales[mejor_indice])

predicciones_ajustadas = (
    probabilidades >= mejor_umbral
).astype(int)

print("\n=== UMBRAL SELECCIONADO ===")
print("Umbral:", round(mejor_umbral, 4))
print("F1 validacion:", round(float(f1[mejor_indice]), 4))

print("\n=== CLASIFICACION AJUSTADA ===")
print(
    classification_report(
        y_val,
        predicciones_ajustadas,
        zero_division=0
    )
)

print("\nMatriz de confusion ajustada:")
print(confusion_matrix(y_val, predicciones_ajustadas))

pr_auc = average_precision_score(
    y_val,
    probabilidades
)

print("\nAverage Precision:", round(pr_auc, 4))

# ==================================================
# 10. GUARDAR METRICAS
# ==================================================

metricas = {
    "modelo": "MLP_Base",
    "capas_ocultas": [64, 32],
    "activacion": "relu",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epocas_ejecutadas": len(historial.history["loss"]),
    "average_precision_val": float(pr_auc),
    "mejor_umbral_val": mejor_umbral,
    "mejor_f1_val": float(f1[mejor_indice]),
    "pesos_clase": class_weight,
    "semilla": SEMILLA
}

with open(
    CARPETA_METRICAS / "mlp_resultados.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(metricas, archivo, indent=4)

# ==================================================
# 11. CURVAS DE ENTRENAMIENTO
# ==================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(13, 5)
)

axes[0].plot(
    historial.history["loss"],
    label="Train"
)

axes[0].plot(
    historial.history["val_loss"],
    label="Validation"
)

axes[0].set_title("Binary Crossentropy")
axes[0].set_xlabel("Epoca")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.2)

axes[1].plot(
    historial.history["pr_auc"],
    label="Train"
)

axes[1].plot(
    historial.history["val_pr_auc"],
    label="Validation"
)

axes[1].set_title("PR-AUC")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("PR-AUC")
axes[1].legend()
axes[1].grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICOS / "mlp_entrenamiento.png",
    dpi=180
)

plt.close(fig)

print("\n=== ENTRENAMIENTO FINALIZADO ===")
print("Modelo:", ruta_modelo)
print("Metricas y graficos guardados.")
print("Conjunto de prueba NO utilizado.")
