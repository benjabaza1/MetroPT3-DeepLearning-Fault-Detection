
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
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================================
# 1. CONFIGURACION
# ==========================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "01_datos" / "procesados"
RESULTADOS = BASE / "06_resultados"

MODELOS = RESULTADOS / "modelos_entrenados"
METRICAS = RESULTADOS / "metricas"
GRAFICOS = RESULTADOS / "graficos"

for ruta in [MODELOS, METRICAS, GRAFICOS]:
    ruta.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.keras.utils.set_random_seed(SEMILLA)

# ==========================================
# 2. CARGAR DATOS
# ==========================================

print("\n=== LSTM - METROPT-3 ===")

X_train = np.load(DATOS / "X_train.npy")
y_train = np.load(DATOS / "y_train.npy")

X_val = np.load(DATOS / "X_validation.npy")
y_val = np.load(DATOS / "y_validation.npy")

assert X_train.shape == (14186, 60, 15)
assert X_val.shape == (3581, 60, 15)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

# ==========================================
# 3. PESOS DE CLASE
# ==========================================

pesos = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

PESOS_CLASE = {
    0: float(pesos[0]),
    1: float(pesos[1])
}

print("Pesos de clase:", PESOS_CLASE)

# ==========================================
# 4. ARQUITECTURA LSTM
# ==========================================

modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(60, 15)),

    tf.keras.layers.LSTM(
        units=32,
        name="memoria_temporal"
    ),

    tf.keras.layers.Dense(
        units=16,
        activation="relu",
        name="capa_densa"
    ),

    tf.keras.layers.Dense(
        units=1,
        activation="sigmoid",
        name="salida_falla"
    )
], name="LSTM_MetroPT3")

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

print("\n=== ARQUITECTURA LSTM ===")
modelo.summary()

# ==========================================
# 5. CALLBACKS
# ==========================================

ruta_modelo = MODELOS / "lstm_base.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=7,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ruta_modelo),
        monitor="val_pr_auc",
        mode="max",
        save_best_only=True
    ),

    tf.keras.callbacks.CSVLogger(
        str(METRICAS / "lstm_historial.csv")
    )
]

# ==========================================
# 6. ENTRENAR MODELO
# ==========================================

print("\n=== ENTRENAMIENTO LSTM ===")

historial = modelo.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCAS,
    batch_size=BATCH_SIZE,
    class_weight=PESOS_CLASE,
    callbacks=callbacks,
    verbose=2
)

# ==========================================
# 7. PREDICCIONES DE VALIDACION
# ==========================================

probabilidades = modelo.predict(
    X_val,
    batch_size=BATCH_SIZE,
    verbose=0
).ravel()

ap = average_precision_score(
    y_val,
    probabilidades
)

pred_05 = (probabilidades >= 0.5).astype(int)

f1_05 = f1_score(
    y_val,
    pred_05,
    zero_division=0
)

# ==========================================
# 8. SELECCIONAR UMBRAL
# ==========================================

precision, recall, umbrales = precision_recall_curve(
    y_val,
    probabilidades
)

f1_valores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

indice = int(np.argmax(f1_valores))
umbral = float(umbrales[indice])

pred_ajustada = (
    probabilidades >= umbral
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_val,
    pred_ajustada,
    labels=[0, 1]
).ravel()

resultado = {
    "modelo": "LSTM_Base",
    "parametros": int(modelo.count_params()),
    "epocas": len(historial.history["loss"]),
    "average_precision": float(ap),
    "f1_umbral_05": float(f1_05),
    "umbral_ajustado": umbral,
    "precision_ajustada": float(
        precision_score(
            y_val, pred_ajustada,
            zero_division=0
        )
    ),
    "recall_ajustado": float(
        recall_score(
            y_val, pred_ajustada,
            zero_division=0
        )
    ),
    "f1_ajustado": float(
        f1_score(
            y_val, pred_ajustada,
            zero_division=0
        )
    ),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "semilla": SEMILLA
}

print("\n=== RESULTADOS LSTM ===")

for clave, valor in resultado.items():
    print(f"{clave}: {valor}")

# ==========================================
# 9. GUARDAR RESULTADOS
# ==========================================

with open(
    METRICAS / "lstm_resultados.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(resultado, archivo, indent=4)

pd.DataFrame({
    "real": y_val,
    "probabilidad": probabilidades,
    "prediccion_05": pred_05,
    "prediccion_ajustada": pred_ajustada
}).to_csv(
    METRICAS / "lstm_predicciones.csv",
    index=False
)

# ==========================================
# 10. GRAFICOS
# ==========================================

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

axes[0].set_title("LSTM - Binary Crossentropy")
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

axes[1].set_title("LSTM - PR-AUC")
axes[1].set_xlabel("Epoca")
axes[1].set_ylabel("PR-AUC")
axes[1].legend()
axes[1].grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    GRAFICOS / "lstm_entrenamiento.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("\n=== LSTM FINALIZADA ===")
print("Modelo guardado:", ruta_modelo)
print("TEST no utilizado.")
