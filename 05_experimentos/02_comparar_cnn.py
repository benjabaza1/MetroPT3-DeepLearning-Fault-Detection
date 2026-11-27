
from pathlib import Path
import os
import gc
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1. CONFIGURACION
# ==========================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "01_datos" / "procesados"

RESULTADOS = BASE / "06_resultados"

CARPETA_METRICAS = (
    RESULTADOS / "metricas" / "experimentos_cnn"
)
CARPETA_MODELOS = (
    RESULTADOS / "modelos_entrenados" / "experimentos_cnn"
)
CARPETA_GRAFICOS = (
    RESULTADOS / "graficos" / "experimentos_cnn"
)

for ruta in [
    CARPETA_METRICAS,
    CARPETA_MODELOS,
    CARPETA_GRAFICOS
]:
    ruta.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# ==========================================
# 2. EXPERIMENTOS
# ==========================================

EXPERIMENTOS = [
    {
        "nombre": "CNN_01_Base",
        "filtros": [32, 64],
        "kernel": 3,
        "activacion": "relu"
    },
    {
        "nombre": "CNN_02_Pequena",
        "filtros": [16, 32],
        "kernel": 3,
        "activacion": "relu"
    },
    {
        "nombre": "CNN_03_Grande",
        "filtros": [64, 128],
        "kernel": 3,
        "activacion": "relu"
    },
    {
        "nombre": "CNN_04_Kernel5",
        "filtros": [32, 64],
        "kernel": 5,
        "activacion": "relu"
    },
    {
        "nombre": "CNN_05_Tanh",
        "filtros": [32, 64],
        "kernel": 3,
        "activacion": "tanh"
    }
]

# ==========================================
# 3. DATOS
# ==========================================

X_train = np.load(
    DATOS / "X_train.npy"
)
y_train = np.load(
    DATOS / "y_train.npy"
)

X_val = np.load(
    DATOS / "X_validation.npy"
)
y_val = np.load(
    DATOS / "y_validation.npy"
)

assert X_train.shape[1:] == (60, 15)
assert X_val.shape[1:] == (60, 15)

pesos = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

PESOS_CLASE = {
    0: float(pesos[0]),
    1: float(pesos[1])
}

print("\n=== COMPARACION CNN 1D ===")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Pesos:", PESOS_CLASE)

# ==========================================
# 4. CONSTRUCTOR CNN
# ==========================================

def construir_cnn(config):

    filtros_1, filtros_2 = config["filtros"]

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(60, 15)
        ),

        tf.keras.layers.Conv1D(
            filters=filtros_1,
            kernel_size=config["kernel"],
            padding="same",
            activation=config["activacion"]
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Conv1D(
            filters=filtros_2,
            kernel_size=config["kernel"],
            padding="same",
            activation=config["activacion"]
        ),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(
            32,
            activation=config["activacion"]
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ], name=config["nombre"])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"
            )
        ]
    )

    return modelo

# ==========================================
# 5. ENTRENAR Y COMPARAR
# ==========================================

resultados = []

for numero, config in enumerate(EXPERIMENTOS, 1):

    nombre = config["nombre"]

    print("\n" + "=" * 55)
    print(f"EXPERIMENTO {numero}/5: {nombre}")
    print("=" * 55)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    np.random.seed(SEMILLA)
    random.seed(SEMILLA)

    modelo = construir_cnn(config)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=7,
            restore_best_weights=True
        )
    ]

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

    # Evaluacion exclusivamente en validacion
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

    pred = (probabilidades >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        pred,
        labels=[0, 1]
    ).ravel()

    resultado = {
        "modelo": nombre,
        "filtros": str(config["filtros"]),
        "kernel": config["kernel"],
        "activacion": config["activacion"],
        "parametros": int(modelo.count_params()),
        "epocas": len(historial.history["loss"]),
        "average_precision": float(ap),
        "f1_umbral_05": float(
            f1_score(y_val, pred_05, zero_division=0)
        ),
        "umbral_ajustado": umbral,
        "precision_ajustada": float(
            precision_score(y_val, pred, zero_division=0)
        ),
        "recall_ajustado": float(
            recall_score(y_val, pred, zero_division=0)
        ),
        "f1_ajustado": float(
            f1_score(y_val, pred, zero_division=0)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    resultados.append(resultado)

    modelo.save(
        CARPETA_MODELOS / f"{nombre}.keras"
    )

    pd.DataFrame(historial.history).to_csv(
        CARPETA_METRICAS / f"{nombre}_historial.csv",
        index=False
    )

    pd.DataFrame({
        "real": y_val,
        "probabilidad": probabilidades,
        "prediccion_05": pred_05,
        "prediccion_ajustada": pred
    }).to_csv(
        CARPETA_METRICAS / f"{nombre}_predicciones.csv",
        index=False
    )

    # Guardar resultados parciales
    pd.DataFrame(resultados).to_csv(
        CARPETA_METRICAS / "comparacion_cnn.csv",
        index=False
    )

    print("\n=== RESULTADOS ===")
    print("Average Precision:", round(ap, 4))
    print("F1 ajustado:", round(resultado["f1_ajustado"], 4))
    print("Recall:", round(resultado["recall_ajustado"], 4))
    print("Parametros:", resultado["parametros"])

    del modelo, historial
    gc.collect()

# ==========================================
# 6. TABLA FINAL
# ==========================================

tabla = pd.DataFrame(resultados)

tabla = tabla.sort_values(
    ["average_precision", "f1_ajustado"],
    ascending=False
).reset_index(drop=True)

tabla.to_csv(
    CARPETA_METRICAS / "comparacion_cnn.csv",
    index=False
)

print("\n=== COMPARACION FINAL CNN 1D ===")

columnas = [
    "modelo",
    "parametros",
    "epocas",
    "average_precision",
    "f1_umbral_05",
    "f1_ajustado",
    "recall_ajustado"
]

print(tabla[columnas].to_string(index=False))

ganador = tabla.iloc[0]

print("\n=== MEJOR CNN EN VALIDACION ===")
print(ganador["modelo"])
print("Average Precision:", ganador["average_precision"])

# ==========================================
# 7. GRAFICO COMPARATIVO
# ==========================================

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(
    tabla["modelo"],
    tabla["average_precision"]
)

ax.set_title(
    "Comparacion de arquitecturas CNN 1D"
)

ax.set_ylabel("Average Precision - Validation")
ax.set_xlabel("Arquitectura")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    CARPETA_GRAFICOS / "comparacion_cnn.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("\nExperimentos CNN finalizados.")
print("TEST no utilizado.")
