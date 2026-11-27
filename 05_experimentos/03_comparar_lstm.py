
from pathlib import Path
import os
import gc
import json
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

METRICAS = RESULTADOS / "metricas" / "experimentos_lstm"
MODELOS = RESULTADOS / "modelos_entrenados" / "experimentos_lstm"
GRAFICOS = RESULTADOS / "graficos" / "experimentos_lstm"

for ruta in [METRICAS, MODELOS, GRAFICOS]:
    ruta.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

EXPERIMENTOS = [
    {
        "nombre": "LSTM_01_Base",
        "unidades": [32],
        "dense": 16,
        "activacion": "relu"
    },
    {
        "nombre": "LSTM_02_Pequena",
        "unidades": [16],
        "dense": 16,
        "activacion": "relu"
    },
    {
        "nombre": "LSTM_03_Grande",
        "unidades": [64],
        "dense": 16,
        "activacion": "relu"
    },
    {
        "nombre": "LSTM_04_Profunda",
        "unidades": [32, 16],
        "dense": 16,
        "activacion": "relu"
    },
    {
        "nombre": "LSTM_05_Tanh",
        "unidades": [32],
        "dense": 16,
        "activacion": "tanh"
    }
]

# ==========================================
# 2. CARGAR MATRICES
# ==========================================

X_train = np.load(DATOS / "X_train.npy")
y_train = np.load(DATOS / "y_train.npy")

X_val = np.load(DATOS / "X_validation.npy")
y_val = np.load(DATOS / "y_validation.npy")

assert X_train.shape == (14186, 60, 15)
assert X_val.shape == (3581, 60, 15)

pesos = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

PESOS_CLASE = {
    0: float(pesos[0]),
    1: float(pesos[1])
}

print("\n=== COMPARACION LSTM ===")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Pesos:", PESOS_CLASE)

# ==========================================
# 3. CONSTRUCTOR DE LSTM
# ==========================================

def construir_lstm(config):

    modelo = tf.keras.Sequential(
        name=config["nombre"]
    )

    modelo.add(
        tf.keras.layers.Input(shape=(60, 15))
    )

    for i, unidades in enumerate(config["unidades"]):
        # Todas las capas, excepto la ultima,
        # deben entregar secuencias completas.
        devolver_secuencias = (
            i < len(config["unidades"]) - 1
        )

        modelo.add(
            tf.keras.layers.LSTM(
                units=unidades,
                return_sequences=devolver_secuencias,
                name=f"lstm_{i+1}"
            )
        )

    modelo.add(
        tf.keras.layers.Dense(
            config["dense"],
            activation=config["activacion"],
            name="dense"
        )
    )

    modelo.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="salida"
        )
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    return modelo

# ==========================================
# 4. EXPERIMENTOS
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

    modelo = construir_lstm(config)
    modelo.summary()

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

    # ======================================
    # 5. EVALUAR SOLO EN VALIDACION
    # ======================================

    probabilidades = modelo.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    ap = average_precision_score(
        y_val,
        probabilidades
    )

    pred_05 = (
        probabilidades >= 0.5
    ).astype(int)

    precision, recall, umbrales = (
        precision_recall_curve(
            y_val,
            probabilidades
        )
    )

    f1_valores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    indice = int(np.argmax(f1_valores))
    umbral = float(umbrales[indice])

    pred = (
        probabilidades >= umbral
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        pred,
        labels=[0, 1]
    ).ravel()

    resultado = {
        "modelo": nombre,
        "unidades": str(config["unidades"]),
        "activacion_dense": config["activacion"],
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

    # ======================================
    # 6. GUARDAR EVIDENCIA
    # ======================================

    modelo.save(
        MODELOS / f"{nombre}.keras"
    )

    pd.DataFrame(historial.history).to_csv(
        METRICAS / f"{nombre}_historial.csv",
        index=False
    )

    pd.DataFrame({
        "real": y_val,
        "probabilidad": probabilidades,
        "prediccion_05": pred_05,
        "prediccion_ajustada": pred
    }).to_csv(
        METRICAS / f"{nombre}_predicciones.csv",
        index=False
    )

    pd.DataFrame(resultados).to_csv(
        METRICAS / "comparacion_lstm.csv",
        index=False
    )

    print("\n=== RESULTADOS ===")
    print("Average Precision:", round(ap, 4))
    print("F1 ajustado:", round(resultado["f1_ajustado"], 4))
    print("Recall:", round(resultado["recall_ajustado"], 4))
    print("Parametros:", resultado["parametros"])
    print("Epocas:", resultado["epocas"])

    del modelo, historial
    gc.collect()

# ==========================================
# 7. COMPARACION FINAL
# ==========================================

tabla = pd.DataFrame(resultados)

tabla = tabla.sort_values(
    ["average_precision", "f1_ajustado"],
    ascending=False
).reset_index(drop=True)

tabla.to_csv(
    METRICAS / "comparacion_lstm.csv",
    index=False
)

print("\n=== COMPARACION FINAL LSTM ===")

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

ganador = tabla.iloc[0].to_dict()

with open(
    METRICAS / "mejor_lstm.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(ganador, archivo, indent=4)

# ==========================================
# 8. GRAFICO
# ==========================================

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(
    tabla["modelo"],
    tabla["average_precision"]
)

ax.set_title(
    "Comparacion de arquitecturas LSTM - MetroPT-3"
)
ax.set_xlabel("Arquitectura")
ax.set_ylabel("Average Precision - Validacion")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    GRAFICOS / "comparacion_lstm.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("\n=== MEJOR LSTM EN VALIDACION ===")
print(ganador["modelo"])
print("Average Precision:",
      round(ganador["average_precision"], 4))
print("F1 ajustado:",
      round(ganador["f1_ajustado"], 4))

print("\nExperimentos LSTM finalizados.")
print("TEST no utilizado.")
