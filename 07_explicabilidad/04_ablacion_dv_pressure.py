
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

# ==================================================
# 1. CONFIGURACION
# ==================================================

BASE = Path(__file__).resolve().parent.parent

DATOS = BASE / "01_datos" / "procesados"
RESULTADOS = BASE / "06_resultados"

METRICAS = RESULTADOS / "metricas" / "ablacion"
MODELOS = RESULTADOS / "modelos_entrenados" / "ablacion"
GRAFICOS = RESULTADOS / "graficos" / "ablacion"

for ruta in (METRICAS, MODELOS, GRAFICOS):
    ruta.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

SENSORES = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses"
]

SENSOR_EXCLUIDO = "DV_pressure"
INDICE_EXCLUIDO = SENSORES.index(SENSOR_EXCLUIDO)

SENSORES_FINALES = [
    sensor for sensor in SENSORES
    if sensor != SENSOR_EXCLUIDO
]

print("\n=== ABLACION: SIN DV_PRESSURE ===")
print("Sensor excluido:", SENSOR_EXCLUIDO)
print("Indice:", INDICE_EXCLUIDO)
print("Sensores restantes:", len(SENSORES_FINALES))

# ==================================================
# 2. CARGAR DATOS
# ==================================================

X_train_original = np.load(
    DATOS / "X_train.npy",
    mmap_mode="r"
)

X_val_original = np.load(
    DATOS / "X_validation.npy",
    mmap_mode="r"
)

y_train = np.load(DATOS / "y_train.npy")
y_val = np.load(DATOS / "y_validation.npy")

assert X_train_original.shape == (14186, 60, 15)
assert X_val_original.shape == (3581, 60, 15)

# Eliminamos exclusivamente la columna 3.
# No modificamos los archivos originales.

X_train = np.delete(
    X_train_original,
    INDICE_EXCLUIDO,
    axis=2
).astype(np.float32)

X_val = np.delete(
    X_val_original,
    INDICE_EXCLUIDO,
    axis=2
).astype(np.float32)

assert X_train.shape == (14186, 60, 14)
assert X_val.shape == (3581, 60, 14)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

with open(
    METRICAS / "sensores_utilizados.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(
        {
            "sensor_excluido": SENSOR_EXCLUIDO,
            "sensores_utilizados": SENSORES_FINALES,
            "indices_originales": [
                i for i in range(15)
                if i != INDICE_EXCLUIDO
            ]
        },
        archivo,
        indent=4,
        ensure_ascii=False
    )

# ==================================================
# 3. PESOS DE CLASE
# ==================================================

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

# ==================================================
# 4. CONSTRUCTORES DE ARQUITECTURAS
# ==================================================

def crear_mlp():

    # Replica la MLP Tanh ganadora:
    # Dense 64 -> Dense 32 -> salida.

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(840,)),
        tf.keras.layers.Dense(
            64, activation="tanh"
        ),
        tf.keras.layers.Dense(
            32, activation="tanh"
        ),
        tf.keras.layers.Dense(
            1, activation="sigmoid"
        )
    ], name="MLP_Sin_DV")

    return modelo


def crear_cnn():

    # Replica CNN_05_Tanh.

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 14)),

        tf.keras.layers.Conv1D(
            32,
            kernel_size=3,
            padding="same",
            activation="tanh"
        ),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Conv1D(
            64,
            kernel_size=3,
            padding="same",
            activation="tanh"
        ),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(
            32,
            activation="tanh"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ], name="CNN_Sin_DV")

    return modelo


def crear_lstm():

    # Replica LSTM_01_Base.

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 14)),

        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ], name="LSTM_Sin_DV")

    return modelo


ARQUITECTURAS = {
    "MLP": crear_mlp,
    "CNN_1D": crear_cnn,
    "LSTM": crear_lstm
}

# ==================================================
# 5. ENTRENAR LOS TRES MODELOS
# ==================================================

resultados = []

for familia, constructor in ARQUITECTURAS.items():

    print("\n" + "=" * 55)
    print("ENTRENAMIENTO:", familia)
    print("SIN DV_PRESSURE")
    print("=" * 55)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    modelo = constructor()

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

    if familia == "MLP":
        entrada_train = X_train.reshape(-1, 840)
        entrada_val = X_val.reshape(-1, 840)
    else:
        entrada_train = X_train
        entrada_val = X_val

    print("Parametros:", modelo.count_params())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=7,
            restore_best_weights=True
        )
    ]

    historial = modelo.fit(
        entrada_train,
        y_train,
        validation_data=(entrada_val, y_val),
        epochs=EPOCAS,
        batch_size=BATCH_SIZE,
        class_weight=PESOS_CLASE,
        callbacks=callbacks,
        verbose=2
    )

    # ==============================================
    # 6. EVALUACION SOLO EN VALIDACION
    # ==============================================

    probabilidades = modelo.predict(
        entrada_val,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    ap = average_precision_score(
        y_val, probabilidades
    )

    pred_05 = (
        probabilidades >= 0.5
    ).astype(int)

    precision, recall, umbrales = (
        precision_recall_curve(
            y_val, probabilidades
        )
    )

    f1_valores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    indice = int(np.argmax(f1_valores))
    umbral_ajustado = float(umbrales[indice])

    pred = (
        probabilidades >= umbral_ajustado
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        pred,
        labels=[0, 1]
    ).ravel()

    resultado = {
        "modelo": familia,
        "experimento": "Sin DV_pressure",
        "sensores": 14,
        "parametros": int(modelo.count_params()),
        "epocas": len(historial.history["loss"]),
        "average_precision": float(ap),
        "f1_umbral_05": float(
            f1_score(
                y_val, pred_05,
                zero_division=0
            )
        ),
        "umbral_ajustado": umbral_ajustado,
        "precision_ajustada": float(
            precision_score(
                y_val, pred,
                zero_division=0
            )
        ),
        "recall_ajustado": float(
            recall_score(
                y_val, pred,
                zero_division=0
            )
        ),
        "f1_ajustado": float(
            f1_score(
                y_val, pred,
                zero_division=0
            )
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "semilla": SEMILLA
    }

    resultados.append(resultado)

    # ==============================================
    # 7. GUARDAR EVIDENCIA
    # ==============================================

    modelo.save(
        MODELOS / f"{familia}_sin_dv.keras"
    )

    pd.DataFrame(historial.history).to_csv(
        METRICAS / f"{familia}_historial.csv",
        index=False
    )

    pd.DataFrame({
        "real": y_val,
        "puntuacion": probabilidades,
        "prediccion_05": pred_05,
        "prediccion_ajustada": pred
    }).to_csv(
        METRICAS / f"{familia}_predicciones.csv",
        index=False
    )

    pd.DataFrame(resultados).to_csv(
        METRICAS / "comparacion_ablacion.csv",
        index=False
    )

    print("\n=== RESULTADOS ===")
    print("Average Precision:", round(ap, 6))
    print(
        "F1 ajustado:",
        round(resultado["f1_ajustado"], 6)
    )
    print(
        "Recall:",
        round(resultado["recall_ajustado"], 6)
    )
    print("Parametros:", resultado["parametros"])

    del modelo, historial
    gc.collect()

# ==================================================
# 8. COMPARACION CON LOS MODELOS ORIGINALES
# ==================================================

ORIGINALES = {
    "MLP": 0.997425,
    "CNN_1D": 0.988896,
    "LSTM": 0.998799
}

tabla = pd.DataFrame(resultados)

tabla["ap_original_15_sensores"] = (
    tabla["modelo"].map(ORIGINALES)
)

tabla["diferencia_ap"] = (
    tabla["average_precision"]
    - tabla["ap_original_15_sensores"]
)

tabla.to_csv(
    METRICAS / "comparacion_ablacion.csv",
    index=False
)

print("\n=== COMPARACION ABLACION ===")

columnas = [
    "modelo",
    "parametros",
    "ap_original_15_sensores",
    "average_precision",
    "diferencia_ap",
    "f1_ajustado"
]

print(tabla[columnas].to_string(index=False))

# ==================================================
# 9. GRAFICO COMPARATIVO
# ==================================================

x = np.arange(len(tabla))
ancho = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    x - ancho / 2,
    tabla["ap_original_15_sensores"],
    ancho,
    label="15 sensores"
)

ax.bar(
    x + ancho / 2,
    tabla["average_precision"],
    ancho,
    label="14 sensores"
)

ax.set_xticks(x)
ax.set_xticklabels(tabla["modelo"])

ax.set_ylim(0, 1.05)
ax.set_ylabel("Average Precision - Validacion")

ax.set_title(
    "Ablacion DV_pressure - Comparacion en validacion"
)

ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    GRAFICOS / "comparacion_ablacion.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("\n=== EXPERIMENTO FINALIZADO ===")
print("Modelos nuevos guardados.")
print("Modelos originales conservados.")
print("TEST no utilizado.")
print("Estudio exploratorio posterior al test original.")
