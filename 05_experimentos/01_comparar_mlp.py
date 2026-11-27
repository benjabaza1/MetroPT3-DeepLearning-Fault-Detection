
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
# 1. CONFIGURACION GENERAL
# ==================================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "01_datos" / "procesados"

SALIDA = BASE / "06_resultados"
METRICAS = SALIDA / "metricas" / "experimentos_mlp"
GRAFICOS = SALIDA / "graficos" / "experimentos_mlp"
MODELOS = SALIDA / "modelos_entrenados" / "experimentos_mlp"

for carpeta in [METRICAS, GRAFICOS, MODELOS]:
    carpeta.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
EPOCAS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# ==================================================
# 2. CONFIGURACIONES EXPERIMENTALES
# ==================================================

EXPERIMENTOS = [
    {
        "nombre": "MLP_01_Base",
        "capas": [64, 32],
        "activacion": "relu",
        "dropout": 0.0
    },
    {
        "nombre": "MLP_02_Pequena",
        "capas": [32, 16],
        "activacion": "relu",
        "dropout": 0.0
    },
    {
        "nombre": "MLP_03_Profunda",
        "capas": [128, 64, 32],
        "activacion": "relu",
        "dropout": 0.0
    },
    {
        "nombre": "MLP_04_Tanh",
        "capas": [64, 32],
        "activacion": "tanh",
        "dropout": 0.0
    },
    {
        "nombre": "MLP_05_Dropout",
        "capas": [64, 32],
        "activacion": "relu",
        "dropout": 0.30
    }
]

# ==================================================
# 3. CARGAR MATRICES
# ==================================================

print("\n=== EXPERIMENTACION MLP ===")

X_train = np.load(DATOS / "X_train.npy")
y_train = np.load(DATOS / "y_train.npy")

X_val = np.load(DATOS / "X_validation.npy")
y_val = np.load(DATOS / "y_validation.npy")

# Convertimos cada ventana 60 x 15 a 900 valores.
X_train = X_train.reshape(-1, 900)
X_val = X_val.reshape(-1, 900)

assert X_train.shape == (14186, 900)
assert X_val.shape == (3581, 900)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)

# ==================================================
# 4. CALCULAR PESOS DE CLASE
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

print("\nPesos de clase:", PESOS_CLASE)

# ==================================================
# 5. FUNCION CONSTRUCTORA DE MLP
# ==================================================

def construir_mlp(configuracion):

    modelo = tf.keras.Sequential(
        name=configuracion["nombre"]
    )

    modelo.add(tf.keras.layers.Input(shape=(900,)))

    for neuronas in configuracion["capas"]:

        modelo.add(
            tf.keras.layers.Dense(
                neuronas,
                activation=configuracion["activacion"]
            )
        )

        if configuracion["dropout"] > 0:
            modelo.add(
                tf.keras.layers.Dropout(
                    configuracion["dropout"]
                )
            )

    modelo.add(
        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
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
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return modelo

# ==================================================
# 6. FUNCION PARA EVALUAR
# ==================================================

def evaluar_modelo(modelo, configuracion, historial):

    nombre = configuracion["nombre"]

    probabilidades = modelo.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    # Average Precision calculada con sklearn
    ap = average_precision_score(
        y_val,
        probabilidades
    )

    # Predicciones con umbral fijo
    pred_05 = (probabilidades >= 0.5).astype(int)

    f1_05 = f1_score(
        y_val, pred_05, zero_division=0
    )

    # Umbral ajustado utilizando SOLO validacion
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
        y_val, pred_ajustada, labels=[0, 1]
    ).ravel()

    # Comparacion de perdidas SIN pesos de clase.
    # Solo al final, con los mejores pesos restaurados.
    loss_train = modelo.evaluate(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        verbose=0,
        return_dict=True
    )["loss"]

    loss_val = modelo.evaluate(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        verbose=0,
        return_dict=True
    )["loss"]

    resultado = {
        "modelo": nombre,
        "capas": str(configuracion["capas"]),
        "activacion": configuracion["activacion"],
        "dropout": configuracion["dropout"],
        "parametros": int(modelo.count_params()),
        "epocas": len(historial.history["loss"]),
        "average_precision": float(ap),
        "f1_umbral_05": float(f1_05),
        "umbral_ajustado": float(umbral),
        "precision_ajustada": float(
            precision_score(y_val, pred_ajustada,
                            zero_division=0)
        ),
        "recall_ajustado": float(
            recall_score(y_val, pred_ajustada,
                         zero_division=0)
        ),
        "f1_ajustado": float(
            f1_score(y_val, pred_ajustada,
                     zero_division=0)
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "loss_train_sin_pesos": float(loss_train),
        "loss_val_sin_pesos": float(loss_val)
    }

    # Guardar predicciones para analisis posterior
    pd.DataFrame({
        "real": y_val,
        "probabilidad": probabilidades,
        "prediccion_05": pred_05,
        "prediccion_ajustada": pred_ajustada
    }).to_csv(
        METRICAS / f"{nombre}_predicciones.csv",
        index=False
    )

    return resultado

# ==================================================
# 7. ENTRENAR CINCO ARQUITECTURAS
# ==================================================

resultados = []

for numero, configuracion in enumerate(EXPERIMENTOS, 1):

    nombre = configuracion["nombre"]

    print("\n" + "=" * 55)
    print(f"EXPERIMENTO {numero}/5: {nombre}")
    print("=" * 55)

    # Misma semilla inicial para cada experimento
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEMILLA)
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)

    modelo = construir_mlp(configuracion)

    print("\nParametros:", modelo.count_params())

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

    # Guardar modelo con los mejores pesos restaurados
    modelo.save(
        MODELOS / f"{nombre}.keras"
    )

    # Guardar curvas por epoca
    pd.DataFrame(historial.history).to_csv(
        METRICAS / f"{nombre}_historial.csv",
        index=False
    )

    resultado = evaluar_modelo(
        modelo, configuracion, historial
    )

    resultados.append(resultado)

    # Guardar resultados parciales
    tabla = pd.DataFrame(resultados)

    tabla.to_csv(
        METRICAS / "comparacion_mlp.csv",
        index=False
    )

    print("\n=== RESULTADOS ===")
    print("Average Precision:",
          round(resultado["average_precision"], 4))
    print("F1 ajustado:",
          round(resultado["f1_ajustado"], 4))
    print("Recall ajustado:",
          round(resultado["recall_ajustado"], 4))
    print("Umbral:",
          round(resultado["umbral_ajustado"], 5))

    del modelo, historial
    gc.collect()

# ==================================================
# 8. ORDENAR Y SELECCIONAR MEJOR MODELO
# ==================================================

comparacion = pd.DataFrame(resultados)

comparacion = comparacion.sort_values(
    ["average_precision", "f1_ajustado"],
    ascending=False
).reset_index(drop=True)

comparacion.to_csv(
    METRICAS / "comparacion_mlp.csv",
    index=False
)

print("\n=== COMPARACION FINAL ===")

columnas = [
    "modelo",
    "parametros",
    "epocas",
    "average_precision",
    "f1_umbral_05",
    "f1_ajustado",
    "recall_ajustado"
]

print(
    comparacion[columnas].to_string(index=False)
)

ganador = comparacion.iloc[0].to_dict()

with open(
    METRICAS / "mejor_mlp.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(ganador, f, indent=4)

# ==================================================
# 9. GRAFICO COMPARATIVO
# ==================================================

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(
    comparacion["modelo"],
    comparacion["average_precision"]
)

ax.set_title(
    "Comparacion de arquitecturas MLP - MetroPT-3"
)

ax.set_ylabel("Average Precision - Validacion")
ax.set_xlabel("Arquitectura")

ax.set_ylim(
    0,
    min(1.05, max(0.1, comparacion["average_precision"].max() * 1.1))
)

ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    GRAFICOS / "comparacion_average_precision.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)

print("\n=== MEJOR MODELO SEGUN VALIDACION ===")
print(ganador["modelo"])
print("Average Precision:",
      round(ganador["average_precision"], 4))
print("F1 ajustado:",
      round(ganador["f1_ajustado"], 4))

print("\nExperimentos terminados.")
print("El conjunto TEST no fue utilizado.")
