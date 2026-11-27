
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

RESULTADOS = (
    BASE / "06_resultados"
    / "metricas" / "evaluacion_final"
)

CONFIGURACIONES = {
    "MLP": 0.98295,
    "CNN_1D": 0.04888,
    "LSTM": 0.62982
}

print("\n=== DIAGNOSTICO DE GENERALIZACION ===")
print("Analisis descriptivo, sin modificar umbrales")

resumen = []

for familia, umbral in CONFIGURACIONES.items():

    ruta = RESULTADOS / f"predicciones_{familia}.csv"
    df = pd.read_csv(ruta)

    normales = df.loc[
        df["real"] == 0, "puntuacion"
    ]

    fallas = df.loc[
        df["real"] == 1, "puntuacion"
    ]

    print(f"\n=== {familia} ===")

    print("Umbral congelado:", umbral)

    print("\nPuntuaciones normales:")
    print(normales.describe())

    print("\nPuntuaciones de falla:")
    print(fallas.describe())

    print("\nFallas por encima del umbral:",
          int((fallas >= umbral).sum()))

    print("Fallas por debajo del umbral:",
          int((fallas < umbral).sum()))

    resumen.append({
        "modelo": familia,
        "umbral": umbral,
        "media_normal": normales.mean(),
        "media_falla": fallas.mean(),
        "max_normal": normales.max(),
        "max_falla": fallas.max(),
        "mediana_normal": normales.median(),
        "mediana_falla": fallas.median()
    })

resultado = pd.DataFrame(resumen)

resultado.to_csv(
    RESULTADOS / "diagnostico_puntuaciones.csv",
    index=False
)

print("\n=== RESUMEN ===")
print(resultado.to_string(index=False))

print("\nDiagnostico completado.")
print("Ningun modelo ni umbral fue modificado.")
