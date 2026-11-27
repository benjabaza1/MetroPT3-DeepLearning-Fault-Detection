
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
METRICAS = BASE / "06_resultados" / "metricas"
DESTINO = BASE / "08_portafolio" / "datos"

DESTINO.mkdir(parents=True, exist_ok=True)

archivos = {
    "MLP": METRICAS / "experimentos_mlp" / "comparacion_mlp.csv",
    "CNN 1D": METRICAS / "experimentos_cnn" / "comparacion_cnn.csv",
    "LSTM": METRICAS / "experimentos_lstm" / "comparacion_lstm.csv"
}

tablas = []

for familia, ruta in archivos.items():
    df = pd.read_csv(ruta)
    df.insert(0, "familia", familia)
    tablas.append(df)

experimentos = pd.concat(
    tablas,
    ignore_index=True
)

experimentos = experimentos.sort_values(
    "average_precision",
    ascending=False
)

experimentos.to_csv(
    DESTINO / "15_experimentos.csv",
    index=False
)

prueba = pd.read_csv(
    METRICAS / "evaluacion_final"
    / "comparacion_final_test.csv"
)

prueba.to_csv(
    DESTINO / "evaluacion_test.csv",
    index=False
)

ablacion = pd.read_csv(
    METRICAS / "ablacion"
    / "comparacion_ablacion.csv"
)

ablacion.to_csv(
    DESTINO / "ablacion.csv",
    index=False
)

importancias = pd.read_csv(
    METRICAS / "explicabilidad"
    / "importancia_sensores.csv"
)

importancias.to_csv(
    DESTINO / "importancia_sensores.csv",
    index=False
)

auditoria = pd.read_csv(
    METRICAS / "explicabilidad"
    / "auditoria_estados_resumen.csv"
)

auditoria.to_csv(
    DESTINO / "auditoria_estados.csv",
    index=False
)

print("\n=== CONSOLIDACION COMPLETADA ===")
print("Experimentos:", len(experimentos))
print("Modelos evaluados en test:", len(prueba))
print("Modelos de ablacion:", len(ablacion))
print("Registros de importancia:", len(importancias))
print("\nTodos los archivos estan en:")
print(DESTINO)

print("\n=== TOP 5 EXPERIMENTOS ===")
print(
    experimentos[
        ["familia", "modelo", "average_precision"]
    ].head(5).to_string(index=False)
)
