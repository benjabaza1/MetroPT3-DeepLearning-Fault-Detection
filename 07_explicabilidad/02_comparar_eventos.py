
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

DATOS = BASE / "01_datos"
SALIDA = BASE / "06_resultados"

METRICAS = SALIDA / "metricas" / "explicabilidad"
GRAFICOS = SALIDA / "graficos" / "explicabilidad"

METRICAS.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

SENSORES = [
    "DV_pressure",
    "TP2",
    "TP3",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current"
]

# ==========================================
# 1. LEER DATOS ORIGINALES Y ETIQUETAS
# ==========================================

df = pd.read_csv(
    DATOS / "original" / "MetroPT3(AirCompressor).csv",
    usecols=["timestamp"] + SENSORES,
    parse_dates=["timestamp"]
)

etiquetas = pd.read_csv(
    DATOS / "procesados" / "etiquetas_temporales.csv",
    usecols=["fila_csv", "evento", "conjunto", "usable"]
)

assert len(df) == len(etiquetas)
assert np.array_equal(
    etiquetas["fila_csv"].to_numpy(),
    np.arange(len(df))
)

df = pd.concat(
    [
        df.reset_index(drop=True),
        etiquetas[["evento", "conjunto", "usable"]]
    ],
    axis=1
)

# ==========================================
# 2. ESTADISTICAS POR EVENTO
# ==========================================

fallas = df[
    df["evento"] != "Normal"
].copy()

resumen = (
    fallas.groupby("evento")[SENSORES]
    .agg(["mean", "median", "std", "min", "max"])
)

resumen.to_csv(
    METRICAS / "comparacion_sensores_eventos.csv"
)

print("\n=== DV_PRESSURE POR EVENTO ===")

print(
    fallas.groupby("evento")["DV_pressure"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .to_string()
)

# ==========================================
# 3. DISTRIBUCION DE DV_PRESSURE
# ==========================================

eventos = ["Falla_1", "Falla_2", "Falla_3", "Falla_4"]

fig, ax = plt.subplots(figsize=(11, 6))

datos = [
    fallas.loc[
        fallas["evento"] == evento,
        "DV_pressure"
    ].to_numpy()
    for evento in eventos
]

ax.boxplot(
    datos,
    tick_labels=eventos,
    showfliers=False
)

ax.set_title(
    "Distribucion de DV_pressure por evento de falla"
)
ax.set_ylabel("DV_pressure - valor original")
ax.set_xlabel("Evento documentado")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(
    GRAFICOS / "dv_pressure_por_evento.png",
    dpi=180,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# 4. COMPARAR CON PERIODOS NORMALES
# ==========================================

print("\n=== DV_PRESSURE POR CONJUNTO Y ESTADO ===")

tabla = (
    df.groupby(["conjunto", "evento"])["DV_pressure"]
    .agg(["count", "mean", "median", "std"])
)

print(tabla.to_string())

tabla.to_csv(
    METRICAS / "dv_pressure_por_conjunto.csv"
)

# ==========================================
# 5. COMPARACION TEMPORAL
# ==========================================

for evento in eventos:

    datos_evento = fallas[
        fallas["evento"] == evento
    ]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        datos_evento["timestamp"],
        datos_evento["DV_pressure"],
        linewidth=0.7
    )

    ax.set_title(f"{evento} - DV_pressure")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Presion")
    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        GRAFICOS / f"{evento}_dv_pressure.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

print("\nAnalisis descriptivo completado.")
print("No se modificaron modelos ni umbrales.")
