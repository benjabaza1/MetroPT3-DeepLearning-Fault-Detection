
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

RUTA_CSV = (
    BASE / "01_datos" / "original"
    / "MetroPT3(AirCompressor).csv"
)

SALIDA = BASE / "06_resultados" / "graficos"
SALIDA.mkdir(parents=True, exist_ok=True)

# Sensores para esta primera visualización
columnas = [
    "timestamp",
    "TP3",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature"
]

print("Cargando sensores seleccionados...")

df = pd.read_csv(
    RUTA_CSV,
    usecols=columnas,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp")
df = df.set_index("timestamp")

# Fechas tomadas de la documentación oficial
eventos = [
    ("Falla 1", "2020-04-18 00:00", "2020-04-19 00:00"),
    ("Falla 2", "2020-05-29 23:30", "2020-05-30 06:01"),
    ("Falla 3", "2020-06-05 10:00", "2020-06-07 14:31"),
    ("Falla 4", "2020-07-15 14:30", "2020-07-15 19:01")
]

sensores = [
    "TP3",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature"
]

# Medias en bloques de 10 minutos
# Los bloques sin mediciones permanecen vacíos
resumen = df.resample("10min").mean()

fig, axes = plt.subplots(
    4, 1,
    figsize=(15, 12),
    sharex=True
)

for ax, sensor in zip(axes, sensores):
    ax.plot(
        resumen.index,
        resumen[sensor],
        linewidth=0.8,
        label=sensor
    )

    for nombre, inicio, fin in eventos:
        ax.axvspan(
            pd.Timestamp(inicio),
            pd.Timestamp(fin),
            alpha=0.18,
            color="red"
        )

    ax.set_ylabel(sensor)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Fecha")

fig.suptitle(
    "MetroPT-3: evolución de sensores "
    "e intervalos documentados de falla",
    fontsize=15
)

plt.tight_layout()

archivo = SALIDA / "evolucion_sensores_fallas.png"

plt.savefig(
    archivo,
    dpi=180,
    bbox_inches="tight"
)

print("Grafico guardado en:", archivo)

plt.show()
