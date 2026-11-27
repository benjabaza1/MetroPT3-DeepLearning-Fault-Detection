
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

RUTA = (
    BASE / "01_datos" / "original"
    / "MetroPT3(AirCompressor).csv"
)

SALIDA = BASE / "06_resultados" / "graficos"
SALIDA.mkdir(parents=True, exist_ok=True)

sensores = [
    "TP3",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature"
]

df = pd.read_csv(
    RUTA,
    usecols=["timestamp"] + sensores,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").set_index("timestamp")

eventos = [
    ("Falla_1", "2020-04-18 00:00", "2020-04-19 00:00"),
    ("Falla_2", "2020-05-29 23:30", "2020-05-30 06:01"),
    ("Falla_3", "2020-06-05 10:00", "2020-06-07 14:31"),
    ("Falla_4", "2020-07-15 14:30", "2020-07-15 19:01")
]

for nombre, inicio, fin in eventos:

    inicio = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin)

    # Incluimos 12 horas antes y después
    desde = inicio - pd.Timedelta(hours=12)
    hasta = fin + pd.Timedelta(hours=12)

    ventana = df.loc[desde:hasta]

    # Promedios de un minuto para visualizar tendencias
    ventana = ventana.resample("1min").mean()

    fig, axes = plt.subplots(
        4, 1,
        figsize=(15, 10),
        sharex=True
    )

    for ax, sensor in zip(axes, sensores):

        ax.plot(
            ventana.index,
            ventana[sensor],
            linewidth=1
        )

        ax.axvspan(
            inicio,
            fin,
            color="red",
            alpha=0.18,
            label="Falla documentada"
        )

        ax.axvline(
            inicio,
            color="red",
            linestyle="--",
            linewidth=1
        )

        ax.set_ylabel(sensor)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")

    fig.suptitle(
        f"{nombre}: comportamiento de sensores",
        fontsize=15
    )

    plt.tight_layout()

    archivo = SALIDA / f"{nombre}_detalle.png"

    plt.savefig(
        archivo,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Generado:", archivo)

print("\nAnalisis de eventos finalizado.")
