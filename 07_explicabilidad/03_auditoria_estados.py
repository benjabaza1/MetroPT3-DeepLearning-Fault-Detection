
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACION
# ==========================================

BASE = Path(__file__).resolve().parent.parent

DATOS = BASE / "01_datos"
RESULTADOS = BASE / "06_resultados"

METRICAS = RESULTADOS / "metricas" / "explicabilidad"
GRAFICOS = RESULTADOS / "graficos" / "explicabilidad"

METRICAS.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

DIGITALES = ["COMP", "DV_eletric", "MPG"]

ANALOGICOS = [
    "DV_pressure",
    "Motor_current",
    "TP3",
    "Reservoirs"
]

COLUMNAS = ["timestamp"] + DIGITALES + ANALOGICOS

# ==========================================
# 2. CARGAR DATOS ORIGINALES
# ==========================================

print("\n=== AUDITORIA DE ESTADOS ===")

df = pd.read_csv(
    DATOS / "original" / "MetroPT3(AirCompressor).csv",
    usecols=COLUMNAS,
    parse_dates=["timestamp"]
)

etiquetas = pd.read_csv(
    DATOS / "procesados" / "etiquetas_temporales.csv",
    usecols=["fila_csv", "evento", "conjunto"]
)

assert len(df) == len(etiquetas)
assert np.array_equal(
    etiquetas["fila_csv"].to_numpy(),
    np.arange(len(df))
)

df["evento"] = etiquetas["evento"].to_numpy()
df["conjunto"] = etiquetas["conjunto"].to_numpy()

print("Registros:", len(df))

# ==========================================
# 3. DISTRIBUCION DE SENALES DIGITALES
# ==========================================

print("\n=== PORCENTAJE DE ESTADO ACTIVO ===")

eventos = [
    "Normal",
    "Falla_1",
    "Falla_2",
    "Falla_3",
    "Falla_4"
]

filas = []

for evento in eventos:

    # Las observaciones normales se separan
    # por conjunto para evitar mezclarlas.
    if evento == "Normal":
        grupos = [
            ("Normal_train", df[
                (df["evento"] == "Normal") &
                (df["conjunto"] == "train")
            ]),
            ("Normal_validation", df[
                (df["evento"] == "Normal") &
                (df["conjunto"] == "validation")
            ]),
            ("Normal_test", df[
                (df["evento"] == "Normal") &
                (df["conjunto"] == "test")
            ])
        ]
    else:
        grupos = [(evento, df[df["evento"] == evento])]

    for nombre, grupo in grupos:

        if grupo.empty:
            continue

        fila = {
            "grupo": nombre,
            "registros": len(grupo)
        }

        for sensor in DIGITALES:
            fila[f"{sensor}_activo_pct"] = (
                grupo[sensor].eq(1).mean() * 100
            )

        for sensor in ANALOGICOS:
            fila[f"{sensor}_media"] = grupo[sensor].mean()
            fila[f"{sensor}_mediana"] = grupo[sensor].median()

        filas.append(fila)

resumen = pd.DataFrame(filas)

print(resumen.round(4).to_string(index=False))

resumen.to_csv(
    METRICAS / "auditoria_estados_resumen.csv",
    index=False
)

# ==========================================
# 4. COMBINACIONES DE ESTADOS
# ==========================================

print("\n=== COMBINACIONES DIGITALES ===")

combinaciones = (
    df.groupby(
        ["evento", "conjunto"] + DIGITALES
    )
    .agg(
        registros=("DV_pressure", "size"),
        dv_media=("DV_pressure", "mean"),
        corriente_media=("Motor_current", "mean")
    )
    .reset_index()
)

combinaciones["porcentaje_grupo"] = (
    combinaciones["registros"]
    / combinaciones.groupby(
        ["evento", "conjunto"]
    )["registros"].transform("sum")
    * 100
)

combinaciones = combinaciones.sort_values(
    ["evento", "conjunto", "registros"],
    ascending=[True, True, False]
)

combinaciones.to_csv(
    METRICAS / "combinaciones_estados.csv",
    index=False
)

for evento in eventos:

    subset = combinaciones[
        combinaciones["evento"] == evento
    ]

    print(f"\n--- {evento} ---")
    print(
        subset[
            [
                "conjunto",
                "COMP", "DV_eletric", "MPG",
                "registros", "porcentaje_grupo",
                "dv_media", "corriente_media"
            ]
        ].head(12).round(4).to_string(index=False)
    )

# ==========================================
# 5. RELACION ENTRE ESTADO Y DV_PRESSURE
# ==========================================

print("\n=== DV_PRESSURE SEGUN ESTADO DIGITAL ===")

relaciones = (
    df.groupby(
        ["evento", "conjunto", "DV_eletric", "COMP"]
    )["DV_pressure"]
    .agg(["count", "mean", "median", "std"])
    .reset_index()
)

print(relaciones.round(4).to_string(index=False))

relaciones.to_csv(
    METRICAS / "relacion_estados_presion.csv",
    index=False
)

# ==========================================
# 6. GRAFICO DE ESTADOS ACTIVOS
# ==========================================

grafico = resumen.set_index("grupo")[
    [f"{s}_activo_pct" for s in DIGITALES]
]

ax = grafico.plot(
    kind="bar",
    figsize=(12, 6),
    rot=25
)

ax.set_title(
    "Estados digitales por evento y periodo normal"
)
ax.set_ylabel("Porcentaje de registros con valor 1")
ax.set_xlabel("Grupo")
ax.set_ylim(0, 105)
ax.legend(title="Sensor")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    GRAFICOS / "estados_digitales_por_evento.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# 7. GRAFICO DE PRESION POR ESTADO
# ==========================================

datos_grafico = relaciones.copy()

datos_grafico["estado"] = (
    "DV=" + datos_grafico["DV_eletric"].astype(int).astype(str)
    + " COMP="
    + datos_grafico["COMP"].astype(int).astype(str)
)

datos_grafico["grupo"] = (
    datos_grafico["evento"] + "_"
    + datos_grafico["conjunto"]
)

tabla_grafico = datos_grafico.pivot_table(
    index="grupo",
    columns="estado",
    values="mean"
)

ax = tabla_grafico.plot(
    kind="bar",
    figsize=(13, 6),
    rot=25
)

ax.set_title(
    "DV_pressure medio segun estados de valvulas"
)
ax.set_ylabel("DV_pressure - media")
ax.set_xlabel("Evento y conjunto")
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    GRAFICOS / "presion_segun_estado.png",
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print("\n=== ARCHIVOS GENERADOS ===")
print("auditoria_estados_resumen.csv")
print("combinaciones_estados.csv")
print("relacion_estados_presion.csv")
print("estados_digitales_por_evento.png")
print("presion_segun_estado.png")

print("\nAuditoria terminada.")
print("No se modificaron modelos ni umbrales.")
