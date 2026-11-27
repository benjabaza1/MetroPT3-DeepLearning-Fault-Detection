
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# 1. CONFIGURACION
# ==================================================

BASE = Path(__file__).resolve().parent.parent

ORIGEN = (
    BASE / "01_datos" / "original"
    / "MetroPT3(AirCompressor).csv"
)

DESTINO = BASE / "01_datos" / "procesados"
DESTINO.mkdir(parents=True, exist_ok=True)

TAMANO_VENTANA = 60
MAX_INTERVALO_SEG = 30

# ==================================================
# 2. INTERVALOS DOCUMENTADOS
# ==================================================

EVENTOS = [
    ("Falla_1", "2020-04-18 00:00", "2020-04-18 23:59"),
    ("Falla_2", "2020-05-29 23:30", "2020-05-30 06:00"),
    ("Falla_3", "2020-06-05 10:00", "2020-06-07 14:30"),
    ("Falla_4", "2020-07-15 14:30", "2020-07-15 19:00"),
]

# ==================================================
# 3. LEER FECHAS
# ==================================================

print("\n=== PREPARACION DE ETIQUETAS ===")

df = pd.read_csv(
    ORIGEN,
    usecols=["timestamp"],
    parse_dates=["timestamp"]
)

if df["timestamp"].isna().any():
    raise ValueError("Existen fechas invalidas")

if df["timestamp"].duplicated().any():
    raise ValueError("Existen fechas duplicadas")

if not df["timestamp"].is_monotonic_increasing:
    raise ValueError("Los registros no estan ordenados")

# Posicion original de cada observacion del CSV
df["fila_csv"] = np.arange(len(df), dtype=np.int64)

# ==================================================
# 4. GENERAR ETIQUETAS
# ==================================================

df["falla"] = np.zeros(len(df), dtype=np.int8)
df["evento"] = "Normal"

for nombre, inicio, fin in EVENTOS:
    inicio = pd.Timestamp(inicio)
    fin_exclusivo = pd.Timestamp(fin) + pd.Timedelta(minutes=1)

    mascara = (
        (df["timestamp"] >= inicio)
        & (df["timestamp"] < fin_exclusivo)
    )

    df.loc[mascara, "falla"] = 1
    df.loc[mascara, "evento"] = nombre

print("\n=== DISTRIBUCION ===")
print(df["falla"].value_counts())
print(df["falla"].value_counts(normalize=True).mul(100))

print("\n=== REGISTROS POR EVENTO ===")
print(df["evento"].value_counts())

# ==================================================
# 5. DIVISION CRONOLOGICA
# ==================================================

corte_validacion = pd.Timestamp("2020-06-01")
corte_prueba = pd.Timestamp("2020-07-01")

df["conjunto"] = np.select(
    [
        df["timestamp"] < corte_validacion,
        df["timestamp"] < corte_prueba
    ],
    ["train", "validation"],
    default="test"
)

# Margen de seguridad de una hora en cada corte.
# No utilizaremos esas observaciones para construir ventanas.
EMBARGO = pd.Timedelta(hours=1)

excluir = pd.Series(False, index=df.index)

for corte in [corte_validacion, corte_prueba]:
    excluir |= (
        (df["timestamp"] >= corte - EMBARGO)
        & (df["timestamp"] < corte + EMBARGO)
    )

df["usable"] = ~excluir

# ==================================================
# 6. SEGMENTOS CONTINUOS Y HOMOGENEOS
# ==================================================

intervalos = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
)

# Segmento nuevo si:
# - existe interrupcion temporal
# - cambia la etiqueta
# - cambia el conjunto
# - cambia el evento
# - aparece un registro excluido
nuevo_segmento = (
    intervalos.isna()
    | (intervalos > MAX_INTERVALO_SEG)
    | (intervalos <= 0)
    | df["falla"].ne(df["falla"].shift())
    | df["evento"].ne(df["evento"].shift())
    | df["conjunto"].ne(df["conjunto"].shift())
    | df["usable"].ne(df["usable"].shift())
)

df["segmento"] = nuevo_segmento.cumsum()

# ==================================================
# 7. CONSTRUIR CATALOGO DE VENTANAS
# ==================================================

ventanas = []

for _, grupo in df.groupby("segmento", sort=False):

    if not grupo["usable"].iloc[0]:
        continue

    if len(grupo) < TAMANO_VENTANA:
        continue

    posiciones = grupo["fila_csv"].to_numpy()

    # Ventanas sin superposicion dentro del segmento
    for i in range(0, len(grupo) - TAMANO_VENTANA + 1,
                   TAMANO_VENTANA):

        inicio_fila = int(posiciones[i])
        fin_fila = int(posiciones[i + TAMANO_VENTANA - 1])

        ventanas.append({
            "fila_inicio": inicio_fila,
            "fila_fin": fin_fila,
            "fecha_inicio": grupo["timestamp"].iloc[i],
            "fecha_fin": grupo["timestamp"].iloc[
                i + TAMANO_VENTANA - 1
            ],
            "etiqueta": int(grupo["falla"].iloc[i]),
            "evento": grupo["evento"].iloc[i],
            "conjunto": grupo["conjunto"].iloc[i],
        })

catalogo = pd.DataFrame(ventanas)

if catalogo.empty:
    raise ValueError("No se generaron ventanas")

# ==================================================
# 8. AUDITORIA
# ==================================================

print("\n=== VENTANAS POR CONJUNTO Y CLASE ===")

tabla = pd.crosstab(
    catalogo["conjunto"],
    catalogo["etiqueta"]
).reindex(
    index=["train", "validation", "test"],
    columns=[0, 1],
    fill_value=0
)

print(tabla)

print("\n=== VENTANAS DE FALLA POR EVENTO ===")
print(
    catalogo[catalogo["etiqueta"] == 1]
    .groupby(["conjunto", "evento"])
    .size()
)

# Verificar ventanas de 60 filas
assert (
    catalogo["fila_fin"] - catalogo["fila_inicio"] + 1
    == TAMANO_VENTANA
).all()

# Verificar que las ventanas sean disjuntas
ordenadas = catalogo.sort_values("fila_inicio")
assert (
    ordenadas["fila_inicio"].iloc[1:].to_numpy()
    > ordenadas["fila_fin"].iloc[:-1].to_numpy()
).all()

# Verificar que no existan ventanas de falla
# pertenecientes a eventos equivocados
esperados = {
    "Falla_1": "train",
    "Falla_2": "train",
    "Falla_3": "validation",
    "Falla_4": "test"
}

fallas = catalogo[catalogo["etiqueta"] == 1]

assert all(
    esperados[fila.evento] == fila.conjunto
    for fila in fallas.itertuples()
)

# ==================================================
# 9. GUARDAR RESULTADOS
# ==================================================

df[[
    "fila_csv", "timestamp", "falla",
    "evento", "conjunto", "usable", "segmento"
]].to_csv(
    DESTINO / "etiquetas_temporales.csv",
    index=False
)

catalogo.to_csv(
    DESTINO / "catalogo_ventanas.csv",
    index=False
)

print("\n=== ARCHIVOS GENERADOS ===")
print("etiquetas_temporales.csv")
print("catalogo_ventanas.csv")

print("\nPreparacion y auditoria finalizadas.")
