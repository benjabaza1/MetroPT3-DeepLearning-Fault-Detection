
from pathlib import Path
import pandas as pd

# ==================================================
# 1. CONFIGURACION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_CSV = (
    BASE_DIR / "01_datos" / "original"
    / "MetroPT3(AirCompressor).csv"
)

RUTA_RESULTADOS = BASE_DIR / "06_resultados" / "metricas"
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LEER LAS FECHAS
# ==================================================

print("\n=== ANALISIS TEMPORAL METROPT-3 ===")

df = pd.read_csv(
    RUTA_CSV,
    usecols=["timestamp"],
    parse_dates=["timestamp"]
)

print("Registros cargados:", len(df))

# ==================================================
# 3. VALIDAR LA SERIE TEMPORAL
# ==================================================

fechas_invalidas = df["timestamp"].isna().sum()
duplicados = df["timestamp"].duplicated().sum()

diferencias_originales = df["timestamp"].diff()
retrocesos = (diferencias_originales < pd.Timedelta(0)).sum()

print("\n=== INTEGRIDAD TEMPORAL ===")
print("Fechas invalidas:", fechas_invalidas)
print("Timestamps duplicados:", duplicados)
print("Retrocesos cronologicos:", retrocesos)

# Ordenamos para estudiar intervalos temporales
df = df.sort_values("timestamp").reset_index(drop=True)

df["intervalo_segundos"] = (
    df["timestamp"].diff().dt.total_seconds()
)

intervalos = df["intervalo_segundos"].dropna()

print("\n=== FRECUENCIA DE MEDICION ===")
print("Intervalo minimo (s):", intervalos.min())
print("Intervalo mediano (s):", intervalos.median())
print("Intervalo maximo (s):", intervalos.max())

print("\nIntervalos mas frecuentes:")
print(intervalos.value_counts().head(10))

print("\n=== INTERRUPCIONES ===")
print("Intervalos mayores a 60 s:", (intervalos > 60).sum())
print("Intervalos mayores a 1 hora:", (intervalos > 3600).sum())

# ==================================================
# 4. DISTRIBUCION MENSUAL
# ==================================================

df["mes"] = df["timestamp"].dt.to_period("M")

registros_mes = (
    df.groupby("mes")
    .size()
    .rename("registros")
    .reset_index()
)

registros_mes["mes"] = registros_mes["mes"].astype(str)

print("\n=== REGISTROS POR MES ===")
print(registros_mes.to_string(index=False))

registros_mes.to_csv(
    RUTA_RESULTADOS / "registros_por_mes.csv",
    index=False
)

# ==================================================
# 5. EVENTOS DE FALLA DOCUMENTADOS
# ==================================================

eventos = [
    ("Falla 1", "2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("Falla 2", "2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("Falla 3", "2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("Falla 4", "2020-07-15 14:30:00", "2020-07-15 19:00:00")
]

resumen_eventos = []

# Estos intervalos documentales tienen precision de minutos.
# Solo auditamos registros, sin crear etiquetas definitivas.
for nombre, inicio, fin in eventos:

    inicio = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin)

    mascara = (
        (df["timestamp"] >= inicio) &
        (df["timestamp"] < fin + pd.Timedelta(minutes=1))
    )

    registros = df.loc[mascara, "timestamp"]

    resumen_eventos.append({
        "evento": nombre,
        "inicio": inicio,
        "fin": fin,
        "registros": len(registros),
        "primera_medicion": (
            registros.min() if len(registros) else pd.NaT
        ),
        "ultima_medicion": (
            registros.max() if len(registros) else pd.NaT
        )
    })

resultado = pd.DataFrame(resumen_eventos)

print("\n=== EVENTOS DOCUMENTADOS ===")
print(resultado.to_string(index=False))

resultado.to_csv(
    RUTA_RESULTADOS / "auditoria_eventos_falla.csv",
    index=False
)

print("\n=== RESUMEN ===")
print(
    "Registros en intervalos documentados:",
    resultado["registros"].sum()
)

print("\nAnalisis temporal completado.")
