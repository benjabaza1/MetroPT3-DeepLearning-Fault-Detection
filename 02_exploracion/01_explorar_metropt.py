
from pathlib import Path
import pandas as pd

# ==================================================
# 1. CONFIGURACION DEL PROYECTO
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_CSV = (
    BASE_DIR
    / "01_datos"
    / "original"
    / "MetroPT3(AirCompressor).csv"
)

RUTA_RESULTADOS = BASE_DIR / "06_resultados" / "metricas"
RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

if not RUTA_CSV.exists():
    raise FileNotFoundError(f"No se encontro: {RUTA_CSV}")

# ==================================================
# 2. DEFINICION DE SENSORES
# ==================================================

ANALOGICOS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current"
]

DIGITALES = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses"
]

SENSORES = ANALOGICOS + DIGITALES

# ==================================================
# 3. LEER ESTRUCTURA Y PRIMEROS REGISTROS
# ==================================================

print("\n=== EXPLORACION METROPT-3 ===")

muestra = pd.read_csv(RUTA_CSV, nrows=5)

print("\n=== PRIMEROS REGISTROS ===")
print(muestra.to_string(index=False))

print("\n=== COLUMNAS ===")
print(muestra.columns.tolist())

print("\n=== TIPOS DE DATOS INICIALES ===")
print(muestra.dtypes)

faltantes = [
    columna for columna in ["timestamp"] + SENSORES
    if columna not in muestra.columns
]

if faltantes:
    raise ValueError(f"Columnas no encontradas: {faltantes}")

# ==================================================
# 4. ANALISIS POR BLOQUES
# ==================================================

total_registros = 0
fechas_invalidas = 0
nulos = pd.Series(0, index=muestra.columns, dtype="int64")

fecha_minima = None
fecha_maxima = None

minimos = pd.Series(float("inf"), index=SENSORES)
maximos = pd.Series(float("-inf"), index=SENSORES)

estados_digitales = {columna: set() for columna in DIGITALES}

for numero, bloque in enumerate(
    pd.read_csv(RUTA_CSV, chunksize=100_000),
    start=1
):
    total_registros += len(bloque)

    nulos = nulos.add(bloque.isna().sum(), fill_value=0)

    fechas = pd.to_datetime(
        bloque["timestamp"],
        errors="coerce"
    )

    fechas_invalidas += fechas.isna().sum()

    if fechas.notna().any():
        minimo = fechas.min()
        maximo = fechas.max()

        if fecha_minima is None or minimo < fecha_minima:
            fecha_minima = minimo

        if fecha_maxima is None or maximo > fecha_maxima:
            fecha_maxima = maximo

    for sensor in SENSORES:
        valores = pd.to_numeric(
            bloque[sensor],
            errors="coerce"
        )

        if valores.notna().any():
            minimos[sensor] = min(minimos[sensor], valores.min())
            maximos[sensor] = max(maximos[sensor], valores.max())

        if sensor in DIGITALES:
            estados_digitales[sensor].update(
                valores.dropna().unique().tolist()
            )

    print(
        f"Bloque {numero} procesado | "
        f"Registros acumulados: {total_registros:,}"
    )

# ==================================================
# 5. RESULTADOS GENERALES
# ==================================================

print("\n=== DIMENSIONES ===")
print("Filas:", total_registros)
print("Columnas:", len(muestra.columns))
print("Sensores:", len(SENSORES))

print("\n=== PERIODO DE MEDICIONES ===")
print("Primera fecha:", fecha_minima)
print("Ultima fecha:", fecha_maxima)
print("Fechas invalidas:", fechas_invalidas)

print("\n=== VALORES NULOS ===")
print(nulos.astype(int).to_string())

# ==================================================
# 6. RANGOS DE SENSORES
# ==================================================

resumen = pd.DataFrame({
    "Sensor": SENSORES,
    "Tipo": [
        "Analogico" if s in ANALOGICOS else "Digital"
        for s in SENSORES
    ],
    "Minimo": [minimos[s] for s in SENSORES],
    "Maximo": [maximos[s] for s in SENSORES]
})

print("\n=== RANGOS DE SENSORES ===")
print(resumen.to_string(index=False))

print("\n=== ESTADOS DE SENSORES DIGITALES ===")
for sensor, estados in estados_digitales.items():
    print(f"{sensor}: {sorted(estados)}")

# ==================================================
# 7. GUARDAR RESUMEN
# ==================================================

resumen.to_csv(
    RUTA_RESULTADOS / "resumen_sensores.csv",
    index=False
)

print("\n=== VALIDACION ===")

print("Registros esperados: 1516948")
print("Registros obtenidos:", total_registros)

if total_registros == 1_516_948:
    print("Cantidad de registros correcta.")
else:
    print("ATENCION: revisar cantidad de registros.")

print("\nExploracion inicial finalizada.")
