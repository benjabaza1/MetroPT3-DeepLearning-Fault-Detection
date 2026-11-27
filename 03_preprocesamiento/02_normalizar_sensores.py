
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

# ==============================================
# 1. CONFIGURACION
# ==============================================

BASE = Path(__file__).resolve().parent.parent

ORIGEN = (
    BASE / "01_datos" / "original"
    / "MetroPT3(AirCompressor).csv"
)

CATALOGO = (
    BASE / "01_datos" / "procesados"
    / "catalogo_ventanas.csv"
)

DESTINO = BASE / "01_datos" / "procesados"
DESTINO.mkdir(parents=True, exist_ok=True)

ANALOGICOS = [
    "TP2", "TP3", "H1", "DV_pressure",
    "Reservoirs", "Oil_temperature",
    "Motor_current"
]

DIGITALES = [
    "COMP", "DV_eletric", "Towers",
    "MPG", "LPS", "Pressure_switch",
    "Oil_level", "Caudal_impulses"
]

SENSORES = ANALOGICOS + DIGITALES

# ==============================================
# 2. LEER DATOS Y CATALOGO
# ==============================================

print("\n=== NORMALIZACION METROPT-3 ===")

df = pd.read_csv(
    ORIGEN,
    usecols=SENSORES,
    dtype={s: "float32" for s in SENSORES}
)

catalogo = pd.read_csv(CATALOGO)

print("Registros:", len(df))
print("Ventanas:", len(catalogo))

if df[SENSORES].isna().any().any():
    raise ValueError("Hay valores nulos")

# ==============================================
# 3. IDENTIFICAR FILAS DE ENTRENAMIENTO
# ==============================================

train = catalogo[
    catalogo["conjunto"] == "train"
]

indices = np.concatenate([
    np.arange(
        int(fila.fila_inicio),
        int(fila.fila_fin) + 1
    )
    for fila in train.itertuples()
])

if len(np.unique(indices)) != len(indices):
    raise ValueError("Filas de entrenamiento repetidas")

if indices.min() < 0 or indices.max() >= len(df):
    raise ValueError("Indices fuera de rango")

print("Observaciones de entrenamiento:", len(indices))

# ==============================================
# 4. AJUSTAR ESCALADOR SOLO CON TRAIN
# ==============================================

escalador = StandardScaler()

escalador.fit(
    df.iloc[indices][ANALOGICOS]
)

# ==============================================
# 5. TRANSFORMAR SENSORES ANALOGICOS
# ==============================================

datos_escalados = escalador.transform(
    df[ANALOGICOS]
).astype("float32")

# Guardamos los 15 sensores en el orden definido.
# Los digitales se mantienen sin escalar.
matriz = np.column_stack([
    datos_escalados,
    df[DIGITALES].to_numpy(dtype="float32")
])

# ==============================================
# 6. VERIFICACIONES
# ==============================================

if matriz.shape != (len(df), 15):
    raise ValueError("Dimensiones inesperadas")

if not np.isfinite(matriz).all():
    raise ValueError("Existen valores no finitos")

for i, sensor in enumerate(DIGITALES, start=7):
    if not np.isin(matriz[:, i], [0, 1]).all():
        raise ValueError(f"Estado digital invalido: {sensor}")

# ==============================================
# 7. GUARDAR RESULTADOS
# ==============================================

np.save(
    DESTINO / "sensores_normalizados.npy",
    matriz
)

joblib.dump(
    escalador,
    DESTINO / "standard_scaler.joblib"
)

pd.DataFrame({
    "sensor": ANALOGICOS,
    "media_train": escalador.mean_,
    "desviacion_train": escalador.scale_
}).to_csv(
    DESTINO / "estadisticas_normalizacion.csv",
    index=False
)

# Guardamos el orden necesario para reconstruir
# correctamente las entradas de las redes.
pd.DataFrame({
    "posicion": range(len(SENSORES)),
    "sensor": SENSORES
}).to_csv(
    DESTINO / "orden_sensores.csv",
    index=False
)

print("\n=== VALIDACION ===")
print("Forma de la matriz:", matriz.shape)
print("Tipo de datos:", matriz.dtype)

print("\n=== ESTADISTICAS TRAIN ===")
print(pd.DataFrame({
    "sensor": ANALOGICOS,
    "media": escalador.mean_,
    "desviacion": escalador.scale_
}).to_string(index=False))

print("\nArchivos generados correctamente.")
print("Normalizacion finalizada.")
