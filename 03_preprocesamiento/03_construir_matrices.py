
from pathlib import Path
import numpy as np
import pandas as pd

# ==================================================
# 1. CONFIGURACION
# ==================================================

BASE = Path(__file__).resolve().parent.parent
DATOS = BASE / "01_datos" / "procesados"

RUTA_SENSORES = DATOS / "sensores_normalizados.npy"
RUTA_CATALOGO = DATOS / "catalogo_ventanas.csv"

TAMANO_VENTANA = 60
NUM_SENSORES = 15

print("\n=== CONSTRUCCION DE MATRICES ===")

# ==================================================
# 2. CARGAR INFORMACION
# ==================================================

# mmap_mode evita cargar de inmediato toda
# la matriz original en memoria RAM.
sensores = np.load(
    RUTA_SENSORES,
    mmap_mode="r"
)

catalogo = pd.read_csv(
    RUTA_CATALOGO,
    parse_dates=["fecha_inicio", "fecha_fin"]
)

print("Matriz original:", sensores.shape)
print("Ventanas catalogadas:", len(catalogo))

if sensores.shape[1] != NUM_SENSORES:
    raise ValueError("Cantidad de sensores incorrecta")

# ==================================================
# 3. CONSTRUIR CADA CONJUNTO
# ==================================================

resumen = []

for conjunto in ["train", "validation", "test"]:

    print(f"\nProcesando conjunto: {conjunto}")

    subset = (
        catalogo[catalogo["conjunto"] == conjunto]
        .sort_values("fecha_inicio")
        .reset_index(drop=True)
    )

    cantidad = len(subset)

    X = np.empty(
        (cantidad, TAMANO_VENTANA, NUM_SENSORES),
        dtype=np.float32
    )

    y = subset["etiqueta"].to_numpy(
        dtype=np.int8
    )

    # ==================================================
    # 4. RECUPERAR VENTANAS
    # ==================================================

    for i, fila in enumerate(subset.itertuples()):

        inicio = int(fila.fila_inicio)
        fin = int(fila.fila_fin) + 1

        if fin - inicio != TAMANO_VENTANA:
            raise ValueError(
                f"Ventana incorrecta: {i}"
            )

        X[i] = sensores[inicio:fin]

    # ==================================================
    # 5. VALIDACIONES
    # ==================================================

    if not np.isfinite(X).all():
        raise ValueError(
            f"Valores no finitos en {conjunto}"
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Etiquetas invalidas en {conjunto}"
        )

    assert len(X) == len(y)

    # ==================================================
    # 6. GUARDAR MATRICES
    # ==================================================

    np.save(DATOS / f"X_{conjunto}.npy", X)
    np.save(DATOS / f"y_{conjunto}.npy", y)

    normales = int(np.sum(y == 0))
    fallas = int(np.sum(y == 1))

    porcentaje_falla = (
        fallas / cantidad * 100
        if cantidad else 0
    )

    print("Forma X:", X.shape)
    print("Forma y:", y.shape)
    print("Ventanas normales:", normales)
    print("Ventanas de falla:", fallas)
    print(
        "Porcentaje de falla:",
        round(porcentaje_falla, 2),
        "%"
    )

    resumen.append({
        "conjunto": conjunto,
        "ventanas": cantidad,
        "normales": normales,
        "fallas": fallas,
        "porcentaje_falla": porcentaje_falla
    })

# ==================================================
# 7. RESUMEN FINAL
# ==================================================

resultado = pd.DataFrame(resumen)

resultado.to_csv(
    DATOS / "resumen_matrices.csv",
    index=False
)

print("\n=== RESUMEN FINAL ===")
print(resultado.to_string(index=False))

print("\nMatrices generadas correctamente.")
