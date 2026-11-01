import os
import re
import glob
import joblib
import numpy as np

from data_generator import MAX_ZENITH_DEG, RATE_MAP_PATH, ZENITH_HIST_PATH

DATOS = "../Datos"
SALIDA = "datos_reales"  # mapas experimentales para la inferencia
PATRON = "voxel_data_n_*_nhits_1_disMax_4cm_poca_angle_15.0.npz"
ARCHIVOS_POR_DIA = 144  # mismo criterio que anal_voxel_poca_plomo.ipynb

os.makedirs(SALIDA, exist_ok=True)
os.makedirs(os.path.dirname(RATE_MAP_PATH), exist_ok=True)
conteos_totales, dias_totales = 0.0, 0.0

for ruta in sorted(glob.glob(os.path.join(DATOS, "*", PATRON))):
    muestra = os.path.basename(os.path.dirname(ruta))
    if muestra == "poca":  # copias de plomo_bloques y Mallado_arena
        continue
    n_archivos = int(re.search(r"voxel_data_n_(\d+)_", ruta).group(1))
    dias = round(n_archivos / ARCHIVOS_POR_DIA, 1)

    data = np.load(ruta)  # no se toca voxel_theta (array de objetos)
    media_theta2 = np.nan_to_num(data["grid_score"])
    conteos = data["grid_counts"]

    # Varianza por columna: Σθ² / ΣN (independiente de los días)
    n_columna = conteos.sum(axis=2)
    theta2_columna = (media_theta2 * conteos).sum(axis=2)
    var_columna = np.where(n_columna > 0, theta2_columna / np.maximum(n_columna, 1), 0.0)

    nombre = f"{muestra}_{dias}d"
    np.save(os.path.join(SALIDA, f"{nombre}_var_xy.npy"), var_columna.astype(np.float32))
    np.save(os.path.join(SALIDA, f"{nombre}_N_xy.npy"), n_columna.astype(np.float32))
    print(f"{nombre}: mediana var {np.median(var_columna[n_columna > 0]):.2e}")

    conteos_totales = conteos_totales + n_columna
    dias_totales += dias

# Muones por día por píxel: se usa en el generador (punto 2)
np.save(RATE_MAP_PATH, (conteos_totales / dias_totales).astype(np.float32))

# ==============================================================================
# Distribución cenital medida (generación del cielo x aceptancia del hodoscopio)
# ==============================================================================
MUESTRAS_TRAZAS = ["Grosores_mortero_20d", "Mallado_arena_20d", "Mallado_mortero_20d"]
Z_MATRICES_ENTRADA = (0.0, 40.0)  # cm, plane_z_map de poca_voxel.ipynb (P0 -> P1)

bordes = np.linspace(0.0, MAX_ZENITH_DEG, 61)
histograma = np.zeros(len(bordes) - 1)
for muestra in MUESTRAS_TRAZAS:
    # El archivo más grande de cada muestra (la toma más larga)
    ruta = max(glob.glob(os.path.join(DATOS, muestra, "track_lg_*_n_*.pkl")), key=os.path.getsize)
    trazas = joblib.load(ruta)
    p0 = np.asarray(trazas[0][0])[:, :2]
    p1 = np.asarray(trazas[1][0])[:, :2]
    desplazamiento = np.linalg.norm(p1 - p0, axis=1)
    theta = np.degrees(np.arctan2(desplazamiento, Z_MATRICES_ENTRADA[1] - Z_MATRICES_ENTRADA[0]))
    histograma += np.histogram(theta, bordes)[0]
    print(f"{os.path.basename(ruta)}: {len(theta)} trazas, {100 * np.mean(theta <= MAX_ZENITH_DEG):.0f}% con theta <= {MAX_ZENITH_DEG}°")

np.savez(ZENITH_HIST_PATH, bordes=bordes, prob=histograma / histograma.sum())
