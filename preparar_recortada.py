import os
import re
import glob
import numpy as np
from scipy.ndimage import gaussian_filter

from data_generator import ACEPTANCIA_PATH, RATE_MAP_PATH, media_recortada

# Preprocesamiento V14 (corre en el PC, donde están los voxel_data con los ángulos por muón):
#   1. Media RECORTADA de theta^2 por columna: se descarta el FRACCION_RECORTE más alto de cada
#      columna. En el Mallado real duplica el SNR de las barras frente a la media simple.
#   2. Intervalos de días independientes: las tomas acumuladas son prefijos unas de otras
#      (los muones de la toma corta son los primeros de la larga), así que cada intervalo se
#      arma con los muones nuevos de cada vóxel.
#   3. Aceptancia del hodoscopio desde la toma libre simulada (suave, sin huella de muestras).
#   4. Muestras de theta^2 por muón (normalizadas a su región) para simular el ruido del estimador.

DATOS = "../Datos"
SALIDA = "datos_reales"
PATRON = "voxel_data_n_*_nhits_1_disMax_4cm_poca_angle_15.0.npz"
ARCHIVOS_POR_DIA = 144  # mismo criterio que anal_voxel_poca_plomo.ipynb
TOMA_LIBRE = os.path.join(SALIDA, "toma_libre_voxelizacion.npz")
MUESTRAS_THETA_PATH = "calibracion/theta2_por_muon.npz"
px = lambda c: int(round(c / 0.5))

def columnas(vt, desde=None):
    # theta^2 de cada columna; con 'desde' (largos por vóxel de la toma anterior) solo los muones nuevos
    cols = np.empty(vt.shape[:2], dtype=object)
    largos = np.zeros(vt.shape, dtype=np.int32)
    for i in range(vt.shape[0]):
        for j in range(vt.shape[1]):
            partes = []
            for k in range(vt.shape[2]):
                v = vt[i, j, k]
                if v is None:
                    continue
                n = len(v)
                largos[i, j, k] = n
                inicio = 0 if desde is None else desde[i, j, k]
                if n > inicio:
                    partes.append(np.asarray(v[inicio:], float) ** 2)
            cols[i, j] = np.concatenate(partes) if partes else np.zeros(0)
    return cols, largos

def mapas(cols):
    N = np.array([[len(c) for c in fila] for fila in cols], dtype=np.float32)
    media = np.array([[c.mean() if len(c) else 0.0 for c in fila] for fila in cols], dtype=np.float32)
    rec = np.array([[media_recortada(c) if len(c) else 0.0 for c in fila] for fila in cols], dtype=np.float32)
    return media, rec, N

def guardar(nombre, media, rec, N):
    np.save(os.path.join(SALIDA, f"{nombre}_var_xy.npy"), media)
    np.save(os.path.join(SALIDA, f"{nombre}_varrec_xy.npy"), rec)
    np.save(os.path.join(SALIDA, f"{nombre}_N_xy.npy"), N)

# Regiones homogéneas (cm) de donde se toman las muestras de theta^2 por muón
REGIONES_THETA = {
    "aire": ("plomo_bloques_84h", [((2, 8), (4, 52)), ((22, 28), (4, 16)), ((22, 28), (44, 52))]),
    "concreto": ("Grosores_mortero_20d", [((15, 39), (30, 35))]),
    "plomo": ("plomo_bloques_84h", [((12, 18), (25, 33)), ((31, 37), (22, 38))]),
}

if __name__ == "__main__":
    theta_por_clase = {}
    for carpeta in sorted(glob.glob(os.path.join(DATOS, "*"))):
        muestra = os.path.basename(carpeta)
        rutas = sorted(glob.glob(os.path.join(carpeta, PATRON)), key=lambda r: int(re.search(r"_n_(\d+)_", r).group(1)))
        if muestra == "poca" or not rutas:  # 'poca' son copias de plomo_bloques y Mallado_arena
            continue
        anterior, dias_anterior = None, 0.0
        for ruta in rutas:
            dias = round(int(re.search(r"_n_(\d+)_", ruta).group(1)) / ARCHIVOS_POR_DIA, 1)
            vt = np.load(ruta, allow_pickle=True)["voxel_theta"]
            cols, largos = columnas(vt)
            media, rec, N = mapas(cols)
            guardar(f"{muestra}_{dias}d", media, rec, N)
            print(f"{muestra}_{dias}d: N total {int(N.sum())}, mediana media {np.median(media[N > 0]):.2e}, "
                  f"recortada {np.median(rec[N > 0]):.2e}", flush=True)
            if anterior is not None:
                cols_int, _ = columnas(vt, desde=anterior)
                guardar(f"{muestra}_int_{dias_anterior}-{dias}d", *mapas(cols_int))
            elif len(rutas) > 1:
                guardar(f"{muestra}_int_0-{dias}d", media, rec, N)
            # theta^2 por muón de las regiones homogéneas (toma más larga de cada muestra)
            if ruta == rutas[-1]:
                for clase, (m, regiones) in REGIONES_THETA.items():
                    if m != muestra:
                        continue
                    t2 = np.concatenate([np.concatenate(list(cols[px(x[0]):px(x[1]), px(y[0]):px(y[1])].ravel()))
                                         for x, y in regiones])
                    theta_por_clase[clase] = (t2 / t2.mean()).astype(np.float32)
            anterior, dias_anterior = largos, dias

    np.savez(MUESTRAS_THETA_PATH, **theta_por_clase)
    print({k: len(v) for k, v in theta_por_clase.items()})

    # Aceptancia: toma libre simulada, suavizada (1 px) y normalizada a 1 en el centro;
    # la escala absoluta (muones/día) sale del mapa de tasa medido en el centro
    libre = np.load(TOMA_LIBRE, allow_pickle=True)["grid_counts"].sum(axis=2).astype(np.float32)
    aceptancia = gaussian_filter(libre, 1.0)
    aceptancia /= aceptancia[40:72, 40:72].mean()
    tasa_centro = np.load(RATE_MAP_PATH)[40:72, 40:72].mean()
    np.save(ACEPTANCIA_PATH, (aceptancia * tasa_centro).astype(np.float32))
    print(f"Aceptancia guardada en {ACEPTANCIA_PATH} (tasa central {tasa_centro:.2f} muones/día/píxel)")
