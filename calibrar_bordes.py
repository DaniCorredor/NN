import os
import re
import glob
import numpy as np
from scipy.optimize import least_squares

from data_generator import BANCO_RUIDO_PATH, EXCESO_BORDES_PATH, N_MAX_BANCO, media_recortada
from preparar_recortada import DATOS, PATRON, columnas

# Calibra el exceso de varianza que el DETECTOR agrega cerca de sus bordes (coordenadas fijas del
# detector). Se mide sin muestra (hodoscopio vacío, 16 h) y en el aire de las tomas de plomo y
# Grosores; el Mallado no se usa (queda para comprobar). Es igual en aire y en concreto (se suma) y
# tiene tres partes (perfiles medios de 1 cm, con todos los muones de cada franja juntos):
#   y = 0 : una BANDA centrada en y ~ 10 cm (debajo de y ~ 7 cm no hay exceso)
#   y = 56: un ESCALÓN que empieza en y ~ 48 cm
#   x = 0 : una subida exponencial; en x = 56 no aparece
# Las formas de banda y escalón son las que la red confundía con barras paralelas a x en V15.
#
# Modelo (escala de la media, antes del ruido del estimador):
#   g(x, y) = aB exp(-(y - yB)^2 / 2 wB^2) + aT / (1 + exp(-(y - yT) / wT)) + aL exp(-x / lL) + aR exp(-(56 - x) / lR)

LADO_CM = 56.0
CENTRO_CM = (20, 36)
VACIO = "datos_reales/toma_libre_voxelizacion_exp.npz"

banco = np.load(BANCO_RUIDO_PATH)
E_AIRE = banco["aire"][N_MAX_BANCO].mean()      # media recortada / media con muchos muones (aire)

def voxel_data(muestra):
    # Toma más larga de la muestra (mismos cortes que los mapas originales)
    rutas = glob.glob(os.path.join(DATOS, muestra, PATRON))
    return max(rutas, key=lambda r: int(re.search(r"_n_(\d+)_", r).group(1)))

def perfil(ruta, franjas, eje):
    # Exceso de la media recortada por franja de 1 cm (escala de la media) respecto al centro, y su error.
    # eje = 1: perfil en y con las columnas x de 'franjas'; eje = 0: perfil en x con las filas y de 'franjas'
    cols, _ = columnas(np.load(ruta, allow_pickle=True)["voxel_theta"])
    def juntar(a, b):
        partes = []
        for f0, f1 in franjas:
            bloque = cols[int(f0 * 2):int(f1 * 2), int(a * 2):int(b * 2)] if eje == 1 else \
                     cols[int(a * 2):int(b * 2), int(f0 * 2):int(f1 * 2)]
            partes += [c for c in bloque.ravel() if len(c)]
        t2 = np.concatenate(partes)
        return media_recortada(t2), len(t2)
    centro, _ = juntar(*CENTRO_CM)
    pos = np.arange(1, 56) + 0.5
    valores = [juntar(p - 0.5, p + 0.5) for p in pos]
    exceso = np.array([v for v, _ in valores]) - centro
    error = centro * np.sqrt(1.5 / np.array([n for _, n in valores]))
    return pos, exceso / E_AIRE, error / E_AIRE

def g_y(y, aB, yB, wB, aT, yT, wT):
    return aB * np.exp(-0.5 * ((y - yB) / wB) ** 2) + aT / (1 + np.exp(-(y - yT) / wT))

def g_x(x, aL, lL, aR, lR):
    return aL * np.exp(-x / lL) + aR * np.exp(-(LADO_CM - x) / lR)

if __name__ == "__main__":
    plomo, grosores = voxel_data("plomo_bloques_84h"), voxel_data("Grosores_mortero_20d")
    fuentes_y = {   # aire en todo el rango de y
        "empty hodoscope": perfil(VACIO, [(14, 42)], 1),
        "lead (air)": perfil(plomo, [(21.5, 27), (39.5, 43)], 1),
        "thicknesses (air)": perfil(grosores, [(46, 55)], 1),
    }
    fuentes_x = {   # aire en todo el rango de x
        "empty hodoscope": perfil(VACIO, [(14, 44)], 0),
        "lead (air)": perfil(plomo, [(15, 19.5), (42, 46)], 0),
    }

    def ajustar(fuentes, modelo, p0, lo, hi):
        y = np.concatenate([f[0] for f in fuentes.values()])
        v = np.concatenate([f[1] for f in fuentes.values()])
        e = np.concatenate([f[2] for f in fuentes.values()])
        r = least_squares(lambda p: (modelo(y, *p) - v) / e, p0, bounds=(lo, hi), loss="soft_l1")
        err = np.sqrt(np.diag(np.linalg.pinv(r.jac.T @ r.jac)))
        # Amplitud relativa de cada fuente con la forma ajustada (cambia entre tomas)
        escalas = {k: float(np.sum(f[1] * modelo(f[0], *r.x) / f[2] ** 2) / np.sum(modelo(f[0], *r.x) ** 2 / f[2] ** 2))
                   for k, f in fuentes.items()}
        return r.x, err, escalas

    py, ey, sy = ajustar(fuentes_y, g_y, [5e-4, 10, 1.5, 6e-4, 49, 1], [0, 7, 0.5, 0, 44, 0.3], [5e-3, 14, 4, 5e-3, 54, 4])
    px, ex, sx = ajustar(fuentes_x, g_x, [1e-3, 4, 1e-4, 4], [0, 1, 0, 1], [5e-3, 15, 5e-3, 15])

    nombres_y = ["aB", "yB", "wB", "aT", "yT", "wT"]
    nombres_x = ["aL", "lL", "aR", "lR"]
    print("Exceso del detector (escala de la media; amplitudes en rad^2, posiciones en cm):")
    for n, p, e in list(zip(nombres_y, py, ey)) + list(zip(nombres_x, px, ex)):
        print(f"  {n:3s} = {p:.3g} ± {e:.2g}")
    print("Amplitud relativa por fuente  en y:", {k: round(v, 2) for k, v in sy.items()},
          " en x:", {k: round(v, 2) for k, v in sx.items()})
    np.savez(EXCESO_BORDES_PATH, **dict(zip(nombres_y, py)), **dict(zip(nombres_x, px)))
    print(f"Guardado en {EXCESO_BORDES_PATH}")

    # Comprobación con el Mallado (no usado en el ajuste): franjas sin barras a lo largo de y.
    # En concreto el exceso pasa por el estimador como ~0.5 (banco de concreto a N grande)
    E_CONC = banco["concreto"][N_MAX_BANCO].mean()
    for muestra, franjas in [("Mallado_mortero_20d", [(27, 31), (36, 41)]), ("Mallado_arena_20d", [(36, 42), (46, 52)])]:
        pos, exceso, _ = perfil(voxel_data(muestra), franjas, 1)
        filas = [f"y{p:.0f}: {E_AIRE * exceso[i] * 1e3:+.2f}/{E_CONC * g_y(p, *py) * 1e3:+.2f}"
                 for i, p in enumerate(pos) if p in (8.5, 9.5, 10.5, 11.5, 48.5, 50.5, 52.5)]
        print(f"  {muestra} exceso medido / predicho (media recortada, 1e-3 rad^2): " + "  ".join(filas))
