import numpy as np
from scipy.optimize import least_squares

from data_generator import BANCO_RUIDO_PATH, EXCESO_BORDES_PATH, N_MAX_BANCO, VOXEL_CM

# Calibra el exceso de varianza cerca de los bordes del detector (coordenadas del detector, fijas).
# Medido sin muestra (hodoscopio vacío, 16 h) y en zonas homogéneas de plomo y Grosores:
# se SUMA a la varianza (igual en aire y en concreto), sube en los bordes y = 0, y = 56 y x = 0,
# y no aparece en x = 56. Ningún filtro individual (DOCA, ángulo de incidencia) lo quita.
# El Mallado no se usa aquí: queda para comprobar.
#
# Modelo (escala de la media, antes del ruido): varianza + sum_b a_b * exp(-d_b / lambda_b),
# d_b = distancia al borde b. Lo medido es media recortada: E_banco(clase, N) * (base + exceso).

DATOS = "datos_reales"
BORDES = ("x=0", "x=56", "y=0", "y=56")
LADO_CM = 56.0
c = (np.arange(112) + 0.5) * VOXEL_CM
X, Y = np.meshgrid(c, c, indexing="ij")
DISTANCIAS = [X, LADO_CM - X, Y, LADO_CM - Y]

banco = np.load(BANCO_RUIDO_PATH)
E_BANCO = {k: banco[k].mean(axis=1) for k in ("aire", "concreto")}
S_BANCO = {k: banco[k].std(axis=1) for k in ("aire", "concreto")}

def exceso(a, lam):
    return sum(a[b] * np.exp(-DISTANCIAS[b] / lam[b]) for b in range(4))

def region(muestra, clase, mascara):
    v = np.load(f"{DATOS}/{muestra}_varrec_xy.npy")
    N = np.load(f"{DATOS}/{muestra}_N_xy.npy")
    m = mascara & (N > 0) & (X > 1.5) & (Y > 1.5)
    n = np.clip(N[m], 1, N_MAX_BANCO).astype(int)
    E = E_BANCO[clase][n]
    # Escala fija del error (si dependiera de los parámetros, el ajuste inflaría la base)
    nivel = np.median(v[m] / E)
    return dict(nombre=f"{muestra} ({clase})", v=v[m], E=E, sigma=S_BANCO[clase][n] * nivel, m=m)

plomo = np.zeros_like(X, dtype=bool)
for (x0, x1), (y0, y1) in [((11, 19), (24, 34)), ((30, 38), (20, 40)), ((44, 54), (20, 40))]:
    plomo |= (X > x0 - 4) & (X < x1 + 4) & (Y > y0 - 4) & (Y < y1 + 4)

REGIONES = [
    region("vacio_exp_0.7d", "aire", np.ones_like(X, dtype=bool)),
    region("plomo_bloques_84h_3.5d", "aire", ~plomo),
    region("Grosores_mortero_20d_20.6d", "aire", (X > 46) | (X < 10)),
    region("Grosores_mortero_20d_20.6d", "concreto", (X > 35) & (X < 41) & (Y > 9)),
]

def residuos(p):
    a, lam, bases = p[:4], p[4:8], p[8:]
    D = exceso(a, lam)
    out = []
    for r, base in zip(REGIONES, bases):
        pred = r["E"] * (base + D[r["m"]])
        out.append((r["v"] - pred) / r["sigma"])
    return np.concatenate(out)

if __name__ == "__main__":
    p0 = [1e-3] * 4 + [5.0] * 4 + [1.7e-3, 1.7e-3, 1.4e-3, 4e-3]
    lo = [0] * 4 + [1.0] * 4 + [0] * 4
    hi = [2e-2] * 4 + [12.0] * 4 + [2e-2] * 4
    ajuste = least_squares(residuos, p0, bounds=(lo, hi), loss="soft_l1")
    a, lam, bases = ajuste.x[:4], ajuste.x[4:8], ajuste.x[8:]
    J = ajuste.jac
    err = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)) * np.mean(ajuste.fun ** 2))

    print("Exceso en el borde (escala de la media, rad^2) y longitud de caída:")
    for b in range(4):
        print(f"  {BORDES[b]:5s}  a = {a[b]:.2e} ± {err[b]:.1e}   lambda = {lam[b]:4.1f} ± {err[4 + b]:.1f} cm")
    for r, base in zip(REGIONES, bases):
        print(f"  base {r['nombre']:42s} {base:.2e}")
    np.savez(EXCESO_BORDES_PATH, a=a, lam=lam, err_a=err[:4], err_lam=err[4:8])
    print(f"Guardado en {EXCESO_BORDES_PATH}")

    # Comprobación con el Mallado (no usado en el ajuste): franja central x 30-42, bordes en y
    D = exceso(a, lam)
    for muestra in ("Mallado_mortero_20d_18.9d", "Mallado_arena_20d_21.6d"):
        v = np.load(f"{DATOS}/{muestra}_varrec_xy.npy")
        perfil = v[60:84].mean(axis=0)
        base = perfil[50:70].mean()
        filas = []
        for y in (44, 48, 50, 52):
            i = int(y / VOXEL_CM)
            medido = perfil[i:i + 4].mean() / base - 1
            # el exceso pasa por el estimador recortado como el concreto (~0.49 en N grande)
            predicho = E_BANCO["concreto"][100] * D[60:84, i:i + 4].mean() / base
            filas.append(f"y{y}: {100 * medido:+.0f}% / {100 * predicho:+.0f}%")
        print(f"  {muestra} medido / predicho:  " + "   ".join(filas))
