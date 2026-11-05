import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.optimize import least_squares
from scipy.stats import chi2, norm
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

import inference_unet_2_real as inf
from confiabilidad import exceso_detector, mapa_confiable
from data_generator import BANCO_RUIDO_PATH, EXCESO_BORDES_PATH, N_MAX_BANCO, PSF_SIGMA_RANGO_CM, VOXEL_CM

# NIVEL A: ajuste rígido de barras. La red propone y los datos deciden.
#   1. Candidatas: segmentos rectos (Hough) sobre el esqueleto del mapa de probabilidad de hierro,
#      más el eje principal de cada mancha compacta (barras anchas); se quitan las repetidas.
#   2. Cada candidata es un cilindro recto (extremos, radio) proyectado: cuerda de la sección
#      circular, con la PSF del detector. Se ajusta DIRECTO a la media recortada medida junto con
#      un fondo suave (B-spline de nudos cada ESPACIADO_FONDO_CM) y las tres formas del exceso del
#      detector (banda en y ~ 10, escalón en y ~ 49.5, subida en x = 0) con amplitud libre.
#      Los parámetros lineales (fondo, bandas, amplitud de cada barra) se resuelven exactos en cada
#      paso (proyección variable); least_squares mueve solo extremos y radios.
#   3. Significancia de cada barra: cuánto empeora el ajuste sin ella (Δχ², 6 parámetros).
#      Se quita la menos significativa mientras quede alguna bajo SIGMA_MINIMO.
#   4. Barras anchas: se prueba si son dos barras paralelas cercanas (se acepta si mejora a 3σ).
# Coordenadas del detector (cm); las figuras usan la orientación de la tesis (inf.TRANSPONER).

PSF_CM = float(np.mean(PSF_SIGMA_RANGO_CM))
ESPACIADO_FONDO_CM = 14.0   # más fino absorbe barras cortas y anchas (Grosores: 10 x 6 cm)
EROSION_BLOQUE_CM = 2.5
# Umbral de significancia: el calibrado en simulación (validar_ajuste_barras.py) si ya existe
SIGMA_MINIMO = 3.0
_UMBRAL_PATH = "resultados/V16/postprocessing/rigid_fit_validation/umbral_significancia.json"
if os.path.exists(_UMBRAL_PATH):
    with open(_UMBRAL_PATH) as _f:
        SIGMA_MINIMO = json.load(_f)["threshold_sigma"]
RADIO_CM = (0.4, 3.5)
SUPERMUESTREO = 3
MUONES_MINIMOS = 10

LADO = 112
CENTROS = (np.arange(LADO) + 0.5) * VOXEL_CM
XX, YY = np.meshgrid(CENTROS, CENTROS, indexing="ij")

_banco = np.load(BANCO_RUIDO_PATH)
_REL = _banco["concreto"].std(axis=1) / _banco["concreto"].mean(axis=1)   # dispersión relativa vs N

# ------------------------------------------------------------------------------------------
# Modelo
# ------------------------------------------------------------------------------------------
def base_bspline(n=LADO, espaciado=ESPACIADO_FONDO_CM, grado=3):
    lado = n * VOXEL_CM
    interiores = np.arange(0, lado + 1e-6, espaciado)
    nudos = np.concatenate([[interiores[0]] * grado, interiores, [interiores[-1]] * grado])
    nb = len(nudos) - grado - 1
    return np.stack([BSpline(nudos, np.eye(nb)[i], grado, extrapolate=False)(CENTROS) for i in range(nb)], axis=1)

_BX = np.nan_to_num(base_bspline())

def formas_detector():
    # Las tres partes del exceso del detector por separado (forma calibrada, amplitud libre en el ajuste)
    p = {k: float(v) for k, v in np.load(EXCESO_BORDES_PATH).items()}
    banda = np.exp(-0.5 * ((YY - p["yB"]) / p["wB"]) ** 2)
    escalon = 1 / (1 + np.exp(-(YY - p["yT"]) / p["wT"]))
    subida = np.exp(-XX / p["lL"])
    return [banda, escalon, subida]

_FORMAS = formas_detector()

def cuerda_barra(x1, y1, x2, y2, r):
    # Espesor de hierro proyectado (cm) de un cilindro recto entre (x1,y1) y (x2,y2) de radio r,
    # promediado dentro de cada píxel (supermuestreo); extremos planos
    lx, ly = x2 - x1, y2 - y1
    largo = max(np.hypot(lx, ly), 1e-6)
    ux, uy = lx / largo, ly / largo
    lo = np.floor((min(x1, x2) - r - 1) / VOXEL_CM).astype(int), np.floor((min(y1, y2) - r - 1) / VOXEL_CM).astype(int)
    hi = np.ceil((max(x1, x2) + r + 1) / VOXEL_CM).astype(int), np.ceil((max(y1, y2) + r + 1) / VOXEL_CM).astype(int)
    i0, j0 = max(lo[0], 0), max(lo[1], 0)
    i1, j1 = min(hi[0], LADO), min(hi[1], LADO)
    imagen = np.zeros((LADO, LADO))
    if i1 <= i0 or j1 <= j0:
        return imagen
    s = SUPERMUESTREO
    sub = (np.arange(s) + 0.5) / s * VOXEL_CM
    px = (np.arange(i0, i1)[:, None] * VOXEL_CM + sub[None, :]).ravel()
    py = (np.arange(j0, j1)[:, None] * VOXEL_CM + sub[None, :]).ravel()
    PX, PY = np.meshgrid(px, py, indexing="ij")
    t = (PX - x1) * ux + (PY - y1) * uy
    d = -(PX - x1) * uy + (PY - y1) * ux
    c = np.where((t >= 0) & (t <= largo), 2 * np.sqrt(np.clip(r ** 2 - d ** 2, 0, None)), 0.0)
    imagen[i0:i1, j0:j1] = c.reshape(i1 - i0, s, j1 - j0, s).mean(axis=(1, 3))
    return imagen

def imagen_barra(params):
    return gaussian_filter(cuerda_barra(*params), PSF_CM / VOXEL_CM)

class Ajuste:
    def __init__(self, medida, muones, region):
        self.region = region
        self.y = medida[region]
        nivel = gaussian_filter(medida, 4) / np.maximum(gaussian_filter(region.astype(float), 4), 1e-6)
        # Error de la media recortada por píxel: dispersión del banco real de concreto para su N
        # (más allá del banco, como 1/sqrt(N))
        n = np.clip(muones, 1, N_MAX_BANCO).astype(int)
        sigma = _REL[n] * nivel * np.sqrt(np.clip(N_MAX_BANCO / np.maximum(muones, 1), None, 1.0))
        self.w = 1.0 / sigma[region]
        # Columnas fijas: fondo B-spline (producto tensorial) + formas del detector
        fondo = np.einsum("ia,jb->ijab", _BX, _BX).reshape(LADO, LADO, -1)[region]
        usadas = np.abs(fondo).sum(axis=0) > 1e-3 * np.abs(fondo).sum(axis=0).max()
        self.fijas = np.concatenate([fondo[:, usadas], np.stack([f[region] for f in _FORMAS], axis=1)], axis=1)
        self.n_fijas = self.fijas.shape[1]

    def resolver(self, barras):
        # Parámetros lineales exactos para unas barras dadas -> (coeficientes, residuo pesado, imágenes)
        imagenes = [imagen_barra(b) for b in barras]
        A = np.concatenate([self.fijas, np.stack([im[self.region] for im in imagenes], axis=1)], axis=1) \
            if imagenes else self.fijas
        coef, *_ = np.linalg.lstsq(A * self.w[:, None], self.y * self.w, rcond=None)
        return coef, (self.y - A @ coef) * self.w, imagenes

    def chi2(self, barras):
        return float(np.sum(self.resolver(barras)[1] ** 2))

    def ajustar(self, barras):
        if not barras:
            return [], np.zeros((0, 5))
        p0 = np.concatenate(barras)
        lo = np.concatenate([[b[0] - 3, b[1] - 3, b[2] - 3, b[3] - 3, RADIO_CM[0]] for b in barras])
        hi = np.concatenate([[b[0] + 3, b[1] + 3, b[2] + 3, b[3] + 3, RADIO_CM[1]] for b in barras])
        p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)
        r = least_squares(lambda p: self.resolver(list(p.reshape(-1, 5)))[1], p0, bounds=(lo, hi),
                          diff_step=1e-3, x_scale=np.tile([1, 1, 1, 1, 0.3], len(barras)), max_nfev=60)
        dof = max(len(self.y) - self.n_fijas - 6 * len(barras), 1)
        try:
            cov = np.linalg.pinv(r.jac.T @ r.jac) * np.sum(r.fun ** 2) / dof
            err = np.sqrt(np.clip(np.diag(cov), 0, None)).reshape(-1, 5)
        except np.linalg.LinAlgError:
            err = np.full((len(barras), 5), np.nan)
        return list(r.x.reshape(-1, 5)), err

    def significancias(self, barras):
        # Δχ² al quitar cada barra (los demás parámetros lineales se reajustan) -> σ con 6 parámetros
        base = self.chi2(barras)
        coef = self.resolver(barras)[0][self.n_fijas:]
        sig = []
        for k in range(len(barras)):
            if coef[k] <= 0:   # amplitud negativa: no es hierro
                sig.append(0.0)
                continue
            dchi = max(self.chi2(barras[:k] + barras[k + 1:]) - base, 0.0)
            sig.append(max(float(norm.isf(max(chi2.sf(dchi, 6), 1e-300))), 0.0) if dchi > 0 else 0.0)
        return np.array(sig), coef

# ------------------------------------------------------------------------------------------
# Candidatas desde la red
# ------------------------------------------------------------------------------------------
def candidatas(prob, umbral, region, largo_min_cm=4.0):
    from skimage.measure import label, regionprops
    mascara = (prob > umbral) & region
    esqueleto = skeletonize(mascara)
    segmentos = probabilistic_hough_line(esqueleto, threshold=8, line_length=int(largo_min_cm / VOXEL_CM),
                                         line_gap=4, rng=0)
    # skimage devuelve (columna, fila) = (j, i): a cm del detector (x = fila i, y = columna j)
    segs = [np.array([(p0[1] + 0.5) * VOXEL_CM, (p0[0] + 0.5) * VOXEL_CM,
                      (p1[1] + 0.5) * VOXEL_CM, (p1[0] + 0.5) * VOXEL_CM]) for p0, p1 in segmentos]
    # Manchas compactas (una barra ancha y corta): su eje principal, del largo de la mancha
    for comp in regionprops(label(mascara)):
        if comp.area * VOXEL_CM ** 2 > 150 or comp.axis_major_length * VOXEL_CM < largo_min_cm:
            continue
        ci, cj = comp.centroid
        th = comp.orientation                        # ángulo del eje mayor respecto a las filas
        mitad = 0.5 * (comp.axis_major_length - comp.axis_minor_length * 0.5)
        di, dj = np.cos(th) * mitad, np.sin(th) * mitad
        segs.append(np.array([ci - di + 0.5, cj - dj + 0.5, ci + di + 0.5, cj + dj + 0.5]) * VOXEL_CM)
    return quitar_repetidas(unir_colineales(segs))

def quitar_repetidas(segs, radio_cm=1.0, solape_max=0.3):
    # Dos candidatas sobre la misma barra se reparten la señal y ninguna parece significativa:
    # se deja la más larga de cada par que se solapa
    segs = sorted(segs, key=lambda s: -np.hypot(*(s[2:] - s[:2])))
    huellas, elegidas = [], []
    for s in segs:
        h = cuerda_barra(*s, radio_cm) > 0
        if any((h & g).sum() > solape_max * min(h.sum(), g.sum()) for g in huellas):
            continue
        huellas.append(h)
        elegidas.append(s)
    return elegidas

def unir_colineales(segs, angulo_max=8.0, offset_max=1.5, hueco_max=4.0):
    # Une segmentos casi colineales (misma barra partida por el ruido)
    segs = [s for s in segs]
    cambio = True
    while cambio:
        cambio = False
        for a in range(len(segs)):
            for b in range(a + 1, len(segs)):
                s1, s2 = segs[a], segs[b]
                d1, d2 = s1[2:] - s1[:2], s2[2:] - s2[:2]
                u = d1 / max(np.linalg.norm(d1), 1e-9)
                ang = np.degrees(np.arccos(min(abs(np.dot(u, d2 / max(np.linalg.norm(d2), 1e-9))), 1.0)))
                n = np.array([-u[1], u[0]])
                off = abs(np.dot((s2[:2] + s2[2:]) / 2 - s1[:2], n))
                t = [np.dot(p - s1[:2], u) for p in (s1[:2], s1[2:], s2[:2], s2[2:])]
                hueco = max(min(t[2], t[3]) - max(t[0], t[1]), min(t[0], t[1]) - max(t[2], t[3]), 0)
                if ang < angulo_max and off < offset_max and hueco < hueco_max:
                    t0, t1 = min(t), max(t)
                    segs[a] = np.concatenate([s1[:2] + t0 * u, s1[:2] + t1 * u])
                    segs.pop(b)
                    cambio = True
                    break
            if cambio:
                break
    return segs

def radio_inicial(prob, umbral, seg):
    # Medio ancho de la zona marcada a lo largo del segmento, acotado
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(prob > umbral) * VOXEL_CM
    t = np.linspace(0.2, 0.8, 7)
    puntos = seg[:2][None] + t[:, None] * (seg[2:] - seg[:2])[None]
    i = np.clip((puntos[:, 0] / VOXEL_CM).astype(int), 0, LADO - 1)
    j = np.clip((puntos[:, 1] / VOXEL_CM).astype(int), 0, LADO - 1)
    return float(np.clip(0.6 * np.median(dist[i, j]), RADIO_CM[0] + 0.05, RADIO_CM[1] - 0.05))

# ------------------------------------------------------------------------------------------
# Ajuste completo de una imagen
# ------------------------------------------------------------------------------------------
def region_bloque(densidad, muones):
    # Bloque: densidad suavizada sobre el 25% de la mediana (el aire da ~0; la arena entre barras
    # puede bajar mucho, así que no se usa la densidad sin suavizar)
    suave = gaussian_filter(densidad, 4)
    bloque = suave > 0.25 * np.median(suave[suave > 0.3]) if (suave > 0.3).any() else suave > 0
    return binary_erosion(bloque, iterations=int(EROSION_BLOQUE_CM / VOXEL_CM)) & (muones >= MUONES_MINIMOS)

def ajustar_imagen(medida, muones, prob, densidad, umbral, sigma_minimo=None, probar_dobles=True, verbose=False):
    sigma_minimo = SIGMA_MINIMO if sigma_minimo is None else sigma_minimo
    region = region_bloque(densidad, muones)
    aj = Ajuste(medida, muones, region)
    barras = [np.concatenate([s, [radio_inicial(prob, umbral, s)]]) for s in candidatas(prob, umbral, region)]
    if verbose:
        print(f"  {len(barras)} candidatas")
    barras, err = aj.ajustar(barras)
    while barras:
        sig, _ = aj.significancias(barras)
        k = int(np.argmin(sig))
        if sig[k] >= sigma_minimo and sig[k] > 0:
            break
        barras.pop(k)
        barras, err = aj.ajustar(barras)
    if probar_dobles and barras:
        barras, err = probar_barras_dobles(aj, barras, err, sigma_minimo, verbose)
    sig, coef = aj.significancias(barras) if barras else (np.zeros(0), np.zeros(0))
    chi2_final = aj.chi2(barras)
    return dict(barras=barras, errores=err, significancia=sig, amplitud=coef, region=region,
                chi2_dof=chi2_final / max(region.sum() - aj.n_fijas - 6 * len(barras), 1), ajuste=aj)

def probar_barras_dobles(aj, barras, err, sigma_minimo, verbose):
    # Una barra ancha puede ser dos barras paralelas juntas: se prueba la separación y se acepta
    # si mejora el ajuste a sigma_minimo con los 6 parámetros extra
    k = 0
    while k < len(barras):
        x1, y1, x2, y2, r = barras[k]
        if r < 1.0:
            k += 1
            continue
        u = np.array([x2 - x1, y2 - y1]) / max(np.hypot(x2 - x1, y2 - y1), 1e-9)
        n = np.array([-u[1], u[0]]) * r * 0.9
        par = [np.array([x1 + s * n[0], y1 + s * n[1], x2 + s * n[0], y2 + s * n[1], r * 0.5]) for s in (-1, 1)]
        prueba, err_p = aj.ajustar(barras[:k] + par + barras[k + 1:])
        dchi = aj.chi2(barras) - aj.chi2(prueba)
        if dchi > 0 and norm.isf(max(chi2.sf(dchi, 6), 1e-300)) >= sigma_minimo:
            if verbose:
                print(f"  barra {k} (Ø{2 * r:.1f} cm) -> dos barras (Δχ² = {dchi:.0f})")
            barras, err = prueba, err_p
            k += 2
        else:
            k += 1
    return barras, err

# ------------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------------
def tabla_barras(res, confiable):
    filas = []
    for k, (b, e, s, a) in enumerate(zip(res["barras"], res["errores"], res["significancia"], res["amplitud"])):
        x1, y1, x2, y2, r = b
        cuerda = cuerda_barra(*b) > 0
        filas.append({
            "bar": k + 1, "x1_cm": round(x1, 2), "y1_cm": round(y1, 2), "x2_cm": round(x2, 2), "y2_cm": round(y2, 2),
            "length_cm": round(float(np.hypot(x2 - x1, y2 - y1)), 1),
            "angle_deg": round(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180), 1),
            "diameter_cm": round(2 * r, 2), "diameter_err_cm": round(2 * float(e[4]), 2),
            "amplitude_rad2_per_cm": float(a), "significance_sigma": round(float(min(s, 38.0)), 1),
            "fraction_in_reliable_zone": round(float(confiable[cuerda].mean()) if cuerda.any() else 0.0, 2),
        })
    return filas

def modelo_rigido(res):
    return sum((cuerda_barra(*b) for b in res["barras"]), np.zeros((LADO, LADO)))

def figura(nombre, medida, prob, res, path, mostrar=None, etiqueta_entrada="Measured ASD"):
    # mostrar (opcional): mapa que se dibuja como medida (p. ej. el estimador clásico de la tesis)
    geometria = inf.geometria_de(nombre)
    titulo = inf.NOMBRES_TESIS.get(nombre, nombre)
    aj = res["ajuste"]
    coef, residuo, _ = aj.resolver(res["barras"])
    mapa_res = np.full((LADO, LADO), np.nan)
    mapa_res[res["region"]] = residuo
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    inf.show_map(axes[0], (medida if mostrar is None else mostrar) / VOXEL_CM, 'asd', *inf.ESCALA_ENTRADA_ASD,
                 etiqueta_entrada, geometria)
    inf.show_map(axes[1], prob, 'prob', 0.0, 1.0, "Iron probability", geometria)
    rigido = modelo_rigido(res)
    inf.TIPOS_MAPA['rigid'] = ('bone_r', False, 'Projected iron thickness [cm]')
    inf.show_map(axes[2], rigido, 'rigid', 0.0, max(float(rigido.max()), 1.0),
                 "Rigid bar fit", geometria)
    for k, b in enumerate(res["barras"]):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if inf.TRANSPONER:
            cx, cy = cy, cx
        axes[2].text(cx, cy, str(k + 1), color='tab:red', fontsize=7, ha='center', va='center')
    inf.TIPOS_MAPA['resid'] = ('RdBu_r', False, 'Normalized residual')
    inf.show_map(axes[3], np.nan_to_num(mapa_res), 'resid', -3.0, 3.0,
                 "Normalized residual", geometria)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    VERSION = "V16"
    SALIDA = f"resultados/{VERSION}/postprocessing/rigid_fit"
    os.makedirs(SALIDA, exist_ok=True)
    with open(f"resultados/{VERSION}/test_evaluation/iron_threshold.json") as f:
        inf.UMBRAL_HIERRO = json.load(f)["iron_threshold"]
    model, device = inf.cargar_modelo(f"modelos/{VERSION}_fp16.pt")
    confiable = mapa_confiable()
    resumen = {}
    for base in ["Mallado_arena_20d_21.6d", "Mallado_mortero_20d_18.9d", "Grosores_mortero_20d_20.6d"]:
        nombre = base.rsplit("_", 1)[0]
        medida = inf.limpiar(np.load(f"datos_reales/{base}_varrec_xy.npy"))
        muones = inf.limpiar(np.load(f"datos_reales/{base}_N_xy.npy"))
        _, dens, prob = inf.inferir_mapa(model, device, medida, muones)
        print(base)
        res = ajustar_imagen(medida, muones, prob, dens, inf.UMBRAL_HIERRO, verbose=True)
        filas = tabla_barras(res, confiable)
        with open(os.path.join(SALIDA, f"{base}_bars.csv"), "w", newline="") as f:
            if filas:
                w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
                w.writeheader()
                w.writerows(filas)
        np.save(os.path.join(SALIDA, f"{base}_rigid_model.npy"), modelo_rigido(res).astype(np.float32))
        figura(nombre, medida, prob, res, os.path.join(SALIDA, f"{base}_rigid_fit.png"))
        for fila in filas:
            print("   ", fila)
        resumen[base] = {"bars": len(filas), "chi2_dof": res["chi2_dof"]}
    with open(os.path.join(SALIDA, "rigid_fit_summary.json"), "w") as f:
        json.dump(resumen, f, indent=2)
