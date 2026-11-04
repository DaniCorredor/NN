import os
import csv
import json
import numpy as np
from scipy.ndimage import gaussian_filter

import data_generator as dg
import inference_unet_2_real as inf
import ajuste_barras as aj

# Valida el ajuste rígido (Nivel A) en escenas SIMULADAS con la geometría conocida: misma cadena
# que con los datos reales (ruido de 20 días con las bandas del detector -> red -> ajuste).
#   recall por diámetro      barras reales encontradas (huella dentro del bloque solapada > 50%)
#   barras falsas por escena ajustadas que no corresponden a ninguna real (y su significancia)
#   error del diámetro       ajustado - real, en las barras encontradas
# Sirve para fijar el umbral de significancia sin mirar las muestras reales: el ajuste se corre sin
# umbral (se guardan todas las barras con su significancia) y luego se barre el umbral. Se elige el
# menor umbral con a lo sumo FALSAS_MAXIMAS barras falsas por escena -> umbral_significancia.json,
# que usa ajuste_barras.py.

VERSION = "V16"
SALIDA = f"resultados/{VERSION}/postprocessing/rigid_fit_validation"
ESCENAS = {"malla": range(16, 48), "grosores": range(48, 70), "barras": range(70, 88)}
POR_TIPO = 15
DIAS = (15.0, 25.0)
FALSAS_MAXIMAS = 0.1
UMBRALES = np.arange(0.0, 10.01, 0.25)

def escena_con_verdad(gen, i):
    # Construye la escena i y guarda las barras de hierro que se agregan (cm del detector)
    verdad = []
    original = gen.add_bar
    def registrar(start, end, radius, material='iron'):
        if material == 'iron':
            verdad.append(np.array([start[0], start[1], end[0], end[1], radius]) * dg.VOXEL_CM)
        return original(start, end, radius, material)
    gen.add_bar = registrar
    gen.reset(background='air')
    tipo = dg.build_scene(gen, i)
    gen.add_bar = original
    return tipo, verdad

def simular(gen, aceptancia, rng):
    T, T_hierro = gen.simulate_transmission()
    T_det = T - (1.0 - rng.uniform(*dg.FACTOR_HIERRO)) * T_hierro
    x_limpio = dg.measured_variance(gaussian_filter(T_det, rng.uniform(*dg.PSF_SIGMA_RANGO_CM) / dg.VOXEL_CM), clean=True)
    medida, muones, _ = dg.agregar_ruido(x_limpio, aceptancia, rng, dias=rng.uniform(*DIAS))
    return medida, muones.astype(np.float32)

def emparejar(verdad, ajustadas, region):
    huellas_v = [(aj.cuerda_barra(*b) > 0) & region for b in verdad]
    huellas_a = [aj.cuerda_barra(*b) > 0 for b in ajustadas]
    usadas = set()
    encontrada = []
    for hv in huellas_v:
        mejor = None
        for k, ha in enumerate(huellas_a):
            if k in usadas or hv.sum() == 0:
                continue
            if (hv & ha).sum() > 0.5 * min(hv.sum(), ha.sum()):
                mejor = k
                break
        encontrada.append(mejor)
        if mejor is not None:
            usadas.add(mejor)
    falsas = [k for k in range(len(ajustadas)) if k not in usadas]
    return encontrada, falsas, [hv.sum() for hv in huellas_v]

if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    np.random.seed(2024)
    rng = np.random.default_rng(2024)
    with open(f"resultados/{VERSION}/test_evaluation/iron_threshold.json") as f:
        umbral = json.load(f)["iron_threshold"]
    model, device = inf.cargar_modelo(f"modelos/{VERSION}_fp16.pt")
    gen = dg.PhysicalMuographyGenerator(grid_size=dg.GRID, voxel_size_cm=dg.VOXEL_CM, muones_csv_path="Momento_Muones_filtrado.csv")
    aceptancia = np.load(dg.ACEPTANCIA_PATH)
    filas_barras, filas_falsas, filas_escenas = [], [], []
    for tipo, indices in ESCENAS.items():
        for i in rng.choice(list(indices), POR_TIPO, replace=False):
            escena, verdad = escena_con_verdad(gen, int(i))
            medida, muones = simular(gen, aceptancia, rng)
            _, dens, prob = inf.inferir_mapa(model, device, medida, muones)
            res = aj.ajustar_imagen(medida, muones, prob, dens, umbral, sigma_minimo=0.0, probar_dobles=False)
            encontrada, falsas, huella = emparejar(verdad, res["barras"], res["region"])
            for b, k, h in zip(verdad, encontrada, huella):
                if h < 8:       # la barra real casi no está dentro de la zona ajustada
                    continue
                filas_barras.append({"scene": escena, "true_diameter_cm": round(2 * b[4], 2),
                                     "true_length_cm": round(float(np.hypot(b[2] - b[0], b[3] - b[1])), 1),
                                     "found": k is not None,
                                     "fit_diameter_cm": round(2 * res["barras"][k][4], 2) if k is not None else np.nan,
                                     "significance": round(float(res["significancia"][k]), 1) if k is not None else np.nan})
            for k in falsas:
                filas_falsas.append({"scene": escena, "significance": round(float(res["significancia"][k]), 1),
                                     "fit_diameter_cm": round(2 * res["barras"][k][4], 2)})
            filas_escenas.append({"scene": escena, "true_bars": int(sum(h >= 8 for h in huella)),
                                  "fitted_bars": len(res["barras"]), "false_bars": len(falsas)})
            print(filas_escenas[-1], flush=True)

    for nombre, filas in [("bars", filas_barras), ("false_bars", filas_falsas), ("scenes", filas_escenas)]:
        with open(os.path.join(SALIDA, f"validation_{nombre}.csv"), "w", newline="") as f:
            if filas:
                w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
                w.writeheader()
                w.writerows(filas)

    # Barrido del umbral: una barra real cuenta como encontrada si su barra ajustada lo supera
    d = np.array([f["true_diameter_cm"] for f in filas_barras])
    sig_v = np.array([f["significance"] if f["found"] else -1.0 for f in filas_barras])
    sig_f = np.array([f["significance"] for f in filas_falsas])
    rangos = [(1, 1.5), (1.5, 2.5), (2.5, 4), (4, 6.1)]
    barrido = []
    for u in UMBRALES:
        fila = {"threshold_sigma": float(u), "false_bars_per_scene": float((sig_f >= u).sum() / len(filas_escenas))}
        for lo, hi in rangos:
            sel = (d >= lo) & (d < hi)
            fila[f"recall_d{lo}-{hi}"] = float((sig_v[sel] >= u).mean()) if sel.any() else np.nan
        barrido.append(fila)
    with open(os.path.join(SALIDA, "threshold_scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(barrido[0].keys()))
        w.writeheader()
        w.writerows(barrido)
    elegido = next(f for f in barrido if f["false_bars_per_scene"] <= FALSAS_MAXIMAS)
    with open(os.path.join(SALIDA, "umbral_significancia.json"), "w") as f:
        json.dump(elegido, f, indent=2)
    print("threshold  false/scene  " + "  ".join(f"Ø{lo}-{hi}" for lo, hi in rangos))
    for f in barrido[::4]:
        print(f"{f['threshold_sigma']:6.2f}  {f['false_bars_per_scene']:8.2f}     " +
              "  ".join(f"{f[f'recall_d{lo}-{hi}']:6.2f}" for lo, hi in rangos))
    print("Chosen:", elegido)
    err = np.array([f["fit_diameter_cm"] - f["true_diameter_cm"] for f in filas_barras
                    if f["found"] and f["significance"] >= elegido["threshold_sigma"]])
    print(f"Diameter error at that threshold (fit - true): mean {err.mean():+.2f} cm, std {err.std():.2f} cm")
