import os
import csv
import json
import shutil
import numpy as np

import inference_unet_2_real as inf
import ajuste_barras as aj
import bootstrap_muones as bs
from train_Unet_2 import plot_loss_curves

# Rehace las figuras de la tesis desde los resultados guardados (sin reentrenar, sin repetir el
# bootstrap ni el ajuste rígido) y las reúne en resultados/V16/thesis_figures.

VERSION = "V16"
POST = f"resultados/{VERSION}/postprocessing"
DESTINO = f"resultados/{VERSION}/thesis_figures"
TOMAS = {"Mallado_arena_20d": "21.6d", "Mallado_mortero_20d": "18.9d", "Grosores_mortero_20d": "20.6d"}

def curvas():
    with open(f"checkpoints/{VERSION}/historial.csv") as f:
        historial = [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]
    mejor = int(min(historial, key=lambda h: h['val_total'])['epoca'])
    plot_loss_curves(historial, mejor, os.path.join(DESTINO, "01_training_validation_curves.png"))

def leer_barras(path):
    with open(path) as f:
        filas = list(csv.DictReader(f))
    barras = [np.array([float(r['x1_cm']), float(r['y1_cm']), float(r['x2_cm']), float(r['y2_cm']),
                        float(r['diameter_cm']) / 2]) for r in filas]
    return barras, np.array([float(r['significance_sigma']) for r in filas])

if __name__ == "__main__":
    os.makedirs(DESTINO, exist_ok=True)
    with open(f"resultados/{VERSION}/test_evaluation/iron_threshold.json") as f:
        inf.UMBRAL_HIERRO = json.load(f)["iron_threshold"]
    curvas()
    model, device = inf.cargar_modelo(f"modelos/{VERSION}_fp16.pt")
    for muestra, dias in TOMAS.items():
        base = f"{muestra}_{dias}"
        medida = inf.limpiar(np.load(f"datos_reales/{base}_varrec_xy.npy"))
        muones = inf.limpiar(np.load(f"datos_reales/{base}_N_xy.npy"))
        salida = inf.inferir_mapa(model, device, medida, muones)

        # 06: bootstrap (frecuencia e incertidumbre guardadas por bootstrap_muones.py)
        frecuencia = np.load(f"{POST}/bootstrap/{muestra}_detection_frequency.npy")
        std_dens = np.load(f"{POST}/bootstrap/{muestra}_density_std.npy")
        bs.figura(muestra, medida, salida, frecuencia, std_dens, f"{POST}/bootstrap/{muestra}_bootstrap.png")
        shutil.copy(f"{POST}/bootstrap/{muestra}_bootstrap.png", f"{DESTINO}/06_bootstrap_{muestra}.png")

        # 07: ajuste rígido (barras guardadas por ajuste_barras.py; el fondo lineal se recalcula)
        barras, sig = leer_barras(f"{POST}/rigid_fit/{base}_bars.csv")
        region = aj.region_bloque(salida[1], muones)
        ajuste = aj.Ajuste(medida, muones, region)
        res = dict(barras=barras, significancia=sig, region=region, ajuste=ajuste,
                   chi2_dof=ajuste.chi2(barras) / max(region.sum() - ajuste.n_fijas - 6 * len(barras), 1))
        aj.figura(muestra, medida, salida[2], res, f"{POST}/rigid_fit/{base}_rigid_fit.png")
        shutil.copy(f"{POST}/rigid_fit/{base}_rigid_fit.png", f"{DESTINO}/07_{base}_rigid_fit.png")
    print(f"Figuras en {DESTINO}")
