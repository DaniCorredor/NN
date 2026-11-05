import os
import re
import csv
import glob
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import inference_unet_2_real as inf
import ajuste_barras as aj
import bootstrap_muones as bs
import confiabilidad as cf
from preparar_recortada import ARCHIVOS_POR_DIA, DATOS, PATRON
from train_Unet_2 import plot_loss_curves

# Rehace las figuras de la tesis desde los resultados guardados (sin reentrenar, sin repetir el
# bootstrap ni el ajuste rígido) y las reúne en resultados/V16/thesis_figures.
#
# El panel "medido" usa el ESTIMADOR CLÁSICO de la tesis (como los toma_*_d_X-Y.npy): en cada vóxel
# la media de theta^2 de sus muones, promediada en z contando los vóxeles vacíos como 0, sobre la
# voxelización con cortes DOCA <= 4 cm e incidencia < 15°. Es solo para mostrar: la red recibe la
# media recortada de cada columna (preparar_recortada.py), con los mismos muones.

VERSION = "V16"
POST = f"resultados/{VERSION}/postprocessing"
DESTINO = f"resultados/{VERSION}/thesis_figures"
TOMAS = {"Mallado_arena_20d": "21.6d", "Mallado_mortero_20d": "18.9d", "Grosores_mortero_20d": "20.6d"}
ETIQUETA_CLASICA = "Measured ASD"

def mapa_clasico(vt, desde=None):
    # Media de theta^2 por vóxel (solo los muones nuevos si se da 'desde'), promedio en z con vacíos = 0 [rad^2]
    medias = np.zeros(vt.shape)
    largos = np.zeros(vt.shape, dtype=np.int32)
    for idx in zip(*np.nonzero(np.vectorize(lambda v: v is not None and len(v) > 0, otypes=[bool])(vt))):
        v = np.asarray(vt[idx], float)
        largos[idx] = len(v)
        inicio = 0 if desde is None else desde[idx]
        if len(v) > inicio:
            medias[idx] = np.mean(v[inicio:] ** 2)
    return medias.mean(axis=2).astype(np.float32), largos

def clasicos_de_muestra(muestra):
    # Mapas clásicos de la toma acumulada más larga y de cada intervalo independiente: {nombre: mapa}
    rutas = sorted(glob.glob(os.path.join(DATOS, muestra, PATRON)), key=lambda r: int(re.search(r"_n_(\d+)_", r).group(1)))
    mapas, anterior, dias_ant = {}, None, 0   # mismo nombre que preparar_recortada: int_0-Xd
    for ruta in rutas:
        dias = round(int(re.search(r"_n_(\d+)_", ruta).group(1)) / ARCHIVOS_POR_DIA, 1)
        vt = np.load(ruta, allow_pickle=True)["voxel_theta"]
        acumulado, largos = mapa_clasico(vt)
        mapas[f"{muestra}_{dias}d"] = acumulado
        mapas[f"{muestra}_int_{dias_ant}-{dias}d"] = acumulado if anterior is None else mapa_clasico(vt, anterior)[0]
        anterior, dias_ant = largos, dias
    return mapas

def curvas():
    with open(f"checkpoints/{VERSION}/historial.csv") as f:
        historial = [{k: float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]
    mejor = int(min(historial, key=lambda h: h['val_total'])['epoca'])
    plot_loss_curves(historial, mejor, os.path.join(DESTINO, "01_training_validation_curves.png"))

BINS_RED = np.arange(0.2, 1.0001, 0.1)

def estadistica_por_red(red):
    # Por rango de RED, dentro del bloque y lejos de la geometría estimada (solo para evaluar, no se
    # dibuja): ruido de la media recortada relativo al centro y fracción marcada como hierro por V16
    from scipy.ndimage import binary_dilation, gaussian_filter
    c = (np.arange(112) + 0.5) * inf.VOXEL_CM
    X, Y = np.meshgrid(c, c, indexing="ij")
    salida = {}
    for muestra, dias in TOMAS.items():
        guia = np.zeros((112, 112), bool)
        for r in inf.geometria_de(muestra):
            guia |= (X >= r['x'][0]) & (X <= r['x'][1]) & (Y >= r['y'][0]) & (Y <= r['y'][1])
        lejos = ~binary_dilation(guia, iterations=4)
        dens = np.load(f"resultados/{VERSION}/{muestra}_{dias}_density_clean.npy")
        hierro = np.load(f"resultados/{VERSION}/{muestra}_{dias}_iron_probability.npy") > inf.UMBRAL_HIERRO
        v = np.load(f"datos_reales/{muestra}_{dias}_varrec_xy.npy")
        bloque = gaussian_filter(dens, 4) > 0.3 * np.median(dens[dens > 0.3])
        fino = v - gaussian_filter(v, 3)
        zona = bloque & lejos
        ruido_centro = fino[zona & (red >= 0.8)].std()
        ruido, falso = [], []
        for a, b in zip(BINS_RED[:-1], BINS_RED[1:]):
            sel = zona & (red >= a) & (red < b)
            ruido.append(fino[sel].std() / ruido_centro if sel.sum() > 30 else np.nan)
            falso.append(hierro[sel].mean() if sel.sum() > 30 else np.nan)
        salida[muestra] = (np.array(ruido), np.array(falso))
    return salida

def figura_confiabilidad(path):
    # 1) RED (Monte Carlo) con el umbral; 2) ruido medido vs RED con 1/sqrt(RED)
    red = cf.densidad_relativa()
    centros = 0.5 * (BINS_RED[:-1] + BINS_RED[1:])
    datos = estadistica_por_red(red)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    inf.TIPOS_MAPA['red'] = ('inferno', False, 'Relative event density')
    inf.show_map(axes[0], red, 'red', 0.0, 1.0, "Relative event density", [])
    axes[0].contour((np.arange(112) + 0.5) * inf.VOXEL_CM, (np.arange(112) + 0.5) * inf.VOXEL_CM,
                    red if inf.TRANSPONER else red.T, levels=[cf.RED_MINIMA], colors='cyan', linewidths=1.5)
    colores = {'Mallado_arena_20d': 'tab:orange', 'Mallado_mortero_20d': 'tab:blue', 'Grosores_mortero_20d': 'tab:green'}
    r = np.linspace(0.2, 1.0, 100)
    axes[1].plot(r, 1 / np.sqrt(r), 'k--', label='$1/\\sqrt{RED}$')
    for muestra, (ruido, falso) in datos.items():
        nombre = inf.NOMBRES_TESIS.get(muestra, muestra)
        axes[1].plot(centros, ruido, 'o-', color=colores[muestra], label=nombre)
    for ax, titulo, etiqueta in [(axes[1], "Noise vs relative event density", "Noise relative to centre")]:
        ax.axvline(cf.RED_MINIMA, color='cyan', lw=1.5, label=f'Threshold RED = {cf.RED_MINIMA:g}')
        ax.set_xlabel("Relative event density")
        ax.set_ylabel(etiqueta)
        ax.set_title(titulo, fontsize=10)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=8)
    axes[1].set_ylim(0, 4)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

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
    figura_confiabilidad(os.path.join(DESTINO, "04_reliability_map.png"))
    model, device = inf.cargar_modelo(f"modelos/{VERSION}_fp16.pt")
    for muestra, dias in TOMAS.items():
        base = f"{muestra}_{dias}"
        clasicos = clasicos_de_muestra(muestra)
        medida = inf.limpiar(np.load(f"datos_reales/{base}_varrec_xy.npy"))
        muones = inf.limpiar(np.load(f"datos_reales/{base}_N_xy.npy"))
        salida = inf.inferir_mapa(model, device, medida, muones)

        # 05: intervalos independientes (la red recibe la media recortada de cada intervalo)
        filas = []
        for nombre in sorted((k for k in clasicos if "_int_" in k), key=lambda k: float(k.split("_int_")[1].split("-")[0])):
            m = inf.limpiar(np.load(f"datos_reales/{nombre}_varrec_xy.npy"))
            n = inf.limpiar(np.load(f"datos_reales/{nombre}_N_xy.npy"))
            v, d, p = inf.inferir_mapa(model, device, m, n)
            a, b = nombre.split("_int_")[1][:-1].split("-")
            filas.append({'etiqueta': f"days {float(a):g}-{float(b):g}", 'input': m, 'input_display': clasicos[nombre],
                          'var': v, 'dens': d, 'prob': p})
        inf.plot_tomas(muestra, filas, os.path.join(DESTINO, f"05_independent_intervals_{muestra}.png"),
                       "", entrada=ETIQUETA_CLASICA)

        # 06: bootstrap (frecuencia e incertidumbre guardadas por bootstrap_muones.py)
        frecuencia = np.load(f"{POST}/bootstrap/{muestra}_detection_frequency.npy")
        std_dens = np.load(f"{POST}/bootstrap/{muestra}_density_std.npy")
        bs.figura(muestra, medida, salida, frecuencia, std_dens, f"{DESTINO}/06_bootstrap_{muestra}.png",
                  mostrar=clasicos[base], etiqueta_entrada=ETIQUETA_CLASICA)

        # 07: ajuste rígido (barras guardadas por ajuste_barras.py; el fondo lineal se recalcula)
        barras, sig = leer_barras(f"{POST}/rigid_fit/{base}_bars.csv")
        region = aj.region_bloque(salida[1], muones)
        ajuste = aj.Ajuste(medida, muones, region)
        res = dict(barras=barras, significancia=sig, region=region, ajuste=ajuste,
                   chi2_dof=ajuste.chi2(barras) / max(region.sum() - ajuste.n_fijas - 6 * len(barras), 1))
        aj.figura(muestra, medida, salida[2], res, f"{DESTINO}/07_{base}_rigid_fit.png",
                  mostrar=clasicos[base], etiqueta_entrada=ETIQUETA_CLASICA)
        filas_csv = aj.tabla_barras(dict(res, errores=np.zeros((len(barras), 5)),
                                         amplitud=ajuste.resolver(barras)[0][ajuste.n_fijas:]), cf.mapa_confiable())
        with open(f"{POST}/rigid_fit/{base}_bars.csv") as f:
            originales = list(csv.DictReader(f))
        for fila, orig in zip(filas_csv, originales):   # errores y significancia del ajuste original
            fila.update(diameter_err_cm=orig['diameter_err_cm'], significance_sigma=orig['significance_sigma'])
        with open(f"{DESTINO}/07_{base}_bars.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
            w.writeheader()
            w.writerows(filas_csv)
        print(muestra, "ok", flush=True)
    print(f"Figuras en {DESTINO}")
