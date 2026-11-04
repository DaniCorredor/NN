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
import calibrar_bordes as cb
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
ETIQUETA_CLASICA = "Measured (classic estimator)"

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

def figura_confiabilidad(path):
    # 1) evidencia: perfil del exceso medido en aire (sin muestra) a lo largo del y del detector, con el
    #    modelo ajustado; 2) modelo 2D del exceso; 3) zona confiable (exceso < mitad de la señal de Ø2 cm)
    p = {k: float(v) for k, v in np.load(cb.EXCESO_BORDES_PATH).items()}
    plomo, grosores = cb.voxel_data("plomo_bloques_84h"), cb.voxel_data("Grosores_mortero_20d")
    fuentes = {"Empty hodoscope (16 h)": cb.perfil(cb.VACIO, [(14, 42)], 1),
               "Lead run (air regions)": cb.perfil(plomo, [(21.5, 27), (39.5, 43)], 1),
               "Thicknesses run (air regions)": cb.perfil(grosores, [(46, 55)], 1)}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.2))
    ax = axes[0]
    for (nombre, (pos, exceso, error)), color in zip(fuentes.items(), ['k', 'tab:blue', 'tab:orange']):
        ax.errorbar(pos, exceso * 1e3, yerr=error * 1e3, fmt='o', ms=3, color=color, alpha=0.7, label=nombre)
    y = np.linspace(0, 56, 400)
    ax.plot(y, cb.g_y(y, p["aB"], p["yB"], p["wB"], p["aT"], p["yT"], p["wT"]) * 1e3, 'r-', lw=2,
            label='Fit: band + step')
    umbral = cf.FRACCION_SENAL * cf.senal_barra_referencia()
    ax.axhline(umbral * 1e3, color='green', ls='--', label=f'Threshold: ½ signal of a Ø{cf.DIAMETRO_REFERENCIA_CM:g} cm bar')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlim(*inf.LIMITES_CM)
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Excess variance [10$^{-3}$ rad$^2$]")
    ax.set_title("Detector excess without sample (profile across X)", fontsize=10)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=8)
    exceso = cf.exceso_detector()
    inf.TIPOS_MAPA['excess'] = ('magma', False, 'Detector excess variance [rad$^2$]')
    inf.show_map(axes[1], exceso, 'excess', 0.0, float(np.percentile(exceso, 99)),
                 "Fitted detector excess (band + step + ramp)", [])
    inf.TIPOS_MAPA['ok'] = ('Greens', False, 'Reliable (1) / reduced reliability (0)')
    inf.show_map(axes[2], cf.mapa_confiable().astype(float), 'ok', 0.0, 1.0,
                 f"Reliable zone: excess < ½ Ø{cf.DIAMETRO_REFERENCIA_CM:g} cm bar signal, ≥ {cf.MUONES_MINIMOS} muons", [])
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
                       f"{inf.NOMBRES_TESIS.get(muestra, muestra)}: INDEPENDENT time intervals (no muon in common)",
                       entrada=ETIQUETA_CLASICA)

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
        print(muestra, "ok", flush=True)
    print(f"Figuras en {DESTINO}")
