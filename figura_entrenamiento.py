import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

import data_generator as dg
import inference_unet_2_real as inf
import calibrar_respuesta as cr

# Figura explicativa del entrenamiento (reunión): cómo se construye un dato simulado y con qué
# se calibró la respuesta del detector.
#   Fila 1: escena simulada -> material proyectado (target) -> varianza esperada -> input con ruido (20 d)
#   Fila 2: calibración V = V0 + A T^gamma (vs Highland lineal) y ruido simulado vs medido en concreto

SALIDA = "resultados/V16/reunion"
DIAS = 20.6
REGION_CONCRETO = ((15, 39), (30, 35))     # Grosores sin barras: la misma zona de calibrar_respuesta.py

def escena_ejemplo(gen):
    # Bloque de concreto con una malla tipo M1 y una fila de barras de distinto grosor
    np.random.seed(11)
    gen.reset(background='air')
    gen.materials['concrete'] = 2.0 / 26.6
    bounds = gen.add_block((56, 56, 60), (90, 90, 42))
    dg.add_rebar_mesh(gen, bounds)
    return gen.simulate_transmission()

def ruido_concreto(aceptancia, rng, repeticiones=40):
    # Mismo concreto que la región de Grosores (25 cm, 1.82 g/cm3), mismo tiempo de medición
    T = np.full((112, 112), 25 * 1.82 / 26.6, dtype=np.float32)
    limpio = dg.measured_variance(T, clean=True)
    (x0, x1), (y0, y1) = REGION_CONCRETO
    sl = (slice(int(x0 / 0.5), int(x1 / 0.5)), slice(int(y0 / 0.5), int(y1 / 0.5)))
    return np.concatenate([dg.agregar_ruido(limpio, aceptancia, rng, dias=DIAS)[0][sl].ravel()
                           for _ in range(repeticiones)]), sl

if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    inf.MOSTRAR_CONFIABILIDAD = False
    gen = dg.PhysicalMuographyGenerator(grid_size=dg.GRID, voxel_size_cm=dg.VOXEL_CM,
                                        muones_csv_path="Momento_Muones_filtrado.csv")
    aceptancia = np.load(dg.ACEPTANCIA_PATH)
    rng = np.random.default_rng(3)

    T, T_hierro = escena_ejemplo(gen)
    T_det = T - (1 - np.mean(dg.FACTOR_HIERRO)) * T_hierro
    limpio = dg.measured_variance(gaussian_filter(T_det, np.mean(dg.PSF_SIGMA_RANGO_CM) / dg.VOXEL_CM), clean=True)
    medido, muones, _ = dg.agregar_ruido(limpio, aceptancia, rng, dias=DIAS)

    fig = plt.figure(figsize=(22, 10))
    g = fig.add_gridspec(2, 4)
    ejes = [fig.add_subplot(g[0, k]) for k in range(4)]
    inf.show_map(ejes[0], T, 'dens', 0.0, float(T.max()), "1. Simulated sample: $\\int dl/X_0$ (target)", [])
    inf.show_map(ejes[1], limpio / dg.VOXEL_CM, 'asd', *inf.ESCALA_ENTRADA_ASD,
                 "2. Expected ASD: detector response + PSF", [])
    inf.show_map(ejes[2], medido / dg.VOXEL_CM, 'asd', *inf.ESCALA_ENTRADA_ASD,
                 f"3. Network input: {DIAS:g} days of muons", [])
    inf.TIPOS_MAPA['n'] = ('viridis', False, 'Muons per pixel')
    inf.show_map(ejes[3], muones.astype(float), 'n', 0.0, float(np.percentile(muones, 99.5)),
                 "4. Second input: muons per pixel (acceptance)", [])

    # Calibración de la respuesta: regiones de material conocido (calibrar_respuesta.py)
    ax = fig.add_subplot(g[1, 0:2])
    t = np.linspace(0, 40, 300)
    respuesta = lambda x, A, gamma: cr.PEDESTAL + A * x ** gamma
    (A, gamma), _ = curve_fit(respuesta, cr.T, cr.V, p0=[2e-3, 0.56], sigma=cr.E)
    (k,), _ = curve_fit(lambda x, k: cr.PEDESTAL + k * x, cr.T, cr.V, sigma=cr.E)
    nombres = ["Lead 20 cm", "Lead 10 cm", "Lead 5 cm", "Concrete $\\mathcal{M}_2$", "Concrete $\\mathcal{M}_1$"]
    ax.errorbar(cr.T, cr.V * 1e3, yerr=cr.E * 1e3, fmt='o', color='k', ms=7, capsize=3, label='Measured (known material)')
    desplazamientos = [(8, -12), (8, -12), (8, -12), (10, 8), (10, -14)]
    for x, y, n, d in zip(cr.T, cr.V, nombres, desplazamientos):
        ax.annotate(n, (x, y * 1e3), textcoords='offset points', xytext=d, fontsize=9)
    ax.plot(t, respuesta(t, A, gamma) * 1e3, 'r-', lw=2, label=f'Fit: $V = V_0 + A\\,T^{{{gamma:.2f}}}$')
    ax.plot(t, (cr.PEDESTAL + k * t) * 1e3, 'b--', lw=1.5, label='Highland ($V \\propto T$), best linear fit')
    ax.axhline(cr.PEDESTAL * 1e3, color='0.5', ls=':', label='$V_0$: air (detector resolution)')
    ax.set_xlabel("Material crossed $T = \\int dl/X_0$")
    ax.set_ylabel("Measured variance [$10^{-3}$ rad$^2$]")
    ax.set_title("Detector response calibrated with known materials", fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9, loc='lower right')

    # Ruido: concreto homogéneo medido vs simulado con el mismo tiempo
    ax = fig.add_subplot(g[1, 2:4])
    simulado, sl = ruido_concreto(aceptancia, rng)
    real = np.load("datos_reales/Grosores_mortero_20d_20.6d_varrec_xy.npy")[sl].ravel()
    bins = np.linspace(0.5e-3, 4e-3, 36)
    ax.hist(real * 1e3, bins=bins * 1e3, density=True, alpha=0.6, color='tab:orange',
            label=f'Measured ({real.size} pixels)')
    ax.hist(simulado * 1e3, bins=bins * 1e3, density=True, histtype='step', lw=2, color='k',
            label='Simulated (same material and time)')
    ax.set_xlabel("Per-pixel measured variance [$10^{-3}$ rad$^2$]")
    ax.set_ylabel("Probability density")
    ax.set_title(f"Noise check: homogeneous concrete, {DIAS:g} days", fontsize=11)
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend(fontsize=9)
    print(f"real: mean {real.mean():.2e} std {real.std():.2e} | sim: mean {simulado.mean():.2e} std {simulado.std():.2e}")
    print(f"fit A = {A:.3e}, gamma = {gamma:.3f}; linear k = {k:.3e}")

    plt.tight_layout()
    plt.savefig(os.path.join(SALIDA, "training_data_and_detector_response.png"), dpi=150, bbox_inches='tight')
    plt.close()
