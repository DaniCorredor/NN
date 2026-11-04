import numpy as np
from scipy.ndimage import gaussian_filter

from data_generator import (ACEPTANCIA_PATH, EXCESO_BORDES_PATH, FACTOR_HIERRO, PSF_SIGMA_RANGO_CM, VOXEL_CM,
                            detector_response)

# Mapa de confiabilidad (coordenadas del detector, fijo para todas las tomas). Un píxel es poco
# confiable si:
#   1. el exceso que agrega el detector (bandas en y ~ 10 y y > 49.5 cm, subida hacia x = 0;
#      calibrar_bordes.py) supera la mitad de la señal de una barra de Ø2 cm en el concreto de las
#      muestras: ahí una banda del detector se confunde con una barra (V15/V16 marcaban hierro ahí);
#   2. o en 20 días llegan menos de MUONES_MINIMOS muones (bordes de la aceptancia).
# El criterio no usa las muestras: solo la calibración del detector y la respuesta simulada.

DIAMETRO_REFERENCIA_CM = 2.0
CONCRETO_T = 21 * 2.0 / 26.6          # Mallado: 21 cm de concreto de 2.0 g/cm3 (∫dl/X0)
X0_HIERRO_CM, INV_X0_CONCRETO_CM = 1.757, 2.0 / 26.6
FRACCION_SENAL = 0.5
DIAS_REFERENCIA = 20.0
MUONES_MINIMOS = 20

def senal_barra_referencia():
    # Pico de varianza (escala de la media) de una barra de Ø2 cm dentro del concreto, con el
    # contraste real (FACTOR_HIERRO medio) y la PSF media
    r = DIAMETRO_REFERENCIA_CM / 2
    d = (np.arange(-40, 41) + 0.0) * VOXEL_CM
    cuerda = 2 * np.sqrt(np.clip(r ** 2 - d ** 2, 0, None))
    dT = cuerda * (1 / X0_HIERRO_CM - INV_X0_CONCRETO_CM) * np.mean(FACTOR_HIERRO)
    perfil = detector_response(CONCRETO_T + dT) - detector_response(CONCRETO_T)
    return float(gaussian_filter(perfil, np.mean(PSF_SIGMA_RANGO_CM) / VOXEL_CM).max())

def exceso_detector(forma=(112, 112)):
    # Exceso nominal (amplitudes ajustadas, sin sorteo), escala de la media
    p = {k: float(v) for k, v in np.load(EXCESO_BORDES_PATH).items()}
    x = (np.arange(forma[0]) + 0.5) * VOXEL_CM
    y = (np.arange(forma[1]) + 0.5) * VOXEL_CM
    en_x = p["aL"] * np.exp(-x / p["lL"]) + p["aR"] * np.exp(-(forma[0] * VOXEL_CM - x) / p["lR"])
    en_y = (p["aB"] * np.exp(-0.5 * ((y - p["yB"]) / p["wB"]) ** 2) +
            p["aT"] / (1 + np.exp(-(y - p["yT"]) / p["wT"])))
    return en_x[:, None] + en_y[None, :]

def mapa_confiable(forma=(112, 112)):
    # True donde la salida de la red (y el ajuste rígido) es confiable
    banda = exceso_detector(forma) > FRACCION_SENAL * senal_barra_referencia()
    pocos_muones = np.load(ACEPTANCIA_PATH) * DIAS_REFERENCIA < MUONES_MINIMOS
    return ~(banda | pocos_muones)

def figura(path):
    # Figura para la tesis (orientación de inference_unet_2_real.TRANSPONER)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import inference_unet_2_real as inf
    exceso = exceso_detector()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    inf.TIPOS_MAPA['excess'] = ('magma', False, 'Detector excess variance [rad$^2$]')
    inf.show_map(axes[0], exceso, 'excess', 0.0, float(np.percentile(exceso, 99)),
                 "Detector excess (empty hodoscope + air regions)", [])
    inf.TIPOS_MAPA['ok'] = ('Greens', False, 'Reliable (1) / reduced reliability (0)')
    inf.show_map(axes[1], mapa_confiable().astype(float), 'ok', 0.0, 1.0,
                 f"Reliable zone: excess < {FRACCION_SENAL:g} × Ø{DIAMETRO_REFERENCIA_CM:g} cm bar signal", [])
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import os
    os.makedirs("resultados/V16/postprocessing", exist_ok=True)
    figura("resultados/V16/postprocessing/reliability_map.png")
    s = senal_barra_referencia()
    m = mapa_confiable()
    c = (np.arange(112) + 0.5) * VOXEL_CM
    print(f"Señal de una barra de Ø2 cm: {s:.2e} rad^2 (umbral del exceso: {FRACCION_SENAL * s:.2e})")
    print(f"Fracción confiable del detector: {m.mean():.2f}")
    print("Confiable en y (x = 30 cm):", "".join("#" if v else "." for v in m[60, ::2]))
    print("Confiable en x (y = 30 cm):", "".join("#" if v else "." for v in m[::2, 60]))
    filas = np.where(m.any(axis=1))[0]
    cols = np.where(m[60])[0]
    print(f"x confiable desde {c[filas[0]]:.1f} cm; en y = 30 cm, y confiable {c[cols[0]]:.1f}-{c[cols[-1]]:.1f} cm")
