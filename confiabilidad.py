import numpy as np
from scipy.ndimage import gaussian_filter

from data_generator import EXCESO_BORDES_PATH, FACTOR_HIERRO, PSF_SIGMA_RANGO_CM, VOXEL_CM, detector_response

# Mapa de confiabilidad (coordenadas del detector, fijo para todas las tomas): se muestran los
# resultados solo donde la densidad relativa de eventos (RED) es al menos RED_MINIMA. La RED sale de
# una simulación Monte Carlo de las trayectorias que cruzan cada vóxel de la ROI
# (../scatter/mapa_hit_count_3d.npy, proyectada en X-Y y normalizada a 1 en el centro), así que
# depende solo de la geometría del hodoscopio y vale para cualquier toma.
# Por qué ~0.35-0.4: el ruido por píxel crece como 1/sqrt(RED) (1.7 veces el del centro en 0.35,
# igual a lo medido); en los datos reales con V16, el hierro falso de las mallas sube a 23-41% por
# debajo de RED ~ 0.3-0.4 (figura 04).
# exceso_detector y senal_barra_referencia se siguen usando en el ajuste rígido (bandas del detector).

RED_PATH = "calibracion/densidad_relativa_eventos_xy.npy"
RED_MINIMA = 0.35
DIAMETRO_REFERENCIA_CM = 2.0
CONCRETO_T = 21 * 2.0 / 26.6          # Mallado: 21 cm de concreto de 2.0 g/cm3 (∫dl/X0)
X0_HIERRO_CM, INV_X0_CONCRETO_CM = 1.757, 2.0 / 26.6

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

def densidad_relativa():
    return np.load(RED_PATH)

def mapa_confiable(forma=(112, 112)):
    # True donde la salida de la red (y el ajuste rígido) es confiable
    return densidad_relativa() >= RED_MINIMA

if __name__ == "__main__":
    red = densidad_relativa()
    m = mapa_confiable()
    c = (np.arange(112) + 0.5) * VOXEL_CM
    cols = np.where(m[56])[0]
    print(f"RED >= {RED_MINIMA}: {m.mean():.2f} del detector; en el centro, de {c[cols[0]]:.1f} a {c[cols[-1]]:.1f} cm")
