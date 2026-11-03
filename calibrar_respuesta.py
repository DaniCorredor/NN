import numpy as np
from scipy.optimize import curve_fit

# Calibra la respuesta del detector (media de theta^2): V = PEDESTAL + A * T^gamma, con T = ∫dl/X0.
# Solo usa regiones homogéneas de grosor conocido: bloques de plomo libres y concreto SIN barras
# según las geometrías M1/M2 ajustadas (las celdas de V12/V13 tenían barras). La densidad del
# concreto sale de los pesos de las muestras: Mallado ~2.0 g/cm3 (103 kg - ~12 kg de hierro),
# Grosores ~1.8 g/cm3 (69 kg - ~3.5 kg de hierro); X0 másico del concreto ~26.6 g/cm2.

DATOS = "datos_reales"
PIXEL_CM = 0.5
PEDESTAL = 1.5e-3
X0_CONCRETO_MASA = 26.6

# (muestra, ∫dl/X0, rango_x_cm, rango_y_cm)
REGIONES = [
    ("plomo_bloques_84h_3.5d", 20 / 0.5612, (12, 18), (25, 33)),
    ("plomo_bloques_84h_3.5d", 10 / 0.5612, (31, 37), (22, 38)),
    ("plomo_bloques_84h_3.5d", 5 / 0.5612, (45, 53), (22, 38)),
    ("Grosores_mortero_20d_20.6d", 25 * 1.82 / X0_CONCRETO_MASA, (15, 39), (30, 35)),
    ("Mallado_mortero_20d_18.9d", 21 * 2.0 / X0_CONCRETO_MASA, (35, 41.5), (30, 34)),
]

def region(muestra, rango_x, rango_y):
    var = np.load(f"{DATOS}/{muestra}_var_xy.npy")
    sub = var[int(rango_x[0] / PIXEL_CM):int(rango_x[1] / PIXEL_CM), int(rango_y[0] / PIXEL_CM):int(rango_y[1] / PIXEL_CM)]
    return sub.mean(), sub.std() / np.sqrt(sub.size)

T = np.array([t for _, t, _, _ in REGIONES])
V, E = np.array([region(s, x, y) for s, _, x, y in REGIONES]).T

respuesta = lambda t, A, gamma: PEDESTAL + A * t**gamma
(A, gamma), _ = curve_fit(respuesta, T, V, p0=[2e-3, 0.56], sigma=E)
print(f"RESPUESTA_A = {A:.4e}\nRESPUESTA_GAMMA = {gamma:.4f}")
for (s, t, _, _), v in zip(REGIONES, V):
    ajuste = respuesta(t, A, gamma)
    print(f"{s:28s} T={t:5.2f}  medido {v:.2e}  ajuste {ajuste:.2e}  ({100 * (ajuste - v) / v:+.0f}%)")
