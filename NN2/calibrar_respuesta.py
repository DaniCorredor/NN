import numpy as np

# Calibra la respuesta del detector: V = PEDESTAL + A * T^gamma, con T = ∫dl/X0.
# Solo usa regiones homogéneas de grosor conocido (bloques de plomo libres y celdas de
# concreto sin barras). Las barras y oclusiones de Mallado/Grosores quedan para evaluar.

DATOS = "datos_reales"
PIXEL_CM = 0.5
MARGEN_CM = 1.0  # se recorta el borde de cada región para evitar la PSF
PEDESTAL = 1.5e-3
INV_X0 = {"plomo": 1.0 / 0.5612, "concreto": 0.086}

# (muestra, material, grosor_cm, rango_x_cm, rango_y_cm)
REGIONES = [
    ("plomo_bloques_84h_3.5d", "plomo", 20, (11, 19), (24, 34)),
    ("plomo_bloques_84h_3.5d", "plomo", 10, (30, 38), (20, 40)),
    ("plomo_bloques_84h_3.5d", "plomo", 5, (44, 54), (20, 40)),
    ("Mallado_mortero_20d_18.9d", "concreto", 21, (28, 34), (26, 32)),
    ("Mallado_mortero_20d_18.9d", "concreto", 21, (38, 46), (14, 22)),
    ("Mallado_mortero_20d_18.9d", "concreto", 21, (38, 46), (36, 40)),
]

def varianza_region(muestra, rango_x, rango_y):
    var = np.load(f"{DATOS}/{muestra}_var_xy.npy")
    conteos = np.load(f"{DATOS}/{muestra}_N_xy.npy")
    ix = slice(int((rango_x[0] + MARGEN_CM) / PIXEL_CM), int((rango_x[1] - MARGEN_CM) / PIXEL_CM))
    iy = slice(int((rango_y[0] + MARGEN_CM) / PIXEL_CM), int((rango_y[1] - MARGEN_CM) / PIXEL_CM))
    return (var[ix, iy] * conteos[ix, iy]).sum() / conteos[ix, iy].sum()

T = np.array([g * INV_X0[m] for _, m, g, _, _ in REGIONES])
V = np.array([varianza_region(s, x, y) for s, _, _, x, y in REGIONES])

gamma, log_A = np.polyfit(np.log(T), np.log(V - PEDESTAL), 1)
A = np.exp(log_A)
print(f"RESPUESTA_A = {A:.4e}\nRESPUESTA_GAMMA = {gamma:.4f}")
for (s, m, g, _, _), t, v in zip(REGIONES, T, V):
    ajuste = PEDESTAL + A * t**gamma
    print(f"{m:8s} {g:2d} cm  T={t:5.2f}  medido {v:.2e}  ajuste {ajuste:.2e}  ({100 * (ajuste - v) / v:+.0f}%)")

# Validación independiente: concreto de Grosores (49x30x25 cm), no usado en el ajuste
v_val = varianza_region("Grosores_mortero_20d_20.6d", (16, 40), (8, 48))
print(f"Validación Grosores 25 cm: medido {v_val:.2e}  predicho {PEDESTAL + A * (25 * 0.086)**gamma:.2e}")
