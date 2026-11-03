import os
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, binary_dilation
from tqdm import tqdm

# ==============================================================================
# CALIBRACIÓN EXPERIMENTAL (compartida por entrenamiento e inferencia)
# ==============================================================================
GRID = (112, 112, 120)      # ROI real: 56x56 cm en xy, 60 cm en z
VOXEL_CM = 0.5
P_EFF = 1.678
PEDESTAL = 1.5e-3           # varianza 3D medida en aire (plomo y Grosores) [rad^2]
# Respuesta medida del detector (media de theta^2): V = PEDESTAL + A * T^gamma (calibrar_respuesta.py).
# Crece ~sqrt(T), más lento que theta0^2 de Highland, por eso no basta un factor de escala.
# V14: recalibrada con la densidad real del concreto (pesos: Mallado ~2.0 g/cm3, Grosores ~1.8 g/cm3)
# y celdas del Mallado realmente sin barras (geometría M1 ajustada).
RESPUESTA_A = 2.1666e-3
RESPUESTA_GAMMA = 0.5284
K_RUIDO = 6.2               # varianza relativa de theta^2 3D por muón con el espectro del CSV
# 1/X0 [1/cm] del concreto y la arena, sorteado por muestra: X0 masa ~26.6 g/cm2 con densidad
# 1.8-2.1 g/cm3 (concreto, de los pesos de las muestras) y ~1.5-1.8 g/cm3 (arena)
INV_X0_CONCRETO = (0.068, 0.080)
INV_X0_ARENA = (0.055, 0.068)
PSF_SIGMA_CM = 0.722        # medida en Deconvolucion/psf.ipynb
RATE_MAP_PATH = "calibracion/rate_map_xy.npy"  # muones/día/píxel medidos (preparar_datos_reales.py)
# Aceptancia limpia (toma libre simulada) escalada a muones/día/píxel (preparar_recortada.py).
# El rate_map anterior suma las muestras reales y lleva impresas las barras del Mallado.
ACEPTANCIA_PATH = "calibracion/aceptancia_xy.npy"
# Estimador por columna: media de theta^2 descartando el 10% más alto (preparar_recortada.py)
FRACCION_RECORTE = 0.10
MAX_ZENITH_DEG = 15.0       # corte cenital de los datos reales (voxel_data_..._poca_angle_15.0)
NUM_ANGULOS = 64            # direcciones promediadas por muestra (pocas generan copias fantasma)
# Cenitales medidos (cielo cos^2*sin*cos x aceptancia del hodoscopio), preparar_datos_reales.py
ZENITH_HIST_PATH = "calibracion/zenith_hist.npz"

# --- Diferencias simulación/realidad medidas en Mallado y Grosores (se sortean por muestra) ---
# Contraste real de las barras / simulado: ~0.7 (Ø2 cm, Mallado) y 0.86 (Ø4-6 cm, Grosores);
# con la respuesta recalibrada sigue siendo ~0.65-0.8 (efecto real de PoCA, no de la densidad)
FACTOR_HIERRO = (0.55, 0.95)
# El ancho del pico de las barras reales es compatible con PSF de 0.6 a 1.0 cm
PSF_SIGMA_RANGO_CM = (0.6, 1.0)
# Ruido del estimador: banco de medias recortadas de theta^2 por muón REAL (aire, concreto, plomo),
# para cada N de 1 a N_MAX_BANCO (construir_banco_ruido). Incluye colas reales y el sesgo con N.
BANCO_RUIDO_PATH = "calibracion/banco_ruido.npz"
N_MAX_BANCO = 400
# Exceso de varianza cerca de los bordes del DETECTOR (aparece también con el hodoscopio vacío):
# se suma, sube en x = 0, y = 0 e y = 56 y no en x = 56 (calibrar_bordes.py). Su amplitud cambia
# entre tomas, así que se sortea por borde; la caída también, y el borde x = 56 recibe un mínimo
# para que la red no dependa de que un borde esté siempre limpio. El ajuste promedia tomas: en
# Grosores el borde y = 56 tiene ~1.9 veces el exceso ajustado, de ahí el máximo de la escala.
EXCESO_BORDES_PATH = "calibracion/exceso_bordes.npz"
EXCESO_ESCALA = (0.3, 2.2)
EXCESO_ESCALA_LAMBDA = (0.7, 1.4)
EXCESO_MINIMO = 0.15        # fracción del mayor exceso que recibe como mínimo cada borde
# PoCA concentra conteos en los objetos densos: N sube ~0.3 x el contraste local de la varianza
# (Mallado y Grosores), sin cambiar en bloques homogéneos
REDISTRIBUCION_N = (0.15, 0.45)
N_NORM = 256.0              # normalización del canal de muones: log1p(N) / log1p(N_NORM)

def media_recortada(theta2, fraccion=FRACCION_RECORTE):
    # Media de theta^2 sin el 'fraccion' más alto, con recorte FRACCIONARIO: se conservan
    # exactamente (1-fraccion)*N muones (el último con peso parcial), así el valor esperado
    # cambia suave con N. theta2: (N,) o (M, N) con N muones por fila.
    t = np.sort(np.atleast_2d(theta2), axis=1)
    n = t.shape[1]
    if n == 0:
        return np.zeros(t.shape[0])
    conservar = (1.0 - fraccion) * n
    pesos = np.clip(conservar - np.arange(n), 0.0, 1.0)
    resultado = (t * pesos).sum(axis=1) / conservar
    return resultado if np.ndim(theta2) > 1 else float(resultado[0])

def detector_response(integral_1_x0):
    T = np.clip(integral_1_x0, 0.0, None)
    return RESPUESTA_A * T**RESPUESTA_GAMMA

def limites_normalizacion():
    T_max = 100.0 * 0.568  # 100 cm de hierro
    max_var = detector_response(T_max) * 1.10
    max_density = T_max * 1.10
    return max_var, max_density

def sample_observation_days(rng=np.random):
    # Prioridad: imágenes de ~20 días; 20% de muestras cortas para robustez
    if rng.random() < 0.8:
        return rng.uniform(15.0, 25.0)
    return rng.uniform(3.0, 15.0)

def normalizar_muones(muones):
    return np.clip(np.log1p(muones) / np.log1p(N_NORM), 0.0, 1.0).astype(np.float32)

def normalizar_entrada(varianza_medida):
    # La media recortada del aire (~0.6 x PEDESTAL) queda bajo el pedestal: no se resta nada
    return np.clip(varianza_medida / limites_normalizacion()[0], 0.0, 1.0).astype(np.float32)

def construir_banco_ruido(muestras_path="calibracion/theta2_por_muon.npz", por_n=1024, semilla=0):
    # Para cada material y cada N: 'por_n' medias recortadas de N valores de theta^2 por muón
    # reales (normalizados a media 1). Medir = varianza esperada (media) x un valor del banco.
    rng = np.random.default_rng(semilla)
    muestras = np.load(muestras_path)
    banco = {}
    for clase in ('aire', 'concreto', 'plomo'):
        w = muestras[clase]
        tabla = np.ones((N_MAX_BANCO + 1, por_n), dtype=np.float32)
        for n in range(1, N_MAX_BANCO + 1):
            tabla[n] = media_recortada(w[rng.integers(0, len(w), (por_n, n))])
        banco[clase] = tabla
    np.savez_compressed(BANCO_RUIDO_PATH, **banco)
    return banco

_BANCO = None

def _factor_ruido(T, muones, rng):
    # Factor multiplicativo de la medición por píxel. El material decide el banco: aire (T~0),
    # concreto (T~1-3) o plomo/hierro grueso (T>~9), mezclados según T
    global _BANCO
    if _BANCO is None:
        datos = np.load(BANCO_RUIDO_PATH)
        _BANCO = np.stack([datos[c] for c in ('aire', 'concreto', 'plomo')])
    p_concreto = np.clip(T / 1.0, 0.0, 1.0)
    p_plomo = np.clip((T - 3.0) / 6.0, 0.0, 1.0)
    u = rng.random(T.shape)
    clase = np.where(u < p_plomo, 2, np.where(u < p_plomo + (1 - p_plomo) * p_concreto, 1, 0))
    n = np.clip(muones, 1, N_MAX_BANCO).astype(int)
    factor = _BANCO[clase, n, rng.integers(0, _BANCO.shape[2], T.shape)]
    # Más muones que el banco: misma media, dispersión reducida como 1/sqrt(N)
    grande = muones > N_MAX_BANCO
    if grande.any():
        media = _BANCO[clase[grande], N_MAX_BANCO].mean(axis=1)
        factor[grande] = media + (factor[grande] - media) * np.sqrt(N_MAX_BANCO / muones[grande])
    return factor

class PhysicalMuographyGenerator:
    def __init__(self, grid_size=(100, 100, 100), voxel_size_cm=1.0, muones_csv_path=None):
        self.grid_size = grid_size
        self.voxel_size_cm = voxel_size_cm
        self.volume = np.zeros(grid_size, dtype=np.float32)

        self.materials = {
            'air': 0.0,
            'concrete': 0.086,
            'sand': 0.083,
            'iron': 0.568
        }

        if muones_csv_path and os.path.exists(muones_csv_path):
            df = pd.read_csv(muones_csv_path)
            mean_inv_p2 = np.mean(1.0 / (df['exit_p_GeV'].values ** 2))
            self.p_eff_gev = 1.0 / np.sqrt(mean_inv_p2)
            print(f"Momento Efectivo (p_eff) calibrado: {self.p_eff_gev:.3f} GeV/c")
        else:
            self.p_eff_gev = 4.0
            print("CSV no encontrado. Usando p=4.0 GeV/c por defecto.")

        hist = np.load(ZENITH_HIST_PATH)
        self.zenith_bordes, self.zenith_prob = hist['bordes'], hist['prob']
        self.pivot_z = grid_size[2] / 2.0

    def reset(self, background='air'):
        self.volume.fill(self.materials[background])
        self.pivot_z = self.grid_size[2] / 2.0

    def add_block(self, center, size, material='concrete'):
        cx, cy, cz = center
        sx, sy, sz = size
        x_min, x_max = max(0, cx - sx//2), min(self.grid_size[0], cx + sx//2)
        y_min, y_max = max(0, cy - sy//2), min(self.grid_size[1], cy + sy//2)
        z_min, z_max = max(0, cz - sz//2), min(self.grid_size[2], cz + sz//2)
        self.volume[x_min:x_max, y_min:y_max, z_min:z_max] = self.materials[material]
        # PoCA ubica la dispersión dentro del objeto: las proyecciones inclinadas pivotan en su centro
        self.pivot_z = (z_min + z_max) / 2.0
        return (x_min, x_max), (y_min, y_max), (z_min, z_max)

    def _local_grid(self, lo, hi):
        # Sub-volumen que contiene la figura (evita recorrer toda la grilla)
        lo = np.maximum(np.floor(lo).astype(int), 0)
        hi = np.minimum(np.ceil(hi).astype(int) + 1, self.grid_size)
        region = tuple(slice(l, h) for l, h in zip(lo, hi))
        return region, np.ogrid[region]

    def add_bar(self, start, end, radius, material='iron'):
        # Cilindro finito entre start y end (coordenadas en vóxeles)
        a, b = np.asarray(start, float), np.asarray(end, float)
        region, (X, Y, Z) = self._local_grid(np.minimum(a, b) - radius, np.maximum(a, b) + radius)
        ab = b - a
        t = ((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (Z - a[2]) * ab[2]) / max(ab @ ab, 1e-9)
        t = np.clip(t, 0.0, 1.0)
        d2 = (X - a[0] - t * ab[0])**2 + (Y - a[1] - t * ab[1])**2 + (Z - a[2] - t * ab[2])**2
        dentro = d2 <= radius**2
        if material == 'air':
            # Tubos vacíos: no cortan las barras de hierro que cruzan
            dentro &= self.volume[region] < self.materials['iron']
        self.volume[region][dentro] = self.materials[material]

    def add_void(self, center, radius):
        # Solo vacía la matriz (concreto/arena): no corta barras ni se sale del bloque
        c = np.asarray(center, float)
        region, (X, Y, Z) = self._local_grid(c - radius, c + radius)
        sub = self.volume[region]
        mask = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2 <= radius**2
        sub[mask & (sub > 0) & (sub < self.materials['iron'])] = self.materials['air']

    def iron_volume(self):
        return np.where(self.volume == self.materials['iron'], self.materials['iron'], 0.0).astype(np.float32)

    def bar_mask(self, dilatacion_px=2):
        # Columnas con hierro (vista vertical), ensanchadas un poco: zonas donde se pondera la pérdida
        columnas = (self.volume == self.materials['iron']).any(axis=2)
        if dilatacion_px == 0:
            return columnas.astype(np.uint8)
        return binary_dilation(columnas, iterations=dilatacion_px).astype(np.uint8)

    def get_sheared_projection(self, capas, pad, zenith_deg, azimuth_deg):
        # Proyección inclinada: cada capa z se desplaza tan(theta)*(z - pivot_z) vóxeles (vecino más cercano).
        # capas: volumen con margen, ordenado (z, canal, x, y) para que cada capa sea contigua en memoria
        theta_rad = np.radians(zenith_deg)
        phi_rad = np.radians(azimuth_deg)

        dx = np.tan(theta_rad) * np.cos(phi_rad)
        dy = np.tan(theta_rad) * np.sin(phi_rad)

        nx, ny, nz = self.grid_size
        zc = self.pivot_z
        proyeccion = np.zeros((capas.shape[1], nx, ny), dtype=np.float32)
        for z in range(nz):
            sx = pad + int(round(dx * (z - zc)))
            sy = pad + int(round(dy * (z - zc)))
            proyeccion += capas[z, :, sx:sx + nx, sy:sy + ny]

        path_length_correction = self.voxel_size_cm / np.cos(theta_rad)
        return proyeccion * path_length_correction

    def simulate_transmission(self, num_angles=NUM_ANGULOS):
        # Devuelve (∫1/X0 total, ∫1/X0 solo del hierro): las dos con las mismas direcciones
        proyeccion_total = np.zeros((2, self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        # Cenitales de la distribución medida: ya incluye flujo, área efectiva y aceptancia,
        # así que cada dirección pesa lo mismo. Azimut uniforme (medido: +-6% entre octantes)
        idx = np.random.choice(len(self.zenith_prob), size=num_angles, p=self.zenith_prob)
        thetas = np.random.uniform(self.zenith_bordes[idx], self.zenith_bordes[idx + 1])
        phis = np.random.uniform(0, 360, num_angles)

        # Margen para el peor caso: pivote en un extremo de la grilla
        pad = int(np.ceil(np.tan(np.radians(MAX_ZENITH_DEG)) * self.grid_size[2])) + 1
        canales = np.stack([self.volume, self.iron_volume()])                     # (canal, x, y, z)
        capas = np.ascontiguousarray(np.pad(canales, ((0, 0), (pad, pad), (pad, pad), (0, 0))).transpose(3, 0, 1, 2))

        for theta, phi in zip(thetas, phis):
            proyeccion_total += self.get_sheared_projection(capas, pad, theta, phi)

        integral_1_x0 = proyeccion_total / num_angles
        return integral_1_x0[0], integral_1_x0[1]

def measured_variance(integral_1_x0, muones=None, clean=False):
    # Lo que mide el detector: respuesta calibrada al material + resolución (pedestal)
    varianza_esperada = detector_response(integral_1_x0) + PEDESTAL

    if clean:
        return varianza_esperada.astype(np.float32)

    # Cada píxel promedia N valores de theta^2: varianza relativa K_RUIDO / N
    forma = np.maximum(muones, 1) / K_RUIDO
    ruido = np.random.gamma(shape=forma, scale=1.0 / forma)
    varianza_medida = np.where(muones > 0, varianza_esperada * ruido, 0.0)

    return varianza_medida.astype(np.float32)

_EXCESO = None

def exceso_bordes(forma, rng):
    # sum_b a_b * u_b * exp(-d_b / (lambda_b * v_b)), d_b = distancia al borde b (x=0, x=56, y=0, y=56)
    global _EXCESO
    if _EXCESO is None:
        datos = np.load(EXCESO_BORDES_PATH)
        _EXCESO = (datos["a"], datos["lam"])
    a, lam = _EXCESO
    a = np.maximum(a, EXCESO_MINIMO * a.max()) * rng.uniform(*EXCESO_ESCALA, 4)
    lam = lam * rng.uniform(*EXCESO_ESCALA_LAMBDA, 4)
    x = (np.arange(forma[0]) + 0.5) * VOXEL_CM
    y = (np.arange(forma[1]) + 0.5) * VOXEL_CM
    lado_x, lado_y = forma[0] * VOXEL_CM, forma[1] * VOXEL_CM
    en_x = a[0] * np.exp(-x / lam[0]) + a[1] * np.exp(-(lado_x - x) / lam[1])
    en_y = a[2] * np.exp(-y / lam[2]) + a[3] * np.exp(-(lado_y - y) / lam[3])
    return (en_x[:, None] + en_y[None, :]).astype(np.float32)

def agregar_ruido(varianza_limpia, aceptancia, rng=np.random, dias=None):
    # Medición de 'dias' días (media recortada de theta^2 por columna) a partir de la varianza
    # esperada con PSF (escala de la media). Se llama al cargar cada muestra en el entrenamiento:
    # la red nunca ve dos veces el mismo ruido. aceptancia: muones/día/píxel (ACEPTANCIA_PATH).
    if rng is np.random:
        # Generador con semilla tomada del estado global (conjunto de prueba reproducible)
        rng = np.random.default_rng(np.random.randint(2**31))
    if dias is None:
        dias = sample_observation_days(rng)

    # PoCA concentra conteos en los objetos densos: N sigue el contraste local de la varianza
    contraste_local = varianza_limpia / np.maximum(gaussian_filter(varianza_limpia, 2.0 / VOXEL_CM), 1e-9) - 1.0
    redistribucion = np.clip(1.0 + rng.uniform(*REDISTRIBUCION_N) * contraste_local, 0.3, None)
    muones = rng.poisson(dias * aceptancia * redistribucion)

    # Exceso de los bordes del detector (coordenadas del detector, no de la muestra)
    varianza = varianza_limpia + exceso_bordes(varianza_limpia.shape, rng)

    # Material de cada píxel (para elegir el banco de ruido): se invierte la respuesta del detector
    T = (np.clip(varianza_limpia - PEDESTAL, 0.0, None) / RESPUESTA_A) ** (1.0 / RESPUESTA_GAMMA)
    varianza_medida = np.where(muones > 0, varianza * _factor_ruido(T, muones, rng), 0.0)
    return varianza_medida.astype(np.float32), muones, dias

def get_enmallado_config():
    n = lambda: np.random.randint(2, 7)
    if np.random.rand() < 0.25:
        return (n(), 0) if np.random.rand() < 0.5 else (0, n())
    return n(), n()

def sample_block(gen):
    # Bloque anfitrión con tamaños cercanos a las muestras reales (47x47x21 y 49x30x25 cm);
    # la muestra se coloca en el hodoscopio sin posición fija
    gx, gy, gz = gen.grid_size
    sx, sy = (np.random.uniform(25, 60, size=2) / VOXEL_CM).astype(int)
    sz = int(np.random.uniform(18, 28) / VOXEL_CM)
    shift = int(8 / VOXEL_CM)
    cx = gx // 2 + np.random.randint(-shift, shift + 1)
    cy = gy // 2 + np.random.randint(-shift, shift + 1)
    cz = gz // 2 + np.random.randint(-int(10 / VOXEL_CM), int(10 / VOXEL_CM) + 1)
    material = 'sand' if np.random.rand() < 0.2 else 'concrete'
    return (cx, cy, cz), (sx, sy, sz), material

def _random_point(bounds, margin):
    return np.array([np.random.uniform(lo + margin, hi - margin) for lo, hi in bounds])

def _posiciones_irregulares(lo, hi, n, margen):
    # n posiciones entre lo+margen y hi-margen: 40% regulares, 60% con huecos distintos (M1: 4, 6, 8, 10 cm)
    if n == 1:
        return [np.random.uniform(lo + margen, hi - margen)]
    if np.random.rand() < 0.4:
        return list(np.linspace(lo + margen, hi - margen, n))
    huecos = np.random.uniform(1.0, 3.5, n - 1)
    return list(lo + margen + np.concatenate([[0], np.cumsum(huecos)]) / huecos.sum() * (hi - lo - 2 * margen))

def add_rebar_mesh(gen, bounds):
    # Malla tipo Mallado (M1): espaciado irregular, cada dirección con su diámetro (1-3 cm) y
    # profundidad, profundidad que puede variar barra a barra, barras que sobresalen del bloque
    # y alguna barra faltante
    (x0, x1), (y0, y1), (z0, z1) = bounds
    for eje, n in enumerate(get_enmallado_config()):
        if n == 0:
            continue
        r = np.random.uniform(0.5, 1.5) / VOXEL_CM
        # eje 0: barras a lo largo de x, repartidas en y; eje 1: a lo largo de y, repartidas en x
        lo, hi = (y0, y1) if eje == 0 else (x0, x1)
        a0, a1 = (x0, x1) if eje == 0 else (y0, y1)
        posiciones = _posiciones_irregulares(lo, hi, n, margen=np.random.uniform(2, 9) / VOXEL_CM)
        z_base = np.random.uniform(z0 + r, z1 - r)
        dz = np.random.uniform(0, 8) / VOXEL_CM if np.random.rand() < 0.5 else 0.0
        extremo = np.random.uniform(0, 5) / VOXEL_CM if np.random.rand() < 0.5 else -np.random.uniform(1, 4) / VOXEL_CM
        for pos in posiciones:
            if np.random.rand() < 0.1:
                continue
            z = np.clip(z_base + np.random.uniform(-dz / 2, dz / 2), z0 + r, z1 - r)
            if eje == 0:
                gen.add_bar((a0 - extremo, pos, z), (a1 + extremo, pos, z), r)
            else:
                gen.add_bar((pos, a0 - extremo, z), (pos, a1 + extremo, z), r)

def add_thickness_row(gen, bounds):
    # Tipo Grosores (M2): fila de 2-5 barras paralelas, mismo largo (6-15 cm) y profundidad,
    # diámetros distintos (1-6 cm); en 60% de los casos cada una sigue con un tubo vacío
    (x0, x1), (y0, y1), (z0, z1) = bounds
    eje = np.random.randint(2)
    lo, hi = (y0, y1) if eje == 0 else (x0, x1)       # dirección en que se reparten
    a0, a1 = (x0, x1) if eje == 0 else (y0, y1)       # dirección de las barras
    radios = np.random.uniform(0.5, 3.0, np.random.randint(2, 6)) / VOXEL_CM
    if np.random.rand() < 0.5:
        radios = np.sort(radios)
    largo = min(np.random.uniform(6, 15) / VOXEL_CM, 0.6 * (a1 - a0))
    hueco = np.random.uniform(3, 7) / VOXEL_CM if np.random.rand() < 0.6 else 0.0
    hueco = min(hueco, 0.3 * (a1 - a0))
    inicio = np.random.uniform(a0 + 2 / VOXEL_CM, max(a0 + 2 / VOXEL_CM, a1 - largo - hueco - 2 / VOXEL_CM))
    z = np.random.uniform(z0 + radios.max(), max(z0 + radios.max(), z1 - radios.max()))
    pos = lo + np.random.uniform(2, 6) / VOXEL_CM
    for r in radios:
        pos += r
        if pos + r > hi - 1 / VOXEL_CM:
            break
        p1, p2, p3 = inicio, inicio + largo, inicio + largo + hueco
        if eje == 0:
            gen.add_bar((p1, pos, z), (p2, pos, z), r)
            if hueco > 0:
                gen.add_bar((p2, pos, z), (p3, pos, z), r, material='air')
        else:
            gen.add_bar((pos, p1, z), (pos, p2, z), r)
            if hueco > 0:
                gen.add_bar((pos, p2, z), (pos, p3, z), r, material='air')
        pos += r + np.random.uniform(4, 12) / VOXEL_CM

def add_pvc_tubes(gen, bounds):
    # Tubos vacíos de manipulación (ambas muestras reales los tienen): 1-2 a lo largo del bloque,
    # cerca del fondo y de un borde
    (x0, x1), (y0, y1), (z0, z1) = bounds
    eje = np.random.randint(2)
    lo, hi = (y0, y1) if eje == 0 else (x0, x1)
    a0, a1 = (x0, x1) if eje == 0 else (y0, y1)
    for lado in np.random.permutation([0, 1])[:np.random.randint(1, 3)]:
        r = np.random.uniform(0.75, 1.5) / VOXEL_CM
        d = np.random.uniform(2, 8) / VOXEL_CM
        pos = lo + d if lado == 0 else hi - d
        z = min(z0 + np.random.uniform(3, 7) / VOXEL_CM, z1 - r)
        if eje == 0:
            gen.add_bar((a0, pos, z), (a1, pos, z), r, material='air')
        else:
            gen.add_bar((pos, a0, z), (pos, a1, z), r, material='air')

def add_isolated_bars(gen, bounds, n_bars):
    # Barras sueltas: largo 5-40 cm, diámetro 1-6 cm; mitad alineadas con x o y,
    # mitad con azimut libre; 30% inclinadas hasta +-15° respecto a la horizontal
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    for _ in range(n_bars):
        r = np.random.uniform(0.5, 3.0) / VOXEL_CM
        largo = np.random.uniform(5, 40) / VOXEL_CM
        phi = np.random.choice([0.0, np.pi / 2]) if np.random.rand() < 0.5 else np.random.uniform(0, np.pi)
        inclinacion = np.radians(np.random.uniform(-15, 15)) if np.random.rand() < 0.3 else 0.0
        d = np.array([np.cos(phi) * np.cos(inclinacion), np.sin(phi) * np.cos(inclinacion), np.sin(inclinacion)])
        centro = _random_point(bounds, r)
        p1 = np.clip(centro - d * largo / 2, lo + r, hi - r)
        p2 = np.clip(centro + d * largo / 2, lo + r, hi - r)
        gen.add_bar(p1, p2, r)

def add_void_clusters(gen, bounds, n_clusters):
    # Oclusión principal (1-7 cm de diámetro) + 0-4 esferas menores solapadas: huecos irregulares
    for _ in range(n_clusters):
        r0 = np.random.uniform(0.5, 3.5) / VOXEL_CM
        centro = _random_point(bounds, r0)
        gen.add_void(centro, r0)
        for _ in range(np.random.randint(0, 5)):
            offset = np.random.uniform(-0.8, 0.8, size=3) * r0
            gen.add_void(centro + offset, np.random.uniform(0.4, 0.9) * r0)

def build_scene(gen, i):
    # Distribución (prioridad barras): aire 4%, barras en aire 4%, bloque 8%, malla 32%,
    # grosores 22%, barras 18%, barras + oclusiones 6%, oclusiones 6%.
    # 40% de los bloques con barras llevan tubos vacíos de manipulación.
    opcion = i % 100
    gen.materials['concrete'] = np.random.uniform(*INV_X0_CONCRETO)
    gen.materials['sand'] = np.random.uniform(*INV_X0_ARENA)
    center, size, material = sample_block(gen)
    if opcion < 4:
        return 'aire'
    if opcion < 8:
        bounds = gen.add_block(center, size, material='air')
        add_isolated_bars(gen, bounds, np.random.randint(1, 7))
        return 'barras_aire'
    bounds = gen.add_block(center, size, material=material)
    if opcion < 16:
        escena = 'bloque'
    elif opcion < 48:
        add_rebar_mesh(gen, bounds)
        escena = 'malla'
    elif opcion < 70:
        add_thickness_row(gen, bounds)
        escena = 'grosores'
    elif opcion < 88:
        add_isolated_bars(gen, bounds, np.random.randint(1, 7))
        escena = 'barras'
    elif opcion < 94:
        add_isolated_bars(gen, bounds, np.random.randint(1, 7))
        add_void_clusters(gen, bounds, np.random.randint(1, 4))
        return 'barras_oclusiones'
    else:
        add_void_clusters(gen, bounds, np.random.randint(1, 4))
        return 'oclusiones'
    if np.random.rand() < 0.4:
        add_pvc_tubes(gen, bounds)
    return escena

def generate_deterministic_dataset(num_samples=10000, output_dir="datasets/dataset_muografia_V11", csv_path=None,
                                   guardar_ruido=False, semilla=None):
    # Cada muestra guarda la varianza esperada SIN ruido (x_limpio): el ruido se agrega al cargarla.
    # Con guardar_ruido=True (conjunto de prueba) se guarda además una medición fija (x, muones, dias).
    os.makedirs(output_dir, exist_ok=True)
    if len(glob.glob(os.path.join(output_dir, "sample_*.npz"))) >= num_samples:
        # Las muestras guardan la varianza SIN ruido: el ruido nuevo se aplica al cargarlas
        print(f"'{output_dir}' ya tiene {num_samples} muestras: no se regenera")
        return
    if semilla is not None:
        np.random.seed(semilla)
    grid = GRID
    gen = PhysicalMuographyGenerator(grid_size=grid, voxel_size_cm=VOXEL_CM, muones_csv_path=csv_path)
    aceptancia = np.load(ACEPTANCIA_PATH)

    print(f"Generando {num_samples} muestras rígidamente distribuidas en '{output_dir}'...")

    for i in tqdm(range(num_samples)):
        gen.reset(background='air')
        escena = build_scene(gen, i)

        T, T_hierro = gen.simulate_transmission()

        # El detector ve las barras con menos contraste que la proyección ideal (FACTOR_HIERRO)
        T_detector = T - (1.0 - np.random.uniform(*FACTOR_HIERRO)) * T_hierro

        # El filtro gaussiano simula la PSF del detector de entrada
        detector_psf_sigma = np.random.uniform(*PSF_SIGMA_RANGO_CM) / VOXEL_CM
        x_limpio = measured_variance(gaussian_filter(T_detector, sigma=detector_psf_sigma), clean=True)

        # Target Varianza: lo que mediría el detector con N -> infinito y sin PSF
        y_target = measured_variance(T_detector, clean=True)

        # Target Densidad: integral física (con el hierro real)
        y_density = T.astype(np.float32)

        datos = dict(x_limpio=x_limpio, y=y_target, density=y_density, mascara=gen.bar_mask(),
                     hierro=gen.bar_mask(dilatacion_px=0), T_hierro=T_hierro.astype(np.float16), scene=escena)
        if guardar_ruido:
            datos['x'], muones, datos['dias'] = agregar_ruido(x_limpio, aceptancia)
            datos['muones'] = muones.astype(np.uint16)

        filename = os.path.join(output_dir, f"sample_{i:05d}.npz")
        np.savez_compressed(filename, **datos)

def generate_test_dataset(num_samples=500, output_dir="datasets/dataset_muografia_Test_V12", csv_path=None):
    generate_deterministic_dataset(num_samples=num_samples, output_dir=output_dir, csv_path=csv_path,
                                   guardar_ruido=True, semilla=12345)

if __name__ == "__main__":
    RUTA_CSV = "Momento_Muones_filtrado.csv"

    generate_deterministic_dataset(
        num_samples=10000,
        output_dir="datasets/dataset_muografia_V11",
        csv_path=RUTA_CSV
    )

    generate_test_dataset(
        num_samples=500,
        output_dir="datasets/dataset_muografia_Test_V12",
        csv_path=RUTA_CSV
    )
