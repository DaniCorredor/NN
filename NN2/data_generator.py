import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ==============================================================================
# CALIBRACIÓN EXPERIMENTAL (compartida por entrenamiento e inferencia)
# ==============================================================================
GRID = (112, 112, 120)      # ROI real: 56x56 cm en xy, 60 cm en z
VOXEL_CM = 0.5
P_EFF = 1.678
PEDESTAL = 1.5e-3           # varianza 3D medida en aire (plomo y Grosores) [rad^2]
# Respuesta medida del detector: V = PEDESTAL + A * T^gamma (calibrar_respuesta.py).
# Crece ~sqrt(T), más lento que theta0^2 de Highland, por eso no basta un factor de escala.
RESPUESTA_A = 1.9592e-3
RESPUESTA_GAMMA = 0.5636
K_RUIDO = 6.2               # varianza relativa de theta^2 3D por muón con el espectro del CSV
PSF_SIGMA_CM = 0.722        # medida en Deconvolucion/psf.ipynb
RATE_MAP_PATH = "calibracion/rate_map_xy.npy"  # muones/día/píxel medidos (preparar_datos_reales.py)
MAX_ZENITH_DEG = 15.0       # corte cenital de los datos reales (voxel_data_..._poca_angle_15.0)
NUM_ANGULOS = 64            # direcciones promediadas por muestra (pocas generan copias fantasma)
# Cenitales medidos (cielo cos^2*sin*cos x aceptancia del hodoscopio), preparar_datos_reales.py
ZENITH_HIST_PATH = "calibracion/zenith_hist.npz"

def detector_response(integral_1_x0):
    T = np.clip(integral_1_x0, 0.0, None)
    return RESPUESTA_A * T**RESPUESTA_GAMMA

def limites_normalizacion():
    T_max = 100.0 * 0.568  # 100 cm de hierro
    max_var = detector_response(T_max) * 1.10
    max_density = T_max * 1.10
    return max_var, max_density

def sample_observation_days():
    # Prioridad: imágenes de ~20 días; 20% de muestras cortas para robustez
    if np.random.rand() < 0.8:
        return np.random.uniform(15.0, 25.0)
    return np.random.uniform(3.0, 15.0)

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
        self.volume[region][d2 <= radius**2] = self.materials[material]

    def add_void(self, center, radius):
        # Solo vacía la matriz (concreto/arena): no corta barras ni se sale del bloque
        c = np.asarray(center, float)
        region, (X, Y, Z) = self._local_grid(c - radius, c + radius)
        sub = self.volume[region]
        mask = (X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2 <= radius**2
        sub[mask & (sub > 0) & (sub < self.materials['iron'])] = self.materials['air']

    def get_sheared_projection(self, padded_volume, pad, zenith_deg, azimuth_deg):
        # Proyección inclinada: cada capa z se desplaza tan(theta)*(z - pivot_z) vóxeles (vecino más cercano)
        theta_rad = np.radians(zenith_deg)
        phi_rad = np.radians(azimuth_deg)

        dx = np.tan(theta_rad) * np.cos(phi_rad)
        dy = np.tan(theta_rad) * np.sin(phi_rad)

        nx, ny, nz = self.grid_size
        zc = self.pivot_z
        proyeccion = np.zeros((nx, ny), dtype=np.float32)
        for z in range(nz):
            sx = pad + int(round(dx * (z - zc)))
            sy = pad + int(round(dy * (z - zc)))
            proyeccion += padded_volume[sx:sx + nx, sy:sy + ny, z]

        path_length_correction = self.voxel_size_cm / np.cos(theta_rad)
        return proyeccion * path_length_correction

    def simulate_transmission(self, num_angles=NUM_ANGULOS):
        proyeccion_total = np.zeros((self.grid_size[0], self.grid_size[1]), dtype=np.float32)

        # Cenitales de la distribución medida: ya incluye flujo, área efectiva y aceptancia,
        # así que cada dirección pesa lo mismo. Azimut uniforme (medido: +-6% entre octantes)
        idx = np.random.choice(len(self.zenith_prob), size=num_angles, p=self.zenith_prob)
        thetas = np.random.uniform(self.zenith_bordes[idx], self.zenith_bordes[idx + 1])
        phis = np.random.uniform(0, 360, num_angles)

        # Margen para el peor caso: pivote en un extremo de la grilla
        pad = int(np.ceil(np.tan(np.radians(MAX_ZENITH_DEG)) * self.grid_size[2])) + 1
        padded_volume = np.pad(self.volume, ((pad, pad), (pad, pad), (0, 0)))

        for theta, phi in zip(thetas, phis):
            proyeccion_total += self.get_sheared_projection(padded_volume, pad, theta, phi)

        integral_1_x0 = proyeccion_total / num_angles
        return integral_1_x0

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

def get_enmallado_config():
    n = lambda: np.random.randint(2, 7)
    if np.random.rand() < 0.25:
        return (n(), 0) if np.random.rand() < 0.5 else (0, n())
    return n(), n()

def sample_block(gen):
    # Bloque anfitrión con tamaños cercanos a las muestras reales (47x47x21 y 49x30x25 cm)
    gx, gy, gz = gen.grid_size
    sx, sy = (np.random.uniform(25, 55, size=2) / VOXEL_CM).astype(int)
    sz = int(np.random.uniform(18, 28) / VOXEL_CM)
    shift = int(5 / VOXEL_CM)
    cx = gx // 2 + np.random.randint(-shift, shift + 1)
    cy = gy // 2 + np.random.randint(-shift, shift + 1)
    cz = gz // 2 + np.random.randint(-2 * shift, 2 * shift + 1)
    material = 'sand' if np.random.rand() < 0.2 else 'concrete'
    return (cx, cy, cz), (sx, sy, sz), material

def _random_point(bounds, margin):
    return np.array([np.random.uniform(lo + margin, hi - margin) for lo, hi in bounds])

def add_rebar_mesh(gen, bounds):
    # Malla tipo Mallado: varillas de 1-3 cm de diámetro en una o dos direcciones
    (x0, x1), (y0, y1), (z0, z1) = bounds
    r = np.random.uniform(0.5, 1.5) / VOXEL_CM
    m = np.random.uniform(1, 4) / VOXEL_CM  # recubrimiento en los extremos
    num_x, num_y = get_enmallado_config()
    z = np.random.uniform(z0 + r, z1 - 3 * r)
    for y in np.linspace(y0, y1, num_x + 2)[1:-1]:
        gen.add_bar((x0 + m, y, z), (x1 - m, y, z), r)
    for x in np.linspace(x0, x1, num_y + 2)[1:-1]:
        gen.add_bar((x, y0 + m, z + 2 * r), (x, y1 - m, z + 2 * r), r)

def add_isolated_bars(gen, bounds, n_bars):
    # Barras tipo Grosores: largo 5-40 cm, diámetro 1-6 cm; mitad alineadas con x o y,
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
    # Distribución: aire 5%, barras en aire 5%, bloque 10%, malla 30%, barras 30%,
    # barras + oclusiones 12%, oclusiones 8%  (20% de las muestras con oclusiones)
    opcion = i % 100
    center, size, material = sample_block(gen)
    if opcion < 5:
        return 'aire'
    if opcion < 10:
        bounds = gen.add_block(center, size, material='air')
        add_isolated_bars(gen, bounds, np.random.randint(1, 7))
        return 'barras_aire'
    bounds = gen.add_block(center, size, material=material)
    if opcion < 20:
        return 'bloque'
    if opcion < 50:
        add_rebar_mesh(gen, bounds)
        return 'malla'
    if opcion < 80:
        add_isolated_bars(gen, bounds, np.random.randint(1, 7))
        return 'barras'
    if opcion < 92:
        add_isolated_bars(gen, bounds, np.random.randint(1, 7))
        add_void_clusters(gen, bounds, np.random.randint(1, 4))
        return 'barras_oclusiones'
    add_void_clusters(gen, bounds, np.random.randint(1, 4))
    return 'oclusiones'

def generate_deterministic_dataset(num_samples=10000, output_dir="datasets/dataset_muografia_V9", csv_path=None):
    os.makedirs(output_dir, exist_ok=True)
    grid = GRID
    gen = PhysicalMuographyGenerator(grid_size=grid, voxel_size_cm=VOXEL_CM, muones_csv_path=csv_path)
    rate_map = np.load(RATE_MAP_PATH)
    
    print(f"Generando {num_samples} muestras rígidamente distribuidas en '{output_dir}'...")
    
    for i in tqdm(range(num_samples)):
        gen.reset(background='air')
        escena = build_scene(gen, i)
        
        y_proyectado = gen.simulate_transmission()
        
        # El filtro gaussiano simula la PSF del detector de entrada
        detector_psf_sigma = PSF_SIGMA_CM / VOXEL_CM
        y_degradado_input = gaussian_filter(y_proyectado, sigma=detector_psf_sigma)
        
        # Muones por píxel según la aceptancia medida del hodoscopio
        dias_observacion = sample_observation_days()
        muones = np.random.poisson(dias_observacion * rate_map)
        
        # 1. Input (X): Señal ruidosa de medición corta 
        x_input = measured_variance(y_degradado_input, muones=muones, clean=False)

        # 2. Target Varianza (y_var): Esperanza matemática asintótica
        y_target = measured_variance(y_proyectado, clean=True)
        
        # 3. Target Densidad (density): Integral física 
        y_density = y_proyectado.astype(np.float32)
        
        filename = os.path.join(output_dir, f"sample_{i:05d}.npz")
        # Guardamos los 3 arrays y el tipo de escena (para evaluar por categoría)
        np.savez_compressed(filename, x=x_input, y=y_target, density=y_density, scene=escena)

def generate_test_dataset(num_samples=500, output_dir="datasets/dataset_muografia_Test_V9", csv_path=None):
    generate_deterministic_dataset(num_samples=num_samples, output_dir=output_dir, csv_path=csv_path)

if __name__ == "__main__":
    RUTA_CSV = "Momento_Muones_filtrado.csv" 
    
    generate_deterministic_dataset(
        num_samples=10000, 
        output_dir="datasets/dataset_muografia_V9", 
        csv_path=RUTA_CSV
    )
    
    generate_test_dataset(
        num_samples=500, 
        output_dir="datasets/dataset_muografia_Test_V9", 
        csv_path=RUTA_CSV
    )
