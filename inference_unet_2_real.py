import os
import re
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


from Unet import RRDB_UNet_Deterministic
from data_generator import PEDESTAL, VOXEL_CM, limites_normalizacion, normalizar_muones, normalizar_entrada

# Figuras en inglés (tesis). Mapas de varianza en "average scattering density" = varianza / tamaño
# del vóxel proyectado (0.5 cm), en escala logarítmica; densidad y probabilidad en escala lineal.
LIMITES_CM = (4, 56)
# Las coordenadas de los mapas son las de la voxelización (x = primer índice). Según la profundidad
# de las barras del Mallado coinciden con el esquema M1; True rota 180° solo las figuras.
ROTAR_180 = False

# ==============================================================================
# GEOMETRÍA ESTIMADA (cm). Solo para dibujar encima de las predicciones: no se usa en el entrenamiento.
# Las muestras se colocan en el hodoscopio sin coordenadas, así que las posiciones salen de ajustar
# los esquemas M1/M2 (espaciados y diámetros conocidos) a los perfiles del INPUT medido, no a la red.
# ==============================================================================
def _malla(xs, ys, d=1.9, lo=4.0, hi=54.0):
    # M1: varillas #6 (Ø1.9 cm); xs = varillas a lo largo de y, ys = varillas a lo largo de x
    return ([{"x": (x - d / 2, x + d / 2), "y": (lo, hi)} for x in xs] +
            [{"x": (lo, hi), "y": (y - d / 2, y + d / 2)} for y in ys])

def _grosores(x_barra=(19, 29), x_hueco=(29, 34), ys=(15, 26.5, 39, 52.5), ds=(6, 4, 2, 1)):
    # M2: barras de 10 cm (Ø6, 4, 2, 1 cm) seguidas de un tubo vacío de 5 cm del mismo diámetro
    return ([{"x": x_barra, "y": (y - d / 2, y + d / 2)} for y, d in zip(ys, ds)] +
            [{"x": x_hueco, "y": (y - d / 2, y + d / 2), "hueco": True} for y, d in zip(ys, ds)])

BLOQUES_PLOMO = [
    {"x": (11, 19), "y": (24, 34)}, {"x": (30, 38), "y": (20, 40)}, {"x": (44, 54), "y": (20, 40)},
]
# Prefijo del nombre de la muestra -> geometría
GEOMETRIAS = {
    "Mallado_mortero": _malla(xs=(15, 19, 25, 33, 43), ys=(46, 42, 36, 28, 18)),
    "Mallado_arena": _malla(xs=(13, 18, 24.5, 32.5, 44), ys=(43, 38.5, 31.5, 24, 14)),
    "Grosores": _grosores(),
    "plomo": BLOQUES_PLOMO,
}
NOMBRES_TESIS = {"Mallado_mortero_20d": "Mesh (mortar)", "Mallado_arena_20d": "Mesh (sand)",
                 "Grosores_mortero_20d": "Thicknesses (mortar)", "plomo_bloques_84h": "Lead blocks"}

def geometria_de(nombre):
    for prefijo, geometria in GEOMETRIAS.items():
        if nombre.lower().startswith(prefijo.lower()):
            return geometria
    return []

# ==============================================================================
# FUNCIONES DE PADDING
# ==============================================================================
def pad_to_multiple(tensor, multiple=16):
    _, _, h, w = tensor.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_tensor = F.pad(tensor, padding, mode='replicate')
    return padded_tensor, padding

def unpad(tensor, padding):
    pad_left, pad_right, pad_top, pad_bottom = padding
    _, _, h, w = tensor.shape
    return tensor[:, :, pad_top:h-pad_bottom, pad_left:w-pad_right]

# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================
TIPOS_MAPA = {
    # clave: (cmap, escala log, etiqueta de la barra de color)
    'asd': ('inferno', True, 'Average Scattering Density [rad$^2$/cm]'),
    'dens': ('viridis', False, '$\\int dl/X_0$'),
    'prob': ('gray', False, 'Iron probability'),
}

def show_map(ax, mapa, tipo, vmin, vmax, titulo, geometria):
    # Los mapas están indexados [x, y]: se transponen para dibujar x horizontal, y vertical (en cm)
    cmap, log, etiqueta = TIPOS_MAPA[tipo]
    lado_cm = mapa.shape[0] * VOXEL_CM
    imagen = mapa.T
    if ROTAR_180:
        imagen = imagen[::-1, ::-1]
    norma = LogNorm(vmin=vmin, vmax=vmax) if log else None
    im = ax.imshow(np.clip(imagen, vmin, None) if log else imagen, origin='lower', cmap=cmap, norm=norma,
                   vmin=None if log else vmin, vmax=None if log else vmax, extent=[0, lado_cm, 0, lado_cm])
    for g in geometria:
        (x0, x1), (y0, y1) = g["x"], g["y"]
        if ROTAR_180:
            (x0, x1), (y0, y1) = (lado_cm - x1, lado_cm - x0), (lado_cm - y1, lado_cm - y0)
        hueco = g.get("hueco", False)
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.8,
                                       edgecolor='white' if hueco else 'cyan', facecolor='none',
                                       linestyle=':' if hueco else '--'))
    if tipo == 'prob':
        ax.contour(np.linspace(0, lado_cm, mapa.shape[0]), np.linspace(0, lado_cm, mapa.shape[1]),
                   imagen, levels=[0.5], colors='tab:red', linewidths=0.8)
    ax.set_xlim(*LIMITES_CM)
    ax.set_ylim(*LIMITES_CM)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    plt.colorbar(im, ax=ax, label=etiqueta, fraction=0.046)

def escala_asd(mapas):
    valores = np.concatenate([m[m > 0].ravel() for m in mapas])
    return np.percentile(valores, 1), np.percentile(valores, 99.5)

def plot_tomas(nombre_muestra, tomas, path, titulo):
    # Varias tomas de una muestra, una por fila, con la MISMA escala de color por columna
    geometria = geometria_de(nombre_muestra)
    esc_in = escala_asd([t['input'] / VOXEL_CM for t in tomas])
    esc_var = escala_asd([t['var'] / VOXEL_CM for t in tomas])
    vmax_dens = max(t['dens'].max() for t in tomas)
    fig, axes = plt.subplots(len(tomas), 4, figsize=(22, 4.8 * len(tomas)), squeeze=False)
    for fila, t in enumerate(tomas):
        show_map(axes[fila, 0], t['input'] / VOXEL_CM, 'asd', *esc_in, f"Measured (trimmed mean), {t['etiqueta']}", geometria)
        show_map(axes[fila, 1], t['var'] / VOXEL_CM, 'asd', *esc_var, f"Network: enhanced ASD, {t['etiqueta']}", geometria)
        show_map(axes[fila, 2], t['dens'], 'dens', 0.0, vmax_dens, f"Network: $\\int dl/X_0$, {t['etiqueta']}", geometria)
        show_map(axes[fila, 3], t['prob'], 'prob', 0.0, 1.0, f"Network: iron probability, {t['etiqueta']}", geometria)
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MODELO E INFERENCIA DE UN MAPA
# ==============================================================================
def cargar_modelo(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading RRDB-UNet from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    # .float(): también acepta los pesos exportados en float16 (exportar_pesos.py)
    clean_state_dict = {k.replace('module.', ''): v.float() if v.is_floating_point() else v
                        for k, v in state_dict.items()}
    con_hierro = any(k.startswith('head_iron') for k in clean_state_dict)
    model = RRDB_UNet_Deterministic(in_channels=2, base_dim=32, num_rrdb=3, con_hierro=con_hierro)
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    return model, device

def limpiar(mapa):
    mapa = np.nan_to_num(mapa.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(mapa, 0.0, None)

def inferir_mapa(model, device, varianza_medida, muones):
    # Media recortada medida (rad^2) + muones por píxel -> (varianza N->inf, densidad ∫1/X0,
    # probabilidad de hierro). La varianza de salida está en la escala de la media de theta^2.
    MAX_VAR, MAX_DENSITY = limites_normalizacion()
    entrada = np.stack([normalizar_entrada(varianza_medida), normalizar_muones(muones)])
    condition_tensor = torch.from_numpy(entrada).float().unsqueeze(0)
    condition_padded, padding_info = pad_to_multiple(condition_tensor, multiple=16)

    with torch.no_grad():
        salidas = model(condition_padded.to(device))
    salidas = [unpad(s, padding_info).cpu().numpy()[0, 0] for s in salidas]

    variance_clean = np.clip(salidas[0], 0.0, None) * MAX_VAR + PEDESTAL
    density_clean = np.clip(salidas[1], 0.0, None) * MAX_DENSITY
    prob = 1.0 / (1.0 + np.exp(-salidas[2])) if len(salidas) > 2 else np.zeros_like(density_clean)
    return variance_clean, density_clean, prob

def guardar_resultado(output_folder, base, titulo, medida, var, dens, prob, geometria):
    for nombre, mapa in [('variance_clean', var), ('density_clean', dens), ('iron_probability', prob)]:
        np.save(os.path.join(output_folder, f"{base}_{nombre}.npy"), mapa.astype(np.float32))
    np.savetxt(os.path.join(output_folder, f"{base}_density_clean.txt"), dens, fmt="%.6f")
    np.savetxt(os.path.join(output_folder, f"{base}_iron_probability.txt"), prob, fmt="%.4f")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    show_map(axes[0], medida / VOXEL_CM, 'asd', *escala_asd([medida / VOXEL_CM]), f"Measured (trimmed mean): {titulo}", geometria)
    show_map(axes[1], var / VOXEL_CM, 'asd', *escala_asd([var / VOXEL_CM]), "Network: enhanced ASD (N → ∞)", geometria)
    show_map(axes[2], dens, 'dens', 0.0, dens.max(), "Network: $\\int dl/X_0$ (PSF removed)", geometria)
    show_map(axes[3], prob, 'prob', 0.0, 1.0, "Network: iron probability (red: p = 0.5)", geometria)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{base}_network.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# BUCLE DE INFERENCIA
# ==============================================================================
def run_inference(input_folder, output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)
    # Tomas acumuladas: {muestra}_{dias}d_varrec_xy.npy; intervalos: {muestra}_int_{a}-{b}d_varrec_xy.npy
    # (preparar_recortada.py). Los muones de cada uno: el mismo nombre con _N_xy.npy
    archivos = sorted(glob.glob(os.path.join(input_folder, "*_varrec_xy.npy")))
    if not archivos:
        print(f"No *_varrec_xy.npy files in {input_folder} (run preparar_recortada.py)")
        return

    tomas, intervalos = {}, {}
    for ruta in tqdm(archivos, desc="Network inference"):
        base = os.path.basename(ruta).replace("_varrec_xy.npy", "")
        medida = limpiar(np.load(ruta))
        muones = limpiar(np.load(ruta.replace("_varrec_xy.npy", "_N_xy.npy")))
        var, dens, prob = inferir_mapa(model, device, medida, muones)
        m_int = re.match(r"(.+)_int_([\d.]+)-([\d.]+)d$", base)
        m_tom = re.match(r"(.+)_([\d.]+)d$", base)
        if m_int:
            muestra, a, b = m_int.group(1), float(m_int.group(2)), float(m_int.group(3))
            intervalos.setdefault(muestra, []).append({'dias': b, 'etiqueta': f"days {a:g}-{b:g}", 'input': medida,
                                                       'var': var, 'dens': dens, 'prob': prob})
            np.save(os.path.join(output_folder, f"{base}_iron_probability.npy"), prob.astype(np.float32))
            np.save(os.path.join(output_folder, f"{base}_density_clean.npy"), dens.astype(np.float32))
            continue
        muestra, dias = m_tom.group(1), float(m_tom.group(2))
        nombre = NOMBRES_TESIS.get(muestra, muestra)
        guardar_resultado(output_folder, base, f"{nombre}, {dias:g} days", medida, var, dens, prob, geometria_de(muestra))
        tomas.setdefault(muestra, []).append({'dias': dias, 'etiqueta': f"{dias:g} days", 'input': medida,
                                              'var': var, 'dens': dens, 'prob': prob})

    for muestra, lista in tomas.items():
        nombre = NOMBRES_TESIS.get(muestra, muestra)
        plot_tomas(muestra, sorted(lista, key=lambda t: t['dias']), os.path.join(output_folder, f"comparison_{muestra}.png"),
                   f"{nombre}: cumulative observation times (same color scale per column)")
    for muestra, lista in intervalos.items():
        lista = sorted(lista, key=lambda t: t['dias'])
        nombre = NOMBRES_TESIS.get(muestra, muestra)
        plot_tomas(muestra, lista, os.path.join(output_folder, f"independent_intervals_{muestra}.png"),
                   f"{nombre}: INDEPENDENT time intervals (no muon in common)")
        # Repetibilidad de la estructura fina (sin el fondo suave del bloque) entre intervalos independientes
        fina = lambda m: (m - gaussian_filter(m, 5))[20:100, 20:100].ravel()
        pares = [(i, j) for i in range(len(lista)) for j in range(i + 1, len(lista))]
        for clave in ('dens', 'prob'):
            corr = [np.corrcoef(fina(lista[i][clave]), fina(lista[j][clave]))[0, 1] for i, j in pares]
            print(f"{muestra}: {clave} fine-structure correlation between independent intervals {np.round(corr, 2)}")
    print(f"Results saved in '{output_folder}'")

if __name__ == "__main__":
    INPUT_DIR = "datos_reales"   # salida de preparar_recortada.py
    OUTPUT_DIR = "resultados/V14"
    MODEL_PATH = "checkpoints/V14/best_model.pth"

    model, device = cargar_modelo(MODEL_PATH)
    run_inference(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR, model=model, device=device)
