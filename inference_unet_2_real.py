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
from tqdm import tqdm


from Unet import RRDB_UNet_Deterministic
from data_generator import PEDESTAL, VOXEL_CM, limites_normalizacion

# ==============================================================================
# GEOMETRÍA REAL CONOCIDA (cm, mismas coordenadas que anal_voxel_poca_plomo.ipynb)
# Solo para dibujar encima de las predicciones: no se usa en el entrenamiento.
# ==============================================================================
BARRAS = [
    # Varillas a lo largo de y
    {"x": (11, 13), "y": (5, 55)}, {"x": (17, 19), "y": (5, 55)}, {"x": (25, 27), "y": (5, 55)},
    {"x": (35, 37), "y": (5, 55)}, {"x": (47, 49), "y": (5, 55)},
    # Varillas a lo largo de x
    {"x": (5, 55), "y": (47, 49)}, {"x": (5, 55), "y": (41, 43)}, {"x": (5, 55), "y": (33, 35)},
    {"x": (5, 55), "y": (23, 25)}, {"x": (5, 55), "y": (11, 13)},
]
BLOQUES_PLOMO = [
    {"x": (11, 19), "y": (24, 34)}, {"x": (30, 38), "y": (20, 40)}, {"x": (44, 54), "y": (20, 40)},
]
# Prefijo del nombre de la muestra -> geometría (Grosores: falta definir sus coordenadas)
GEOMETRIAS = {"Mallado": BARRAS, "plomo": BLOQUES_PLOMO}

def geometria_de(nombre):
    for prefijo, geometria in GEOMETRIAS.items():
        if nombre.startswith(prefijo):
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
def show_map(ax, mapa, cmap, vmin, vmax, titulo, geometria, etiqueta):
    # Los mapas están indexados [x, y]: se transponen para dibujar x horizontal, y vertical (en cm)
    lado_cm = mapa.shape[0] * VOXEL_CM
    im = ax.imshow(mapa.T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, extent=[0, lado_cm, 0, lado_cm])
    for g in geometria:
        ax.add_patch(patches.Rectangle((g["x"][0], g["y"][0]), g["x"][1] - g["x"][0], g["y"][1] - g["y"][0],
                                       linewidth=0.8, edgecolor='cyan', facecolor='none', linestyle='--'))
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    plt.colorbar(im, ax=ax, label=etiqueta, fraction=0.046)

def plot_comparison(nombre_muestra, tomas, output_folder):
    # Todas las tomas de una muestra lado a lado y con la MISMA escala de color por columna
    tomas = sorted(tomas, key=lambda t: t['dias'])
    geometria = geometria_de(nombre_muestra)
    escalas = {
        'input': (PEDESTAL * 0.5, np.percentile(np.concatenate([t['input'].ravel() for t in tomas]), 99)),
        'var': (min(t['var'].min() for t in tomas), max(t['var'].max() for t in tomas)),
        'dens': (0.0, max(t['dens'].max() for t in tomas)),
    }
    fig, axes = plt.subplots(len(tomas), 3, figsize=(16, 4.6 * len(tomas)), squeeze=False)
    for fila, t in enumerate(tomas):
        show_map(axes[fila, 0], t['input'], 'inferno', *escalas['input'], f"Input {t['dias']} d", geometria, 'rad$^2$')
        show_map(axes[fila, 1], t['var'], 'inferno', *escalas['var'], f"Varianza mejorada {t['dias']} d", geometria, 'rad$^2$')
        show_map(axes[fila, 2], t['dens'], 'viridis', *escalas['dens'], f"Densidad $\\int 1/X_0$ {t['dias']} d", geometria, '$\\int dl/X_0$')
    fig.suptitle(f"{nombre_muestra}: comparación entre tiempos de observación (misma escala por columna)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"comparacion_{nombre_muestra}.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# MODELO E INFERENCIA DE UN MAPA
# ==============================================================================
def cargar_modelo(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelo dual RRDB-UNet desde: {model_path}")

    model = RRDB_UNet_Deterministic(in_channels=1, base_dim=32, num_rrdb=3)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()
    return model, device

def limpiar(mapa):
    mapa = np.nan_to_num(mapa.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(mapa, 0.0, None)

def inferir_mapa(model, device, mutrec_variance):
    # Varianza medida (rad^2) -> (varianza N->inf, densidad ∫1/X0) en unidades físicas
    PEDESTAL_MEAN = PEDESTAL
    MAX_VAR, MAX_DENSITY = limites_normalizacion()

    # --- 1. Calibración y Normalización ---
    x_calibrada = np.clip(mutrec_variance - PEDESTAL_MEAN, 0.0, None)
    norm_x = np.clip(x_calibrada / MAX_VAR, 0.0, 1.0).astype(np.float32)

    # --- 2. Inferencia de la Red Neuronal (Doble Salida) ---
    condition_tensor = torch.from_numpy(norm_x).float().unsqueeze(0).unsqueeze(0)
    condition_padded, padding_info = pad_to_multiple(condition_tensor, multiple=16)
    condition_padded = condition_padded.to(device)

    with torch.no_grad():
        y_pred_var_padded, y_pred_dens_padded = model(condition_padded)

    y_pred_var_tensor = unpad(y_pred_var_padded, padding_info)
    y_pred_dens_tensor = unpad(y_pred_dens_padded, padding_info)

    y_pred_var_norm = np.clip(y_pred_var_tensor.cpu().numpy()[0, 0], 0.0, None)
    y_pred_dens_norm = np.clip(y_pred_dens_tensor.cpu().numpy()[0, 0], 0.0, None)

    # --- 3. Desnormalización (Output Final) ---
    variance_clean = (y_pred_var_norm * MAX_VAR) + PEDESTAL_MEAN
    density_clean = y_pred_dens_norm * MAX_DENSITY
    return variance_clean, density_clean

def guardar_resultado(output_folder, save_base_name, titulo, mutrec_variance, variance_clean, density_clean, geometria):
    np.save(os.path.join(output_folder, f"{save_base_name}_variance_clean.npy"), variance_clean.astype(np.float32))
    np.save(os.path.join(output_folder, f"{save_base_name}_density_clean.npy"), density_clean.astype(np.float32))
    np.savetxt(os.path.join(output_folder, f"{save_base_name}_variance_clean.txt"), variance_clean, fmt="%.8f")
    np.savetxt(os.path.join(output_folder, f"{save_base_name}_density_clean.txt"), density_clean, fmt="%.6f")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax_input = np.percentile(mutrec_variance, 99)
    show_map(axes[0], mutrec_variance, 'inferno', 0.0, vmax_input, f"Input: {titulo}", geometria, 'rad$^2$')
    show_map(axes[1], variance_clean, 'inferno', None, None, "Varianza mejorada (N → ∞)", geometria, 'rad$^2$')
    show_map(axes[2], density_clean, 'viridis', 0.0, None, "Densidad $\\int 1/X_0$ (sin PSF)", geometria, '$\\int dl/X_0$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{save_base_name}_dual_inference.png"), dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# BUCLE DE INFERENCIA DUAL (VARIANZA Y DENSIDAD)
# ==============================================================================
def run_deterministic_inference(input_folder, output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(input_folder, "*_var_xy.npy")))
    if not npy_files:
        print(f"No se encontraron archivos *_var_xy.npy en {input_folder}")
        return

    por_muestra = {}
    for file_path in tqdm(npy_files, desc="Procesando inferencia dual"):
        base_name = os.path.basename(file_path)
        # Los *_var_xy.npy ya vienen en varianza (Σθ²/ΣN por columna)
        mutrec_variance = limpiar(np.load(file_path))
        variance_clean, density_clean = inferir_mapa(model, device, mutrec_variance)

        save_base_name = base_name.replace(".npy", "")
        match = re.match(r"(.+)_([\d.]+)d_var_xy\.npy$", base_name)
        nombre_muestra, dias = (match.group(1), float(match.group(2))) if match else (save_base_name, 0.0)
        geometria = geometria_de(nombre_muestra)

        guardar_resultado(output_folder, save_base_name, f"{nombre_muestra} {dias} d",
                          mutrec_variance, variance_clean, density_clean, geometria)
        por_muestra.setdefault(nombre_muestra, []).append(
            {'dias': dias, 'input': mutrec_variance, 'var': variance_clean, 'dens': density_clean})

    # Comparación entre tiempos de observación de la misma muestra
    for nombre_muestra, tomas in por_muestra.items():
        plot_comparison(nombre_muestra, tomas, output_folder)
    print(f"Resultados guardados en '{output_folder}'")

# ==============================================================================
# PRUEBA: DATOS FILTRADOS (DOCA, ángulo...) Y DECONVOLUCIONADOS
# La red se calibró con los mapas SIN filtrar, así que estos resultados son solo
# exploratorios. Cada imagen se pasa de dos formas:
#   directo    -> tal cual, con la misma calibración que los datos originales
#   reescalado -> aire y concreto llevados a los niveles del mapa original de la
#                 misma toma (transformación lineal), para compensar el cambio de escala
# ==============================================================================
def niveles_aire_concreto(mapa):
    # Aire: mediana del borde (8 px) con datos; concreto: mediana del centro 32x32 con datos
    con_datos = mapa > 0
    borde = np.ones_like(con_datos)
    borde[8:-8, 8:-8] = False
    aire = np.median(mapa[borde & con_datos])
    centro = mapa[40:72, 40:72]
    concreto = np.median(centro[centro > 0])
    return aire, concreto

def reescalar_a_original(mapa, original):
    aire_f, conc_f = niveles_aire_concreto(mapa)
    aire_o, conc_o = niveles_aire_concreto(original)
    reescalado = aire_o + (mapa - aire_f) * (conc_o - aire_o) / (conc_f - aire_f)
    reescalado[mapa == 0] = 0.0   # sin muones sigue siendo sin muones
    return np.clip(reescalado, 0.0, None)

def buscar_original(originales_folder, nombre_muestra, dias):
    # Empareja sin importar mayúsculas (p. ej. "mallado_arena" con "Mallado_arena")
    buscado = f"{nombre_muestra}_{dias}d_var_xy.npy".lower()
    for ruta in glob.glob(os.path.join(originales_folder, "*_var_xy.npy")):
        if os.path.basename(ruta).lower() == buscado:
            return ruta
    return None

def run_filtered_inference(input_folder, originales_folder, output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)
    archivos = sorted(glob.glob(os.path.join(input_folder, "*_fil.npy")) +
                      glob.glob(os.path.join(input_folder, "*_deconvolution.npz")))
    if not archivos:
        print(f"No se encontraron archivos *_fil.npy ni *_deconvolution.npz en {input_folder}")
        return

    por_muestra = {}
    for file_path in tqdm(archivos, desc="Prueba datos filtrados"):
        base_name = os.path.basename(file_path)
        match = re.match(r"(.+)_([\d.]+)d_(fil|deconvolution)\.np[yz]$", base_name)
        if not match:
            print(f"  Nombre no reconocido, se omite: {base_name}")
            continue
        nombre_muestra, dias, tipo = match.group(1), match.group(2), match.group(3)
        # "mallado_..." y "Mallado_..." son la misma muestra (la geometría se busca con mayúscula)
        nombre_muestra = nombre_muestra[0].upper() + nombre_muestra[1:]

        if tipo == "fil":
            mapa = limpiar(np.load(file_path))
        else:
            mapa = limpiar(np.load(file_path)["imagen_mejorada"])

        entradas = {"directo": mapa}
        ruta_original = buscar_original(originales_folder, nombre_muestra, dias)
        if ruta_original is not None:
            original = limpiar(np.load(ruta_original))
            entradas["reescalado"] = reescalar_a_original(mapa, original)
            por_muestra.setdefault(nombre_muestra, {})["original"] = (original,) + inferir_mapa(model, device, original)
        else:
            print(f"  Sin mapa original para {base_name}: solo modo directo")

        geometria = geometria_de(nombre_muestra)
        for modo, entrada in entradas.items():
            variance_clean, density_clean = inferir_mapa(model, device, entrada)
            etiqueta = f"{tipo}_{modo}"
            guardar_resultado(output_folder, f"{nombre_muestra}_{dias}d_{etiqueta}", f"{nombre_muestra} {dias} d ({etiqueta})",
                              entrada, variance_clean, density_clean, geometria)
            por_muestra.setdefault(nombre_muestra, {})[etiqueta] = (entrada, variance_clean, density_clean)

    # Resumen por muestra: una fila por tipo de entrada (original, filtrado, deconvolución)
    orden = ["original", "fil_directo", "fil_reescalado", "deconvolution_directo", "deconvolution_reescalado"]
    for nombre_muestra, filas in por_muestra.items():
        claves = [k for k in orden if k in filas]
        geometria = geometria_de(nombre_muestra)
        fig, axes = plt.subplots(len(claves), 3, figsize=(16, 4.6 * len(claves)), squeeze=False)
        for fila, clave in enumerate(claves):
            entrada, var, dens = filas[clave]
            show_map(axes[fila, 0], entrada, 'inferno', 0.0, np.percentile(entrada, 99), f"Input ({clave})", geometria, 'rad$^2$')
            show_map(axes[fila, 1], var, 'inferno', None, None, f"Varianza mejorada ({clave})", geometria, 'rad$^2$')
            show_map(axes[fila, 2], dens, 'viridis', 0.0, None, f"Densidad $\\int 1/X_0$ ({clave})", geometria, '$\\int dl/X_0$')
        fig.suptitle(f"{nombre_muestra}: PRUEBA con datos filtrados (la red no está calibrada para ellos)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"resumen_{nombre_muestra}.png"), dpi=120, bbox_inches='tight')
        plt.close()
    print(f"Resultados de prueba guardados en '{output_folder}'")

if __name__ == "__main__":
    INPUT_DIR = "datos_reales"   # salida de preparar_datos_reales.py
    FILTRADOS_DIR = "datos_reales_filtrados"
    OUTPUT_DIR = "resultados/V12"
    MODEL_PATH = "checkpoints/V12/best_model.pth"

    model, device = cargar_modelo(MODEL_PATH)
    run_deterministic_inference(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        model=model,
        device=device
    )
    run_filtered_inference(
        input_folder=FILTRADOS_DIR,
        originales_folder=INPUT_DIR,
        output_folder=os.path.join(OUTPUT_DIR, "filtrados_prueba"),
        model=model,
        device=device
    )
