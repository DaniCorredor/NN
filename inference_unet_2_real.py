import os
import json
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
# Ventana de las figuras: 52 cm centrados en el centro del detector y de la aceptancia (28 cm)
LIMITES_CM = (2, 54)
# True: fuera de la zona confiable los mapas valen 0 y la ventana se recorta a esa zona
RECORTAR_A_CONFIABLE = False
# Escala fija (logarítmica) de la average scattering density medida en todas las figuras
ESCALA_ENTRADA_ASD = (0.8e-3, 1.28e-2)
# Las coordenadas de los mapas son las de la voxelización (x = primer índice). TRANSPONER = True
# dibuja como en la tesis: el y del detector en horizontal y el x en vertical (Grosores: barras
# verticales, de la más gruesa a la izquierda a la más delgada a la derecha; Mallado: zona de más
# barras abajo a la derecha). Solo cambia las figuras, no los .npy.
TRANSPONER = True
# Umbral de la probabilidad de hierro (contorno rojo): el calibrado en el conjunto de prueba
# (evaluar_test.py lo guarda en iron_threshold.json); 0.5 si no existe
UMBRAL_HIERRO = 0.5
# Zona poco confiable del detector (confiabilidad.py: RED < RED_MINIMA): se atenúa en gris con un
# borde difuso (de RED_MINIMA a RED_MINIMA - ATENUACION_ANCHO) y se marca su límite con una línea
MOSTRAR_CONFIABILIDAD = True
# Guías de la geometría estimada encima de los mapas: apagadas en las figuras de la tesis
# (sesgan al lector); se siguen usando para las métricas
MOSTRAR_GUIA = False
ATENUACION_ANCHO = 0.08
ATENUACION_MAXIMA = 0.7

# ==============================================================================
# GEOMETRÍA ESTIMADA (cm). Solo para dibujar encima de las predicciones: no se usa en el entrenamiento.
# Las muestras se colocan en el hodoscopio sin coordenadas, así que las posiciones salen de ajustar
# los esquemas M1/M2 (espaciados y diámetros conocidos) a los perfiles del INPUT medido, no a la red.
# ==============================================================================
def _malla(xs, ys, d=1.9, lo=4.0, hi=54.0):
    # M1: varillas #6 (Ø1.9 cm); xs = varillas a lo largo de y, ys = varillas a lo largo de x
    return ([{"x": (x - d / 2, x + d / 2), "y": (lo, hi)} for x in xs] +
            [{"x": (lo, hi), "y": (y - d / 2, y + d / 2)} for y in ys])

def _grosores(x_barra=(19, 29), x_hueco=(29, 34), ys=(14.0, 27.9, 40.7, 50.7), ds=(6, 4, 2, 1)):
    # M2: barras de 10 cm (Ø6, 4, 2, 1 cm) seguidas de un tubo vacío de 5 cm del mismo diámetro.
    # Centros en y ajustados al input (perfiles de cuerda + PSF, iguales en los 3 intervalos
    # independientes a ±0.3 cm); la de 1 cm no se resuelve y va a 10 cm de la de 2 cm (dato de la muestra)
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
# Nombres de la tesis: M1 = malla (Mallado), M2 = grosores; el material (arena o concreto) va en el pie
NOMBRES_TESIS = {"Mallado_mortero_20d": "$\\mathcal{M}_1$ (concrete)", "Mallado_arena_20d": "$\\mathcal{M}_1$ (sand)",
                 "Grosores_mortero_20d": "$\\mathcal{M}_2$", "plomo_bloques_84h": "Lead blocks"}

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
    'dens': ('YlGnBu', False, '$\\int dl/X_0$'),
    'prob': ('gray', False, 'Iron probability'),
    'freq': ('gray', False, 'Detection frequency'),
    'std': ('cividis', False, 'Std. dev. of $\\int dl/X_0$'),
}

_POCO_CONFIABLE = None

def zona_poco_confiable(forma):
    global _POCO_CONFIABLE
    if _POCO_CONFIABLE is None:
        from confiabilidad import mapa_confiable
        _POCO_CONFIABLE = ~mapa_confiable(forma)
    return _POCO_CONFIABLE

def limites_confiables():
    # Ventana cuadrada que contiene la zona confiable (centrada en el centro del detector)
    confiable = ~zona_poco_confiable((112, 112))
    filas = np.where(confiable.any(axis=1))[0]
    return filas[0] * VOXEL_CM, (filas[-1] + 1) * VOXEL_CM

def atenuacion(forma):
    # Opacidad del velo gris: 0 dentro de la zona confiable, sube suave fuera de ella
    from confiabilidad import densidad_relativa, RED_MINIMA
    red = densidad_relativa()
    return ATENUACION_MAXIMA * np.clip((RED_MINIMA - red) / ATENUACION_ANCHO, 0.0, 1.0)

def show_map(ax, mapa, tipo, vmin, vmax, titulo, geometria):
    # Los mapas están indexados [x, y] del detector; ver TRANSPONER
    cmap, log, etiqueta = TIPOS_MAPA[tipo]
    lado_cm = mapa.shape[0] * VOXEL_CM
    if RECORTAR_A_CONFIABLE:
        mapa = np.where(zona_poco_confiable(mapa.shape), 0.0, mapa)
    imagen = mapa if TRANSPONER else mapa.T
    norma = LogNorm(vmin=vmin, vmax=vmax) if log else None
    im = ax.imshow(np.clip(imagen, vmin, None) if log else imagen, origin='lower', cmap=cmap, norm=norma,
                   vmin=None if log else vmin, vmax=None if log else vmax, extent=[0, lado_cm, 0, lado_cm])
    for g in (geometria if MOSTRAR_GUIA else []):
        (x0, x1), (y0, y1) = g["x"], g["y"]
        if TRANSPONER:
            (x0, x1), (y0, y1) = (y0, y1), (x0, x1)
        hueco = g.get("hueco", False)
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.8,
                                       edgecolor='white' if hueco else 'cyan', facecolor='none',
                                       linestyle=':' if hueco else '--'))
    centros = (np.arange(mapa.shape[0]) + 0.5) * VOXEL_CM
    if tipo in ('prob', 'freq'):
        ax.contour(centros, centros, imagen, levels=[UMBRAL_HIERRO if tipo == 'prob' else 0.5],
                   colors='tab:red', linewidths=0.8)
    if MOSTRAR_CONFIABILIDAD and not RECORTAR_A_CONFIABLE:
        alfa = atenuacion(mapa.shape)
        alfa = alfa if TRANSPONER else alfa.T
        velo = np.zeros(alfa.shape + (4,))
        velo[..., :3] = 0.55
        velo[..., 3] = alfa
        ax.imshow(velo, origin='lower', extent=[0, lado_cm, 0, lado_cm], interpolation='bilinear')
        poco = zona_poco_confiable(mapa.shape).astype(float)
        poco = poco if TRANSPONER else poco.T
        ax.contour(centros, centros, poco, levels=[0.5], colors='white', linewidths=1.0)
    limites = limites_confiables() if RECORTAR_A_CONFIABLE else LIMITES_CM
    ax.set_xlim(*limites)
    ax.set_ylim(*limites)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    plt.colorbar(im, ax=ax, label=etiqueta, fraction=0.046)

def escala_asd(mapas):
    valores = np.concatenate([m[m > 0].ravel() for m in mapas])
    return np.percentile(valores, 1), np.percentile(valores, 99.5)

def plot_tomas(nombre_muestra, tomas, path, titulo, entrada='Measured (trimmed mean)'):
    # Varias tomas de una muestra, una por fila, con la MISMA escala de color por columna
    geometria = geometria_de(nombre_muestra)
    esc_in = ESCALA_ENTRADA_ASD
    esc_var = escala_asd([t['var'] / VOXEL_CM for t in tomas])
    vmax_dens = max(t['dens'].max() for t in tomas)
    fig, axes = plt.subplots(len(tomas), 4, figsize=(22, 4.8 * len(tomas)), squeeze=False)
    for fila, t in enumerate(tomas):
        # 'input_display' (opcional): mapa que se dibuja como medida en vez del que recibió la red
        show_map(axes[fila, 0], t.get('input_display', t['input']) / VOXEL_CM, 'asd', *esc_in,
                 f"{entrada}, {t['etiqueta']}", geometria)
        show_map(axes[fila, 1], t['var'] / VOXEL_CM, 'asd', *esc_var, f"Enhanced ASD, {t['etiqueta']}", geometria)
        show_map(axes[fila, 2], t['dens'], 'dens', 0.0, vmax_dens, f"$\\int dl/X_0$, {t['etiqueta']}", geometria)
        show_map(axes[fila, 3], t['prob'], 'prob', 0.0, 1.0, f"Iron probability, {t['etiqueta']}", geometria)
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

def guardar_resultado(output_folder, base, titulo, medida, var, dens, prob, geometria, entrada='Measured (trimmed mean)'):
    for nombre, mapa in [('variance_clean', var), ('density_clean', dens), ('iron_probability', prob)]:
        np.save(os.path.join(output_folder, f"{base}_{nombre}.npy"), mapa.astype(np.float32))
    np.savetxt(os.path.join(output_folder, f"{base}_density_clean.txt"), dens, fmt="%.6f")
    np.savetxt(os.path.join(output_folder, f"{base}_iron_probability.txt"), prob, fmt="%.4f")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    show_map(axes[0], medida / VOXEL_CM, 'asd', *ESCALA_ENTRADA_ASD, f"{entrada}: {titulo}", geometria)
    show_map(axes[1], var / VOXEL_CM, 'asd', *escala_asd([var / VOXEL_CM]), "Enhanced ASD", geometria)
    show_map(axes[2], dens, 'dens', 0.0, dens.max(), "$\\int dl/X_0$", geometria)
    show_map(axes[3], prob, 'prob', 0.0, 1.0, "Iron probability", geometria)
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
    # La toma vacía (0.7 d) es solo para calibrar el exceso de los bordes: está fuera del rango de
    # días del entrenamiento (3-25 d) y la red inventa estructura en ella
    archivos = sorted(r for r in glob.glob(os.path.join(input_folder, "*_varrec_xy.npy"))
                      if not os.path.basename(r).startswith("vacio_exp"))
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

# ==============================================================================
# PRUEBA DE REFERENCIA: DATOS FILTRADOS (DOCA, ángulo...) Y DECONVOLUCIONADOS
# Solo imágenes de referencia: la red se calibró con la media recortada SIN filtros.
# Cada imagen se pasa de dos formas:
#   direct   -> tal cual
#   rescaled -> aire y concreto llevados a los niveles del mapa recortado original de la misma
#               toma (transformación lineal), para compensar el cambio de escala
# El canal de muones usa el N del mapa original (el N después del filtro no está disponible).
# Las deconvoluciones (.npz) vienen transpuestas respecto a los mapas originales: se corrigen.
# ==============================================================================
def niveles_aire_concreto(mapa):
    # Aire: mediana del borde (8 px) con datos; concreto: mediana del centro 32x32 con datos
    con_datos = mapa > 0
    borde = np.ones_like(con_datos)
    borde[8:-8, 8:-8] = False
    centro = mapa[40:72, 40:72]
    return np.median(mapa[borde & con_datos]), np.median(centro[centro > 0])

def reescalar_a_original(mapa, original):
    aire_f, conc_f = niveles_aire_concreto(mapa)
    aire_o, conc_o = niveles_aire_concreto(original)
    reescalado = aire_o + (mapa - aire_f) * (conc_o - aire_o) / (conc_f - aire_f)
    reescalado[mapa == 0] = 0.0   # sin muones sigue siendo sin muones
    return np.clip(reescalado, 0.0, None)

def buscar_original(originales_folder, nombre_muestra, dias):
    # Empareja sin importar mayúsculas (p. ej. "mallado_arena" con "Mallado_arena")
    buscado = f"{nombre_muestra}_{dias}d_varrec_xy.npy".lower()
    for ruta in glob.glob(os.path.join(originales_folder, "*_varrec_xy.npy")):
        if os.path.basename(ruta).lower() == buscado:
            return ruta
    return None

def run_filtered_inference(input_folder, originales_folder, output_folder, model, device):
    os.makedirs(output_folder, exist_ok=True)
    archivos = sorted(glob.glob(os.path.join(input_folder, "*_fil.npy")) +
                      glob.glob(os.path.join(input_folder, "*_deconvolution.npz")))
    if not archivos:
        print(f"No *_fil.npy or *_deconvolution.npz files in {input_folder}")
        return

    por_muestra = {}
    for file_path in tqdm(archivos, desc="Filtered data (reference)"):
        base_name = os.path.basename(file_path)
        match = re.match(r"(.+)_([\d.]+)d_(fil|deconvolution)\.np[yz]$", base_name)
        if not match:
            print(f"  Unrecognized name, skipped: {base_name}")
            continue
        nombre_muestra, dias, tipo = match.group(1), match.group(2), match.group(3)
        nombre_muestra = nombre_muestra[0].upper() + nombre_muestra[1:]
        ruta_original = buscar_original(originales_folder, nombre_muestra, dias)
        if ruta_original is None:
            print(f"  No original map (and muons) for {base_name}: skipped")
            continue
        original = limpiar(np.load(ruta_original))
        muones = limpiar(np.load(ruta_original.replace("_varrec_xy.npy", "_N_xy.npy")))
        mapa = limpiar(np.load(file_path)) if tipo == "fil" else limpiar(np.load(file_path)["imagen_mejorada"]).T
        etiqueta_tipo = "filtered" if tipo == "fil" else "deconvolved"

        geometria = geometria_de(nombre_muestra)
        filas = por_muestra.setdefault(nombre_muestra, {})
        filas["original (trimmed mean)"] = (original,) + inferir_mapa(model, device, original, muones)
        for modo, entrada in {"direct": mapa, "rescaled": reescalar_a_original(mapa, original)}.items():
            etiqueta = f"{etiqueta_tipo}, {modo}"
            salida = inferir_mapa(model, device, entrada, muones)
            guardar_resultado(output_folder, f"{nombre_muestra}_{dias}d_{tipo}_{modo}",
                              f"{NOMBRES_TESIS.get(nombre_muestra, nombre_muestra)} {dias} days ({etiqueta})",
                              entrada, *salida, geometria, entrada='Input')
            filas[etiqueta] = (entrada,) + salida

    # Un solo filtro (voxelizaciones *_doca.npz / *_angulo.npz -> media recortada con preparar_recortada):
    # mismo estimador que los originales, así que entran directo con su propio N
    for ruta in sorted(glob.glob(os.path.join(input_folder, "*_varrec_xy.npy"))):
        match = re.match(r"(.+)_([\d.]+)d_(doca|angulo)_varrec_xy\.npy$", os.path.basename(ruta))
        if not match:
            continue
        nombre_muestra, dias, filtro = match.group(1), match.group(2), match.group(3)
        nombre_muestra = nombre_muestra[0].upper() + nombre_muestra[1:]
        mapa = limpiar(np.load(ruta))
        salida = inferir_mapa(model, device, mapa, limpiar(np.load(ruta.replace("_varrec_xy.npy", "_N_xy.npy"))))
        etiqueta = {"doca": "DOCA cut only", "angulo": "incidence-angle cut only"}[filtro]
        guardar_resultado(output_folder, f"{nombre_muestra}_{dias}d_{filtro}",
                          f"{NOMBRES_TESIS.get(nombre_muestra, nombre_muestra)} {dias} days ({etiqueta})",
                          mapa, *salida, geometria_de(nombre_muestra), entrada='Input (trimmed mean)')
        por_muestra.setdefault(nombre_muestra, {})[etiqueta] = (mapa,) + salida

    for nombre_muestra, filas in por_muestra.items():
        tomas = [{'etiqueta': k, 'input': v[0], 'var': v[1], 'dens': v[2], 'prob': v[3]} for k, v in filas.items()]
        plot_tomas(nombre_muestra, tomas, os.path.join(output_folder, f"summary_{nombre_muestra}.png"),
                   f"{NOMBRES_TESIS.get(nombre_muestra, nombre_muestra)}: REFERENCE with filtered data "
                   f"(network not calibrated for them)", entrada='Input')
    print(f"Reference results saved in '{output_folder}'")

if __name__ == "__main__":
    INPUT_DIR = "datos_reales"   # salida de preparar_recortada.py
    OUTPUT_DIR = "resultados/V17"
    MODEL_PATH = "checkpoints/V17/best_model.pth"
    umbral_path = os.path.join(OUTPUT_DIR, "test_evaluation", "iron_threshold.json")
    if os.path.exists(umbral_path):
        with open(umbral_path) as f:
            UMBRAL_HIERRO = json.load(f)["iron_threshold"]
    print(f"Iron threshold: {UMBRAL_HIERRO:.2f}")

    model, device = cargar_modelo(MODEL_PATH)
    run_inference(input_folder=INPUT_DIR, output_folder=OUTPUT_DIR, model=model, device=device)
    run_filtered_inference(input_folder="datos_reales_filtrados", originales_folder=INPUT_DIR,
                           output_folder=os.path.join(OUTPUT_DIR, "filtered_reference"), model=model, device=device)
