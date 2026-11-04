import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import inference_unet_2_real as inf
from data_generator import FRACCION_RECORTE
from calibrar_bordes import voxel_data
from preparar_recortada import columnas

# Incertidumbre estadística de la salida de la red por remuestreo de muones (bootstrap).
# En cada columna se sortean con reemplazo tantos muones como tiene (mismo N), se recalcula la
# media recortada y se pasa por la red. Con REPETICIONES mapas:
#   detection frequency  fracción de repeticiones con probabilidad de hierro > umbral calibrado
#                        (1 = la estructura aparece siempre; ~0.5 = depende de qué muones caigan)
#   density std          dispersión de ∫dl/X0 entre repeticiones
# Corre en el PC (necesita los ángulos por muón de ../Datos) con los pesos exportados de la red.

VERSION = "V16"
MODELO = f"modelos/{VERSION}_fp16.pt"
SALIDA = f"resultados/{VERSION}/postprocessing/bootstrap"
REPETICIONES = 50
MUESTRAS = ["Mallado_arena_20d", "Mallado_mortero_20d", "Grosores_mortero_20d"]

class Remuestreo:
    # Todos los theta^2 en un vector, ordenados por columna, para remuestrear sin bucles
    def __init__(self, cols):
        self.forma = cols.shape
        largos = np.array([len(c) for c in cols.ravel()])
        self.N = largos.reshape(self.forma).astype(np.float32)
        self.inicio = np.concatenate([[0], np.cumsum(largos)[:-1]])
        self.columna = np.repeat(np.arange(largos.size), largos)
        self.valores = np.concatenate([c for c in cols.ravel() if len(c)])
        self.largos = largos

    def media_recortada(self, valores):
        # Igual que data_generator.media_recortada (recorte fraccionario), columna por columna
        orden = np.lexsort((valores, self.columna))
        ordenados = valores[orden]
        rango = np.arange(valores.size) - self.inicio[self.columna]
        conservar = (1.0 - FRACCION_RECORTE) * self.largos
        pesos = np.clip(conservar[self.columna] - rango, 0.0, 1.0)
        suma = np.bincount(self.columna, weights=ordenados * pesos, minlength=self.largos.size)
        with np.errstate(invalid="ignore", divide="ignore"):
            mapa = np.where(self.largos > 0, suma / conservar, 0.0)
        return mapa.reshape(self.forma).astype(np.float32)

    def repeticion(self, rng):
        sorteo = self.inicio[self.columna] + (rng.random(self.columna.size) * self.largos[self.columna]).astype(int)
        return self.media_recortada(self.valores[sorteo])

def figura(muestra, original, salida, frecuencia, std_dens, path, mostrar=None,
           etiqueta_entrada="Measured (trimmed mean)"):
    # Fila 1: medida y salidas de la red (ASD mejorada, densidad); fila 2: hierro y bootstrap.
    # mostrar (opcional): mapa que se dibuja como medida (p. ej. el estimador clásico de la tesis)
    var, dens, prob = salida
    nombre = inf.NOMBRES_TESIS.get(muestra, muestra)
    geometria = inf.geometria_de(muestra)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    inf.show_map(axes[0, 0], (original if mostrar is None else mostrar) / inf.VOXEL_CM, 'asd', *inf.ESCALA_ENTRADA_ASD,
                 f"{etiqueta_entrada}: {nombre}", geometria)
    inf.show_map(axes[0, 1], var / inf.VOXEL_CM, 'asd', *inf.escala_asd([var / inf.VOXEL_CM]),
                 "Network: enhanced ASD (N → ∞)", geometria)
    inf.show_map(axes[0, 2], dens, 'dens', 0.0, float(dens.max()), "Network: $\\int dl/X_0$ (PSF removed)", geometria)
    inf.show_map(axes[1, 0], prob, 'prob', 0.0, 1.0, f"Network: iron probability (red: p = {inf.UMBRAL_HIERRO:.2f})", geometria)
    inf.show_map(axes[1, 1], frecuencia, 'freq', 0.0, 1.0,
                 f"Bootstrap ({REPETICIONES} muon resamplings): iron detection frequency", geometria)
    inf.show_map(axes[1, 2], std_dens, 'std', 0.0, float(np.percentile(std_dens, 99.5)),
                 "Bootstrap: statistical uncertainty of $\\int dl/X_0$", geometria)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    with open(f"resultados/{VERSION}/test_evaluation/iron_threshold.json") as f:
        inf.UMBRAL_HIERRO = json.load(f)["iron_threshold"]
    model, device = inf.cargar_modelo(MODELO)
    rng = np.random.default_rng(0)
    resumen = {}
    for muestra in MUESTRAS:
        ruta = voxel_data(muestra)
        cols, _ = columnas(np.load(ruta, allow_pickle=True)["voxel_theta"])
        r = Remuestreo(cols)
        original = r.media_recortada(r.valores)
        salida = inf.inferir_mapa(model, device, original, r.N)
        probs, denss = [], []
        for _ in tqdm(range(REPETICIONES), desc=muestra):
            _, d, p = inf.inferir_mapa(model, device, r.repeticion(rng), r.N)
            probs.append(p)
            denss.append(d)
        probs, denss = np.array(probs), np.array(denss)
        frecuencia = (probs > inf.UMBRAL_HIERRO).mean(axis=0)
        std_dens = denss.std(axis=0)
        base = os.path.join(SALIDA, muestra)
        np.save(f"{base}_detection_frequency.npy", frecuencia.astype(np.float32))
        np.save(f"{base}_density_std.npy", std_dens.astype(np.float32))
        np.save(f"{base}_iron_probability_mean.npy", probs.mean(axis=0).astype(np.float32))
        figura(muestra, original, salida, frecuencia, std_dens, f"{base}_bootstrap.png")
        # Píxeles marcados como hierro por la red: qué fracción es estable (aparece en >= 90%)
        marcado = salida[2] > inf.UMBRAL_HIERRO
        confiable = inf.zona_poco_confiable(original.shape) == False
        resumen[muestra] = {
            "iron_pixels_stable_90pct": float((frecuencia[marcado] >= 0.9).mean()),
            "iron_pixels_unstable_lt50pct": float((frecuencia[marcado] < 0.5).mean()),
            "iron_pixels_stable_90pct_reliable_zone": float((frecuencia[marcado & confiable] >= 0.9).mean()),
            "mean_density_std": float(std_dens[confiable].mean()),
            "file": os.path.basename(ruta),
        }
        print(muestra, resumen[muestra])
    with open(os.path.join(SALIDA, "bootstrap_summary.json"), "w") as f:
        json.dump(resumen, f, indent=2)
