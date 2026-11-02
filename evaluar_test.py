import os
import csv
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from Unet import RRDB_UNet_Deterministic
from train_Unet_2 import MuographyStaticDataset

# Evalúa best_model.pth sobre el conjunto de PRUEBA (nunca visto en entrenamiento ni en la
# selección del mejor modelo) y reporta el error por tipo de escena y por tiempo de observación.
# Línea base "sin red": usar el input ruidoso directamente como estimación de la varianza limpia.
# Métricas de barras (máscara del generador = columnas con hierro):
#   mae_dens_barras  error de densidad dentro de las barras
#   mae_dens_fuera   error de densidad fuera de ellas (estructuras inventadas o perdidas)
#   recuperacion     densidad predicha / real sumada en las barras (1 = contraste completo)

def evaluate_test(test_dir, model_path, output_dir, ejemplos_por_tipo=2):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RRDB_UNet_Deterministic(in_channels=2, base_dim=32, num_rrdb=3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["model"].items()})
    model = model.to(device).eval()
    print(f"Modelo de la época {checkpoint.get('epoch')} (val loss {checkpoint.get('best_val_loss', float('nan')):.5f})")

    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    dataset = MuographyStaticDataset(test_dir, files_list=files, ruido_fijo=True)
    max_var, max_dens = dataset.MAX_VAR, dataset.MAX_DENSITY

    filas, ejemplos = [], {}
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluando test"):
            datos = np.load(files[idx])
            escena, dias = str(datos['scene']), float(datos['dias'])
            x, y_var, y_dens, peso = dataset[idx]
            pred_var, pred_dens = model(x.unsqueeze(0).to(device))
            pred_var, pred_dens = pred_var[0].cpu().clamp(min=0), pred_dens[0].cpu().clamp(min=0)
            barras = peso > 1
            err_dens = (pred_dens - y_dens).abs()

            # Errores en unidades físicas (rad^2 para la varianza, ∫dl/X0 para la densidad)
            fila = {
                'archivo': os.path.basename(files[idx]), 'escena': escena, 'dias': round(dias, 1),
                'mae_var': float((pred_var - y_var).abs().mean()) * max_var,
                'mae_var_sin_red': float((x[:1] - y_var).abs().mean()) * max_var,
                'mae_dens': float(err_dens.mean()) * max_dens,
                'mae_dens_barras': float(err_dens[barras].mean()) * max_dens if barras.any() else np.nan,
                'mae_dens_fuera': float(err_dens[~barras].mean()) * max_dens,
                'recuperacion': float(pred_dens[barras].sum() / y_dens[barras].sum()) if barras.any() else np.nan,
            }
            filas.append(fila)
            if len(ejemplos.setdefault(escena, [])) < ejemplos_por_tipo:
                ejemplos[escena].append((x[0], pred_var[0], y_var[0], pred_dens[0], y_dens[0]))

    with open(os.path.join(output_dir, "metricas_por_muestra.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    metricas = ['mae_var', 'mae_var_sin_red', 'mae_dens', 'mae_dens_barras', 'mae_dens_fuera', 'recuperacion']
    def resumir(grupos, nombre_grupo):
        resumen = []
        for etiqueta, sel in grupos:
            if not sel:
                continue
            fila = {nombre_grupo: etiqueta, 'n': len(sel)}
            fila.update({m: float(np.nanmean([f[m] for f in sel])) for m in metricas})
            resumen.append(fila)
        return resumen

    # Resumen por tipo de escena y por tiempo de observación
    tipos = sorted({f['escena'] for f in filas})
    por_escena = resumir([(t, [f for f in filas if f['escena'] == t]) for t in tipos] + [('TOTAL', filas)], 'escena')
    rangos = [(3, 10), (10, 15), (15, 20), (20, 25.1)]
    por_dias = resumir([(f"{a:g}-{b:g} d", [f for f in filas if a <= f['dias'] < b]) for a, b in rangos], 'dias')
    for nombre, resumen in [("metricas_por_escena.csv", por_escena), ("metricas_por_dias.csv", por_dias)]:
        with open(os.path.join(output_dir, nombre), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
            writer.writeheader()
            writer.writerows(resumen)

    for resumen, clave in [(por_escena, 'escena'), (por_dias, 'dias')]:
        print(f"\n{clave:20s} {'n':>4s} {'MAE var':>10s} {'sin red':>10s} {'MAE dens':>9s} {'en barras':>10s} {'fuera':>8s} {'recup.':>7s}")
        for r in resumen:
            print(f"{r[clave]:20s} {r['n']:4d} {r['mae_var']:10.2e} {r['mae_var_sin_red']:10.2e} {r['mae_dens']:9.3f} "
                  f"{r['mae_dens_barras']:10.3f} {r['mae_dens_fuera']:8.3f} {r['recuperacion']:7.2f}")

    # Barras de error por escena
    sin_total = por_escena[:-1]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    pos = np.arange(len(sin_total))
    axes[0].bar(pos - 0.2, [r['mae_var_sin_red'] for r in sin_total], 0.4, label='Sin red (input)', color='gray')
    axes[0].bar(pos + 0.2, [r['mae_var'] for r in sin_total], 0.4, label='Red', color='tab:orange')
    axes[0].set_title('Error medio absoluto - varianza [rad$^2$]')
    axes[0].legend()
    axes[1].bar(pos - 0.2, [r['mae_dens_barras'] for r in sin_total], 0.4, label='En barras', color='tab:red')
    axes[1].bar(pos + 0.2, [r['mae_dens_fuera'] for r in sin_total], 0.4, label='Fuera de barras', color='tab:green')
    axes[1].set_title('Error medio absoluto - densidad [$\\int dl/X_0$]')
    axes[1].legend()
    axes[2].bar(pos, [r['recuperacion'] for r in sin_total], 0.6, color='tab:purple')
    axes[2].axhline(1.0, color='k', linestyle='--')
    axes[2].set_title('Densidad recuperada en las barras (1 = completa)')
    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels([r['escena'] for r in sin_total], rotation=30, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_por_escena.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Ejemplos por tipo de escena: input / predicción / target.
    # Escala mínima fija: en escenas de aire el target es 0 y una escala automática exagera valores ~0
    lista = [(tipo, e) for tipo in tipos for e in ejemplos[tipo]]
    columnas = ['Input', 'Pred. varianza', 'Target varianza', 'Pred. densidad', 'Target densidad']
    fig, axes = plt.subplots(len(lista), 5, figsize=(15, 3 * len(lista)), squeeze=False)
    for fila, (tipo, imagenes) in enumerate(lista):
        vmax_var, vmax_dens = float(max(imagenes[2].max(), 0.02)), float(max(imagenes[4].max(), 0.02))
        for col, img in enumerate(imagenes):
            axes[fila, col].imshow(img.numpy(), cmap='inferno' if col < 3 else 'viridis',
                                   vmin=0, vmax=vmax_var if col < 3 else vmax_dens)
            axes[fila, col].set_xticks([])
            axes[fila, col].set_yticks([])
            if fila == 0:
                axes[fila, col].set_title(columnas[col])
        axes[fila, 0].set_ylabel(tipo)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ejemplos_por_escena.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nResultados guardados en '{output_dir}'")

if __name__ == "__main__":
    evaluate_test(
        test_dir="datasets/dataset_muografia_Test_V10",
        model_path="checkpoints/V13/best_model.pth",
        output_dir="resultados/V13/evaluacion_test",
    )
