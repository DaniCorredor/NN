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
# selección del mejor modelo) por tipo de escena y por tiempo de observación. Salidas en inglés
# (figuras para la tesis). Métricas:
#   mae_var, mae_var_no_net   error de la varianza con la red y usando el input directamente
#   mae_dens_bars / _outside  error de densidad dentro / fuera de las barras
#   bar_contrast_recovered    exceso de densidad predicho sobre el concreto / exceso real (1 = completo)
#   iron_dice, precision, recall   salida de probabilidad de hierro con umbral 0.5

def evaluate_test(test_dir, model_path, output_dir, ejemplos_por_tipo=2):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RRDB_UNet_Deterministic(in_channels=2, base_dim=32, num_rrdb=3, con_hierro=True)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v.float() for k, v in checkpoint["model"].items()})
    model = model.to(device).eval()
    print(f"Model from epoch {checkpoint.get('epoch')} (val loss {checkpoint.get('best_val_loss', float('nan')):.5f})")

    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    dataset = MuographyStaticDataset(test_dir, files_list=files, ruido_fijo=True)
    max_var, max_dens = dataset.MAX_VAR, dataset.MAX_DENSITY

    filas, ejemplos = [], {}
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating test set"):
            datos = np.load(files[idx])
            escena, dias = str(datos['scene']), float(datos['dias'])
            T_fe = torch.from_numpy(datos['T_hierro'].astype(np.float32)).unsqueeze(0) / max_dens
            x, y_var, y_dens, _, y_fe = dataset[idx]
            pred_var, pred_dens, logits = model(x.unsqueeze(0).to(device))
            pred_var, pred_dens = pred_var[0].cpu().clamp(min=0), pred_dens[0].cpu().clamp(min=0)
            prob = torch.sigmoid(logits[0]).cpu()
            barras = y_fe > 0.5
            err_dens = (pred_dens - y_dens).abs()
            pred_fe = prob > 0.5
            tp = float((pred_fe & barras).sum())

            # Errores en unidades físicas (rad^2 para la varianza, ∫dl/X0 para la densidad)
            fila = {
                'file': os.path.basename(files[idx]), 'scene': escena, 'days': round(dias, 1),
                'mae_var': float((pred_var - y_var).abs().mean()) * max_var,
                'mae_dens': float(err_dens.mean()) * max_dens,
                'mae_dens_bars': float(err_dens[barras].mean()) * max_dens if barras.any() else np.nan,
                'mae_dens_outside': float(err_dens[~barras].mean()) * max_dens,
                # concreto bajo la barra = densidad real - aporte del hierro
                'bar_contrast_recovered': float((pred_dens - y_dens + T_fe)[barras].sum() / T_fe[barras].sum())
                                          if barras.any() else np.nan,
                'iron_dice': 2 * tp / max(float(pred_fe.sum() + barras.sum()), 1.0) if barras.any() else np.nan,
                'iron_precision': tp / float(pred_fe.sum()) if pred_fe.any() else np.nan,
                'iron_recall': tp / float(barras.sum()) if barras.any() else np.nan,
                'false_iron_fraction': float(pred_fe[~barras].float().mean()),
            }
            filas.append(fila)
            if len(ejemplos.setdefault(escena, [])) < ejemplos_por_tipo:
                ejemplos[escena].append((x[0], pred_dens[0], y_dens[0], prob[0], y_fe[0]))

    with open(os.path.join(output_dir, "metrics_per_sample.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    metricas = ['mae_var', 'mae_dens', 'mae_dens_bars', 'mae_dens_outside', 'bar_contrast_recovered',
                'iron_dice', 'iron_precision', 'iron_recall', 'false_iron_fraction']
    def resumir(grupos, nombre_grupo):
        resumen = []
        for etiqueta, sel in grupos:
            if not sel:
                continue
            fila = {nombre_grupo: etiqueta, 'n': len(sel)}
            fila.update({m: float(np.nanmean([f[m] for f in sel])) if not all(np.isnan(f[m]) for f in sel) else np.nan
                         for m in metricas})
            resumen.append(fila)
        return resumen

    tipos = sorted({f['scene'] for f in filas})
    por_escena = resumir([(t, [f for f in filas if f['scene'] == t]) for t in tipos] + [('TOTAL', filas)], 'scene')
    rangos = [(3, 10), (10, 15), (15, 20), (20, 25.1)]
    por_dias = resumir([(f"{a:g}-{b:g} d", [f for f in filas if a <= f['days'] < b]) for a, b in rangos], 'days')
    for nombre, resumen in [("metrics_per_scene.csv", por_escena), ("metrics_per_days.csv", por_dias)]:
        with open(os.path.join(output_dir, nombre), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
            writer.writeheader()
            writer.writerows(resumen)

    for resumen, clave in [(por_escena, 'scene'), (por_dias, 'days')]:
        print(f"\n{clave:18s} {'n':>4s} {'MAE dens':>9s} {'on bars':>8s} {'outside':>8s} {'contrast':>8s} "
              f"{'Dice':>6s} {'prec.':>6s} {'recall':>6s} {'false Fe':>8s}")
        for r in resumen:
            print(f"{r[clave]:18s} {r['n']:4d} {r['mae_dens']:9.3f} {r['mae_dens_bars']:8.3f} {r['mae_dens_outside']:8.3f} "
                  f"{r['bar_contrast_recovered']:8.2f} {r['iron_dice']:6.2f} {r['iron_precision']:6.2f} "
                  f"{r['iron_recall']:6.2f} {r['false_iron_fraction']:8.4f}")

    # Resumen por escena
    sin_total = [r for r in por_escena[:-1] if not np.isnan(r['iron_dice'])]
    pos = np.arange(len(sin_total))
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].bar(pos - 0.2, [r['mae_dens_bars'] for r in sin_total], 0.4, label='On bars', color='tab:red')
    axes[0].bar(pos + 0.2, [r['mae_dens_outside'] for r in sin_total], 0.4, label='Outside bars', color='tab:green')
    axes[0].set_title('Density mean absolute error [$\\int dl/X_0$]')
    axes[0].legend()
    axes[1].bar(pos, [r['bar_contrast_recovered'] for r in sin_total], 0.6, color='tab:purple')
    axes[1].axhline(1.0, color='k', linestyle='--')
    axes[1].set_title('Bar density contrast recovered (1 = complete)')
    for k, (m, c) in enumerate([('iron_dice', 'tab:blue'), ('iron_precision', 'tab:orange'), ('iron_recall', 'tab:gray')]):
        axes[2].bar(pos + (k - 1) * 0.27, [r[m] for r in sin_total], 0.27, label=m.replace('iron_', '').capitalize(), color=c)
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Iron probability map (threshold 0.5)')
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels([r['scene'] for r in sin_total], rotation=30, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_per_scene.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Ejemplos por tipo de escena (x horizontal, y vertical); escala mínima fija para el aire
    lista = [(tipo, e) for tipo in tipos for e in ejemplos[tipo]]
    columnas = ['Input (normalized)', 'Pred. density', 'Target density', 'Iron probability', 'True iron']
    fig, axes = plt.subplots(len(lista), 5, figsize=(15, 3 * len(lista)), squeeze=False)
    for fila, (tipo, imagenes) in enumerate(lista):
        vmax_dens = float(max(imagenes[2].max(), 0.02))
        escalas = [float(np.percentile(imagenes[0].numpy(), 99)), vmax_dens, vmax_dens, 1.0, 1.0]
        for col, img in enumerate(imagenes):
            axes[fila, col].imshow(img.numpy().T, origin='lower', cmap=['inferno', 'viridis', 'viridis', 'gray', 'gray'][col],
                                   vmin=0, vmax=escalas[col])
            axes[fila, col].set_xticks([])
            axes[fila, col].set_yticks([])
            if fila == 0:
                axes[fila, col].set_title(columnas[col])
        axes[fila, 0].set_ylabel(tipo)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "examples_per_scene.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\nResults saved in '{output_dir}'")

if __name__ == "__main__":
    evaluate_test(
        test_dir="datasets/dataset_muografia_Test_V11",
        model_path="checkpoints/V14/best_model.pth",
        output_dir="resultados/V14/test_evaluation",
    )
