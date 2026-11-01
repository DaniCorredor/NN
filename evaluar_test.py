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
# selección del mejor modelo) y reporta el error por tipo de escena.
# Línea base "sin red": usar el input ruidoso directamente como estimación de la varianza limpia.

def evaluate_test(test_dir, model_path, output_dir, ejemplos_por_tipo=2):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = RRDB_UNet_Deterministic(in_channels=1, base_dim=32, num_rrdb=3)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["model"].items()})
    model = model.to(device).eval()
    print(f"Modelo de la época {checkpoint.get('epoch')} (val loss {checkpoint.get('best_val_loss', float('nan')):.5f})")

    files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    dataset = MuographyStaticDataset(test_dir, files_list=files, is_train=False)
    max_var, max_dens = dataset.MAX_VAR, dataset.MAX_DENSITY

    filas, ejemplos = [], {}
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluando test"):
            escena = str(np.load(files[idx])['scene'])
            x, y_var, y_dens = dataset[idx]
            pred_var, pred_dens = model(x.unsqueeze(0).to(device))
            pred_var, pred_dens = pred_var[0].cpu().clamp(min=0), pred_dens[0].cpu().clamp(min=0)

            # Errores en unidades físicas (rad^2 para la varianza, ∫dl/X0 para la densidad)
            filas.append({
                'archivo': os.path.basename(files[idx]), 'escena': escena,
                'mae_var': float((pred_var - y_var).abs().mean()) * max_var,
                'mae_var_sin_red': float((x - y_var).abs().mean()) * max_var,
                'mae_dens': float((pred_dens - y_dens).abs().mean()) * max_dens,
            })
            if len(ejemplos.setdefault(escena, [])) < ejemplos_por_tipo:
                ejemplos[escena].append((x[0], pred_var[0], y_var[0], pred_dens[0], y_dens[0]))

    with open(os.path.join(output_dir, "metricas_por_muestra.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    # Resumen por tipo de escena
    tipos = sorted({f['escena'] for f in filas})
    resumen = []
    for tipo in tipos + ['TOTAL']:
        sel = [f for f in filas if tipo == 'TOTAL' or f['escena'] == tipo]
        resumen.append({
            'escena': tipo, 'n': len(sel),
            'mae_var': np.mean([f['mae_var'] for f in sel]),
            'mae_var_sin_red': np.mean([f['mae_var_sin_red'] for f in sel]),
            'mae_dens': np.mean([f['mae_dens'] for f in sel]),
        })
    with open(os.path.join(output_dir, "metricas_por_escena.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        writer.writeheader()
        writer.writerows(resumen)

    print(f"\n{'escena':20s} {'n':>4s} {'MAE var':>10s} {'sin red':>10s} {'mejora':>7s} {'MAE dens':>9s}")
    for r in resumen:
        mejora = r['mae_var_sin_red'] / max(r['mae_var'], 1e-12)
        print(f"{r['escena']:20s} {r['n']:4d} {r['mae_var']:10.2e} {r['mae_var_sin_red']:10.2e} {mejora:6.1f}x {r['mae_dens']:9.3f}")

    # Barras de error por escena
    sin_total = resumen[:-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pos = np.arange(len(sin_total))
    axes[0].bar(pos - 0.2, [r['mae_var_sin_red'] for r in sin_total], 0.4, label='Sin red (input)', color='gray')
    axes[0].bar(pos + 0.2, [r['mae_var'] for r in sin_total], 0.4, label='Red', color='tab:orange')
    axes[0].set_title('Error medio absoluto - varianza [rad$^2$]')
    axes[1].bar(pos, [r['mae_dens'] for r in sin_total], 0.6, color='tab:green')
    axes[1].set_title('Error medio absoluto - densidad [$\\int dl/X_0$]')
    for ax in axes:
        ax.set_xticks(pos)
        ax.set_xticklabels([r['escena'] for r in sin_total], rotation=30, ha='right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_por_escena.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Ejemplos por tipo de escena: input / predicción / target
    lista = [(tipo, e) for tipo in tipos for e in ejemplos[tipo]]
    columnas = ['Input', 'Pred. varianza', 'Target varianza', 'Pred. densidad', 'Target densidad']
    fig, axes = plt.subplots(len(lista), 5, figsize=(15, 3 * len(lista)), squeeze=False)
    for fila, (tipo, imagenes) in enumerate(lista):
        vmax_var, vmax_dens = float(max(imagenes[2].max(), 1e-6)), float(max(imagenes[4].max(), 1e-6))
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
        test_dir="datasets/dataset_muografia_Test_V9",
        model_path="checkpoints/V12/best_model.pth",
        output_dir="resultados/V12/evaluacion_test",
    )
