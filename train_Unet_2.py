import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")  # servidor sin pantalla
import matplotlib.pyplot as plt

from Unet import RRDB_UNet_Deterministic
from data_generator import (PEDESTAL, ACEPTANCIA_PATH, VOXEL_CM, limites_normalizacion, agregar_ruido,
                            normalizar_muones, normalizar_entrada)

# Peso extra de la pérdida en las columnas con hierro: la prioridad es reconstruir las barras
PESO_BARRAS = 4.0
# Peso de la salida de probabilidad de hierro frente a las de varianza y densidad
PESO_HIERRO = 0.1

# ==============================================================================
# 1. DATASET Y NORMALIZACIÓN 
# ==============================================================================
class MuographyStaticDataset(Dataset):
    # Cada archivo guarda la varianza esperada sin ruido (x_limpio). El ruido se genera al cargar:
    #   is_train=True   -> augmentation + ruido nuevo cada vez (la red no puede memorizarlo)
    #   is_train=False  -> ruido con semilla fija por muestra (validación comparable entre épocas)
    #   ruido_fijo=True -> usa la medición guardada en el archivo (conjunto de prueba)
    def __init__(self, data_dir, files_list=None, is_train=False, ruido_fijo=False):
        if files_list is None:
            self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        else:
            self.files = files_list

        self.is_train = is_train
        self.ruido_fijo = ruido_fijo
        self.aceptancia = np.load(ACEPTANCIA_PATH).astype(np.float32)

        # Pedestal y límites de normalización compartidos con el generador y la inferencia
        self.PEDESTAL_MEAN = PEDESTAL
        self.MAX_VAR, self.MAX_DENSITY = limites_normalizacion()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        mapas = [np.nan_to_num(data[k], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                 for k in ('x_limpio', 'y', 'density', 'mascara', 'hierro')]

        if self.ruido_fijo:
            x_var, muones = data['x'].astype(np.float32), data['muones'].astype(np.float32)
            _, y_var, y_dens, mascara, hierro = mapas
        else:
            # Transformaciones geométricas sobre la escena limpia, ANTES del ruido: el mapa de tasa
            # de muones queda fijo en coordenadas del detector, como en la realidad
            if self.is_train:
                if np.random.rand() > 0.5:
                    mapas = [np.flip(m, axis=1) for m in mapas]
                if np.random.rand() > 0.5:
                    mapas = [np.flip(m, axis=0) for m in mapas]
                k_rot = np.random.randint(0, 4)
                if k_rot > 0:
                    mapas = [np.rot90(m, k_rot) for m in mapas]
            x_limpio, y_var, y_dens, mascara, hierro = [np.ascontiguousarray(m) for m in mapas]
            rng = np.random.default_rng() if self.is_train else np.random.default_rng(idx)
            x_var, muones, _ = agregar_ruido(x_limpio, self.aceptancia, rng)

        y_calibrada_var = np.clip(y_var - self.PEDESTAL_MEAN, 0.0, None)
        y_calibrada_dens = np.clip(y_dens, 0.0, None)

        # Mapeo estricto al dominio [0, 1] para ambos cabezales
        norm_x = normalizar_entrada(x_var)
        norm_y_var = np.clip(y_calibrada_var / self.MAX_VAR, 0.0, 1.0).astype(np.float32)
        norm_y_dens = np.clip(y_calibrada_dens / self.MAX_DENSITY, 0.0, 1.0).astype(np.float32)

        # Entrada de 2 canales: varianza medida y muones por píxel (dice a la red qué tan confiable es cada píxel)
        condition_x = torch.from_numpy(np.stack([norm_x, normalizar_muones(muones)])).float()
        target_y_var = torch.from_numpy(norm_y_var).float().unsqueeze(0)
        target_y_dens = torch.from_numpy(norm_y_dens).float().unsqueeze(0)
        peso = torch.from_numpy(1.0 + PESO_BARRAS * mascara).float().unsqueeze(0)
        target_hierro = torch.from_numpy(hierro).float().unsqueeze(0)
        # 112x112 ya es divisible por 8: sin padding

        return condition_x, target_y_var, target_y_dens, peso, target_hierro
class HybridGradientLoss(nn.Module):
    def __init__(self, alpha=1.0, lambda_grad=2.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.lambda_grad = lambda_grad

    def forward(self, pred, target, peso=None):
        # peso: mapa por píxel (más alto en las barras); se normaliza para no cambiar la escala de la pérdida
        if peso is None:
            peso = torch.ones_like(target)
        norma = peso.mean()

        # 1. Pérdida de magnitud absoluta (L1) ponderada
        loss_l1 = (peso * (pred - target).abs()).mean() / norma

        # 2. Cálculo de Gradientes (Derivadas espaciales mediante diferencias finitas) para capturar bordes y detalles finos
        # Diferencias en el eje X (horizontal)
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        # Diferencias en el eje Y (vertical)
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # 3. Pérdida de Gradiente (Penaliza bordes borrosos)
        loss_grad_x = (peso[:, :, :, 1:] * (pred_dx - target_dx).abs()).mean() / norma
        loss_grad_y = (peso[:, :, 1:, :] * (pred_dy - target_dy).abs()).mean() / norma
        loss_grad = loss_grad_x + loss_grad_y

        # 4. Fusión Híbrida
        return (self.alpha * loss_l1) + (self.lambda_grad * loss_grad)
def _soft_erode(img):
    return torch.min(-F.max_pool2d(-img, (3, 1), 1, (1, 0)), -F.max_pool2d(-img, (1, 3), 1, (0, 1)))

def _soft_skeleton(img, iteraciones=6):
    # Esqueleto diferenciable (Shit et al. 2021, clDice): la línea central de cada barra
    abierto = F.max_pool2d(_soft_erode(img), 3, 1, 1)
    esqueleto = F.relu(img - abierto)
    for _ in range(iteraciones):
        img = _soft_erode(img)
        abierto = F.max_pool2d(_soft_erode(img), 3, 1, 1)
        delta = F.relu(img - abierto)
        esqueleto = esqueleto + F.relu(delta - esqueleto * delta)
    return esqueleto

class IronSegmentationLoss(nn.Module):
    # BCE + Dice + clDice sobre la probabilidad de hierro. clDice premia que la línea central de
    # cada barra salga continua y recta: castiga barras cortadas, curvas o inventadas.
    def __init__(self, peso_cldice=0.5):
        super().__init__()
        self.peso_cldice = peso_cldice

    def forward(self, logits, objetivo):
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, objetivo)
        dice = (2 * (prob * objetivo).sum() + 1.0) / (prob.sum() + objetivo.sum() + 1.0)
        esq_pred, esq_obj = _soft_skeleton(prob), _soft_skeleton(objetivo)
        precision = ((esq_pred * objetivo).sum() + 1.0) / (esq_pred.sum() + 1.0)
        sensibilidad = ((esq_obj * prob).sum() + 1.0) / (esq_obj.sum() + 1.0)
        cldice = 2 * precision * sensibilidad / (precision + sensibilidad)
        return bce + (1 - dice) + self.peso_cldice * (1 - cldice)

# ==============================================================================
# 2. EARLY STOPPING
# ==============================================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            print(f"EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False

# ==============================================================================
# 3. SEGUIMIENTO DEL ENTRENAMIENTO (curvas y muestras fijas de validación)
# ==============================================================================
def plot_loss_curves(historial, best_epoch, path):
    epocas = [h['epoca'] for h in historial]
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    for ax, clave, titulo in zip(axes, ['total', 'var', 'dens', 'hierro'],
                                 ['Total loss', 'Variance head', 'Density head ($\\int 1/X_0$)', 'Iron probability head']):
        ax.plot(epocas, [h[f'train_{clave}'] for h in historial], label='Train', color='blue', linewidth=2)
        ax.plot(epocas, [h[f'val_{clave}'] for h in historial], label='Validation', color='orange', linewidth=2)
        ax.axvline(best_epoch, color='green', linestyle='--', label=f'Best epoch ({best_epoch})')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(titulo)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
    # Lo que más importa: calidad de las barras en validación
    ax = axes[4]
    ax.plot(epocas, [h['val_dice'] for h in historial], color='purple', linewidth=2, label='Iron Dice (p > 0.5)')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax2 = ax.twinx()
    ax2.plot(epocas, [h['val_mae_barras'] for h in historial], color='gray', linewidth=1.5, label='Density MAE on bars')
    ax2.set_ylabel('Density MAE on bars [$\\int dl/X_0$]')
    ax.axvline(best_epoch, color='green', linestyle='--')
    ax.set_title('Validation: bar reconstruction')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

def pick_fixed_samples(dataset, tipos=('malla', 'malla', 'grosores', 'grosores', 'barras')):
    # Una muestra de validación por tipo de escena; siempre las mismas para ver la evolución
    elegidas = []
    for tipo in tipos:
        for idx, ruta in enumerate(dataset.files):
            if str(np.load(ruta)['scene']) == tipo and (tipo, idx) not in elegidas:
                elegidas.append((tipo, idx))
                break
    return elegidas

def plot_fixed_samples(model, muestras, device, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset, items = muestras['dataset'], muestras['items']
    columnas = ['Input (normalized)', 'Pred. variance', 'Target variance', 'Pred. density', 'Target density',
                'Iron probability', 'True iron']
    fig, axes = plt.subplots(len(items), 7, figsize=(21, 3 * len(items)), squeeze=False)
    model.eval()
    with torch.no_grad():
        for fila, (tipo, idx) in enumerate(items):
            x, y_var, y_dens, _, y_fe = dataset[idx]
            pred_var, pred_dens, logits = model(x.unsqueeze(0).to(device))
            imagenes = [x[0], pred_var[0, 0].cpu(), y_var[0], pred_dens[0, 0].cpu(), y_dens[0],
                        torch.sigmoid(logits[0, 0]).cpu(), y_fe[0]]
            # Predicción y target con la misma escala para compararlos directamente
            escalas = [float(np.percentile(x[0].numpy(), 99))] + [float(max(y_var.max(), 0.02))] * 2 + \
                      [float(max(y_dens.max(), 0.02))] * 2 + [1.0, 1.0]
            mapas_color = ['inferno'] * 3 + ['viridis'] * 2 + ['gray'] * 2
            for col, img in enumerate(imagenes):
                # Transpuesta y origin='lower': x horizontal, y vertical, como los mapas reales
                axes[fila, col].imshow(img.numpy().T, origin='lower', cmap=mapas_color[col], vmin=0, vmax=escalas[col])
                axes[fila, col].set_xticks([])
                axes[fila, col].set_yticks([])
                if fila == 0:
                    axes[fila, col].set_title(columnas[col])
            axes[fila, 0].set_ylabel(tipo)
    fig.suptitle(f'Fixed validation samples - epoch {epoch}')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()

# ==============================================================================
# 4. BUCLE PRINCIPAL DE ENTRENAMIENTO MULTI-TAREA
# ==============================================================================
def train_model(data_dir, output_dir="Checkpoints_V7_Denoiser", batch_size=16, epochs=200, lr=3e-4, patience=25):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    all_files = [
        os.path.join(data_dir, filename)
        for filename in sorted(os.listdir(data_dir))
        if filename.endswith(".npz")
    ]

    # Las oclusiones ahora son 20% del dataset (build_scene), ya no se excluyen
    print(f"Archivos incluidos: {len(all_files)}")
    np.random.seed(42) 
    np.random.shuffle(all_files)
    
    train_size = int(0.7 * len(all_files))
    train_files = all_files[:train_size]
    val_files = all_files[train_size:]
    
    train_dataset = MuographyStaticDataset(data_dir, files_list=train_files, is_train=True)
    val_dataset = MuographyStaticDataset(data_dir, files_list=val_files, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = RRDB_UNet_Deterministic(in_channels=2, base_dim=32, num_rrdb=3, con_hierro=True)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    #criterion = nn.L1Loss().to(device)
    criterion = HybridGradientLoss(alpha=1.0, lambda_grad=2.0).to(device)
    criterio_hierro = IronSegmentationLoss().to(device)
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-6)
    
    best_val_loss = float('inf')
    best_epoch = 0
    historial = []
    muestras_fijas = {'dataset': val_dataset, 'items': pick_fixed_samples(val_dataset)}

    print(f"\nIniciando entrenamiento Multi-Tarea L1 (Varianza + Densidad) por {epochs} épocas...")

    for epoch in range(1, epochs + 1):
        t_inicio = time.time()
        # ---------------- ENTRENAMIENTO ----------------
        model.train()
        train_var, train_dens, train_fe = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for condition_x, target_y_var, target_y_dens, peso, target_fe in pbar:
            condition_x = condition_x.to(device, non_blocking=True)
            target_y_var = target_y_var.to(device, non_blocking=True)
            target_y_dens = target_y_dens.to(device, non_blocking=True)
            peso = peso.to(device, non_blocking=True)
            target_fe = target_fe.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Predicción triple: varianza, densidad y probabilidad de hierro
            pred_y_var, pred_y_dens, logits_fe = model(condition_x)

            loss_var = criterion(pred_y_var, target_y_var, peso)
            loss_dens = criterion(pred_y_dens, target_y_dens, peso)
            loss_fe = criterio_hierro(logits_fe, target_fe)
            loss = loss_var + loss_dens + PESO_HIERRO * loss_fe

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_var += loss_var.item()
            train_dens += loss_dens.item()
            train_fe += PESO_HIERRO * loss_fe.item()
            pbar.set_postfix({'L_var': f"{loss_var.item():.4f}", 'L_dens': f"{loss_dens.item():.4f}",
                              'L_fe': f"{loss_fe.item():.4f}"})

        train_var /= len(train_loader)
        train_dens /= len(train_loader)
        train_fe /= len(train_loader)

        # ---------------- VALIDACIÓN ----------------
        model.eval()
        val_var, val_dens, val_fe = 0.0, 0.0, 0.0
        err_barras, px_barras = 0.0, 0.0
        inter, suma = 0.0, 0.0
        with torch.no_grad():
            for condition_x, target_y_var, target_y_dens, peso, target_fe in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                condition_x = condition_x.to(device, non_blocking=True)
                target_y_var = target_y_var.to(device, non_blocking=True)
                target_y_dens = target_y_dens.to(device, non_blocking=True)
                peso = peso.to(device, non_blocking=True)
                target_fe = target_fe.to(device, non_blocking=True)

                pred_y_var, pred_y_dens, logits_fe = model(condition_x)

                val_var += criterion(pred_y_var, target_y_var, peso).item()
                val_dens += criterion(pred_y_dens, target_y_dens, peso).item()
                val_fe += PESO_HIERRO * criterio_hierro(logits_fe, target_fe).item()
                binaria = (logits_fe > 0).float()
                inter += (binaria * target_fe).sum().item()
                suma += (binaria.sum() + target_fe.sum()).item()
                en_barras = (peso > 1).float()
                err_barras += ((pred_y_dens - target_y_dens).abs() * en_barras).sum().item()
                px_barras += en_barras.sum().item()

        val_var /= len(val_loader)
        val_dens /= len(val_loader)
        val_fe /= len(val_loader)
        val_dice = 2 * inter / max(suma, 1.0)
        val_mae_barras = err_barras / max(px_barras, 1.0) * val_dataset.MAX_DENSITY
        avg_train_loss = train_var + train_dens + train_fe
        avg_val_loss = val_var + val_dens + val_fe
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Fin Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | "
              f"Val MAE barras: {val_mae_barras:.4f} | Val Dice hierro: {val_dice:.3f} | LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
            print(f" -> ¡Nuevo mejor modelo guardado! (Val Loss: {best_val_loss:.5f})")

        # Historial por época (se reescribe completo: queda legible aunque se corte el entrenamiento)
        historial.append({
            'epoca': epoch, 'train_total': avg_train_loss, 'val_total': avg_val_loss,
            'train_var': train_var, 'val_var': val_var, 'train_dens': train_dens, 'val_dens': val_dens,
            'train_hierro': train_fe, 'val_hierro': val_fe, 'val_dice': val_dice,
            'val_mae_barras': val_mae_barras,
            'lr': current_lr, 'segundos': time.time() - t_inicio, 'mejor': int(epoch == best_epoch)
        })
        with open(os.path.join(output_dir, "historial.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
            writer.writeheader()
            writer.writerows(historial)

        early_stopping(avg_val_loss)
        ultima = epoch == epochs or early_stopping.early_stop

        if epoch % 5 == 0 or ultima:
            plot_loss_curves(historial, best_epoch, os.path.join(output_dir, "training_loss_curve.png"))

        if epoch % 10 == 0 or epoch == 1 or ultima:
            plot_fixed_samples(model, muestras_fijas, device, epoch,
                               os.path.join(output_dir, "muestras_val", f"epoca_{epoch:03d}.png"))

        if early_stopping.early_stop:
            print(f"\n¡Early Stopping activado en la época {epoch}! El modelo dejó de mejorar. Deteniendo el entrenamiento.")
            break

if __name__ == "__main__":
    DATA_DIR = "datasets/dataset_muografia_V11"
    OUTPUT_DIR = "checkpoints/V15"
    
    train_model(
        data_dir=DATA_DIR, 
        output_dir=OUTPUT_DIR,
        batch_size=16,
        epochs=200,
        lr=5e-4,
        patience=25
    )