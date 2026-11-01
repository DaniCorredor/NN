import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from Unet import RRDB_UNet_Deterministic
from data_generator import PEDESTAL, limites_normalizacion

# ==============================================================================
# 1. DATASET Y NORMALIZACIÓN 
# ==============================================================================
class MuographyStaticDataset(Dataset):
    def __init__(self, data_dir, files_list=None, is_train=False):
        if files_list is None:
            self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        else:
            self.files = files_list

        self.is_train = is_train

        # Pedestal y límites de normalización compartidos con el generador y la inferencia
        self.PEDESTAL_MEAN = PEDESTAL
        self.MAX_VAR, self.MAX_DENSITY = limites_normalizacion()
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        
        x_var = np.nan_to_num(data['x'], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_var = np.nan_to_num(data['y'], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y_dens = np.nan_to_num(data['density'], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        x_calibrada = np.clip(x_var - self.PEDESTAL_MEAN, 0.0, None)
        y_calibrada_var = np.clip(y_var - self.PEDESTAL_MEAN, 0.0, None)
        y_calibrada_dens = np.clip(y_dens, 0.0, None)
        
        # Mapeo estricto al dominio [0, 1] para ambos cabezales
        norm_x = np.clip(x_calibrada / self.MAX_VAR, 0.0, 1.0).astype(np.float32)
        norm_y_var = np.clip(y_calibrada_var / self.MAX_VAR, 0.0, 1.0).astype(np.float32)
        norm_y_dens = np.clip(y_calibrada_dens / self.MAX_DENSITY, 0.0, 1.0).astype(np.float32)
        
        condition_x = torch.from_numpy(norm_x).float().unsqueeze(0)
        target_y_var = torch.from_numpy(norm_y_var).float().unsqueeze(0)
        target_y_dens = torch.from_numpy(norm_y_dens).float().unsqueeze(0)
        # 112x112 ya es divisible por 8: sin padding

        # Transformaciones geométricas idénticas para los tres tensores: DATA AUGMENTATION
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                condition_x = torch.flip(condition_x, dims=[2]) 
                target_y_var = torch.flip(target_y_var, dims=[2])
                target_y_dens = torch.flip(target_y_dens, dims=[2])
                
            if torch.rand(1).item() > 0.5:
                condition_x = torch.flip(condition_x, dims=[1])
                target_y_var = torch.flip(target_y_var, dims=[1])
                target_y_dens = torch.flip(target_y_dens, dims=[1])
                
            k_rot = torch.randint(0, 4, (1,)).item()
            if k_rot > 0:
                condition_x = torch.rot90(condition_x, k_rot, dims=[1, 2])
                target_y_var = torch.rot90(target_y_var, k_rot, dims=[1, 2])
                target_y_dens = torch.rot90(target_y_dens, k_rot, dims=[1, 2])
                
        return condition_x, target_y_var, target_y_dens
class HybridGradientLoss(nn.Module):
    def __init__(self, alpha=1.0, lambda_grad=2.0):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha
        self.lambda_grad = lambda_grad

    def forward(self, pred, target):
        # 1. Pérdida de magnitud absoluta global (L1)
        loss_l1 = self.l1_loss(pred, target)
        
        # 2. Cálculo de Gradientes (Derivadas espaciales mediante diferencias finitas) para capturar bordes y detalles finos
        # Diferencias en el eje X (horizontal)
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # Diferencias en el eje Y (vertical)
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # 3. Pérdida de Gradiente (Penaliza bordes borrosos)
        loss_grad_x = self.l1_loss(pred_dx, target_dx)
        loss_grad_y = self.l1_loss(pred_dy, target_dy)
        loss_grad = loss_grad_x + loss_grad_y
        
        # 4. Fusión Híbrida
        return (self.alpha * loss_l1) + (self.lambda_grad * loss_grad)
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
# 3. BUCLE PRINCIPAL DE ENTRENAMIENTO MULTI-TAREA
# ==============================================================================
def train_model(data_dir, output_dir="Checkpoints_V7_Denoiser", batch_size=16, epochs=300, lr=3e-4):
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
    
    model = RRDB_UNet_Deterministic(in_channels=1, base_dim=32, num_rrdb=3)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    #criterion = nn.L1Loss().to(device)
    criterion = HybridGradientLoss(alpha=1.0, lambda_grad=2.0).to(device)
    early_stopping = EarlyStopping(patience=300, min_delta=1e-5)
    
    best_val_loss = float('inf')
    history_train_loss = []
    history_val_loss = []
    
    print(f"\nIniciando entrenamiento Multi-Tarea L1 (Varianza + Densidad) por {epochs} épocas...")
    
    for epoch in range(1, epochs + 1):
        # ---------------- ENTRENAMIENTO ----------------
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for condition_x, target_y_var, target_y_dens in pbar:
            condition_x = condition_x.to(device, non_blocking=True)
            target_y_var = target_y_var.to(device, non_blocking=True)
            target_y_dens = target_y_dens.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Predicción dual
            pred_y_var, pred_y_dens = model(condition_x)
            
        
            loss_var = criterion(pred_y_var, target_y_var)
            loss_dens = criterion(pred_y_dens, target_y_dens)
            loss = loss_var + loss_dens
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'L_var': f"{loss_var.item():.4f}", 'L_dens': f"{loss_dens.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
  
        # ---------------- VALIDACIÓN ----------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for condition_x, target_y_var, target_y_dens in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False):
                condition_x = condition_x.to(device, non_blocking=True)
                target_y_var = target_y_var.to(device, non_blocking=True)
                target_y_dens = target_y_dens.to(device, non_blocking=True)
                
                pred_y_var, pred_y_dens = model(condition_x)
                
                loss_var = criterion(pred_y_var, target_y_var)
                loss_dens = criterion(pred_y_dens, target_y_dens)
                loss = loss_var + loss_dens
                
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        #scheduler.step(avg_val_loss) 
        scheduler.step()  
        
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Fin Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} | LR: {current_lr:.2e}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            checkpoint = {
                'epoch': epoch,
                'model': state_dict,
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, os.path.join(output_dir, "best_model.pth"))
            print(f" -> ¡Nuevo mejor modelo guardado! (Val Loss: {best_val_loss:.5f})")
            
        if epoch % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, epoch + 1), history_train_loss, label='Train Loss (Var + Dens)', color='blue', linewidth=2)
            plt.plot(range(1, epoch + 1), history_val_loss, label='Val Loss (Var + Dens)', color='orange', linewidth=2)
            plt.xlabel('Época', fontsize=12)
            plt.ylabel('Pérdida L1 Dual', fontsize=12)
            plt.title('Curva de Entrenamiento Dual RRDB-UNet', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plot_path = os.path.join(output_dir, "training_loss_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\n¡Early Stopping activado en la época {epoch}! El modelo dejó de mejorar. Deteniendo el entrenamiento.")
            break

if __name__ == "__main__":
    DATA_DIR = "datasets/dataset_muografia_V9"
    OUTPUT_DIR = "checkpoints/V12"
    
    train_model(
        data_dir=DATA_DIR, 
        output_dir=OUTPUT_DIR,
        batch_size=16,
        epochs=300,
        lr=5e-4
    )