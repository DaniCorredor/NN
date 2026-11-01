import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from Unet import RRDB_UNet_Deterministic
from data_generator import PEDESTAL, limites_normalizacion

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
# BUCLE DE INFERENCIA DUAL (VARIANZA Y DENSIDAD)
# ==============================================================================
def run_deterministic_inference(input_folder, output_folder, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Cargando modelo dual RRDB-UNet desde: {model_path}")
   
    model = RRDB_UNet_Deterministic(in_channels=1, base_dim=32, num_rrdb=3)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.to(device)
    model.eval()

    npy_files = sorted(glob.glob(os.path.join(input_folder, "*_var_xy.npy")))
    if not npy_files:
        print(f"No se encontraron archivos *_var_xy.npy en {input_folder}")
        return

    # --- LÍMITES DE NORMALIZACIÓN (compartidos con el generador) ---
    PEDESTAL_MEAN = PEDESTAL
    MAX_VAR, MAX_DENSITY = limites_normalizacion()

    with torch.no_grad():
        for file_path in tqdm(npy_files, desc="Procesando Inferencia Dual Geant4"):
            base_name = os.path.basename(file_path)
            
            mutrec_raw_np = np.load(file_path).astype(np.float32)
            mutrec_raw_np = np.nan_to_num(mutrec_raw_np, nan=0.0, posinf=0.0, neginf=0.0)
            mutrec_raw_np = np.clip(mutrec_raw_np, 0.0, None)
            # Los *_var_xy.npy ya vienen en varianza (Σθ²/ΣN por columna)
            mutrec_variance = mutrec_raw_np

            # --- 1. Calibración y Normalización ---
            x_calibrada = np.clip(mutrec_variance - PEDESTAL_MEAN, 0.0, None)
            norm_x = np.clip(x_calibrada / MAX_VAR, 0.0, 1.0).astype(np.float32)

            # --- 2. Inferencia de la Red Neuronal (Doble Salida) ---
            condition_tensor = torch.from_numpy(norm_x).float().unsqueeze(0).unsqueeze(0)
            condition_padded, padding_info = pad_to_multiple(condition_tensor, multiple=16)
            condition_padded = condition_padded.to(device)
            
            y_pred_var_padded, y_pred_dens_padded = model(condition_padded)
            
            y_pred_var_tensor = unpad(y_pred_var_padded, padding_info)
            y_pred_dens_tensor = unpad(y_pred_dens_padded, padding_info)
            
            y_pred_var_norm = np.clip(y_pred_var_tensor.cpu().numpy()[0, 0], 0.0, None)
            y_pred_dens_norm = np.clip(y_pred_dens_tensor.cpu().numpy()[0, 0], 0.0, None)
            
            # --- 3. Desnormalización (Output Final) ---
            variance_clean = (y_pred_var_norm * MAX_VAR) + PEDESTAL_MEAN
            density_clean = y_pred_dens_norm * MAX_DENSITY
            
            # ========================================================
            # GUARDADO Y VISUALIZACIÓN
            # ========================================================
            save_base_name = base_name.replace(".npy", "")
            np.savetxt(os.path.join(output_folder, f"{save_base_name}_variance_clean.txt"), variance_clean, fmt="%.8f")
            np.savetxt(os.path.join(output_folder, f"{save_base_name}_density_clean.txt"), density_clean, fmt="%.6f")
            
            plt.figure(figsize=(18, 5))
            
            # Subplot 1: Input (Ruido Geant4)
            plt.subplot(1, 3, 1)
            plt.imshow(mutrec_variance, cmap='inferno', vmin=0.0)
            plt.colorbar(label='Varianza Angular (rad\(^2\))')
            plt.title(f"Input: {base_name}\n(Baja Estadística)")
            
            # Subplot 2: Predicción de Varianza (Denoiser)
            plt.subplot(1, 3, 2)
            plt.imshow(variance_clean, cmap='inferno', vmin=0.0)
            plt.colorbar(label='Varianza Angular (rad\(^2\))')
            plt.title("Superresolución Varianza (RRDB)")
            
            # Subplot 3: Predicción de Densidad (Deconvolución)
            plt.subplot(1, 3, 3)
            plt.imshow(density_clean, cmap='viridis', vmin=0.0)
            plt.colorbar(label='Integral Física \(1/X_0\)')
            plt.title("Topología Densidad (Sin PSF)")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, f"{save_base_name}_dual_inference.png"), dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    INPUT_DIR = "datos_reales"   # salida de preparar_datos_reales.py
    OUTPUT_DIR = "resultados/V12"
    MODEL_PATH = "checkpoints/V12/best_model.pth"

    run_deterministic_inference(
        input_folder=INPUT_DIR,
        output_folder=OUTPUT_DIR,
        model_path=MODEL_PATH
    )