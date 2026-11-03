import os
import sys
import torch

# Guarda solo los pesos del modelo en float16 (~74 MB) para poder subirlos a GitHub:
# el best_model.pth completo (con el estado del optimizador) pasa el límite de 100 MB.
# Uso (en el servidor):  python3 exportar_pesos.py V13

version = sys.argv[1] if len(sys.argv) > 1 else "V13"
origen = os.path.join("checkpoints", version, "best_model.pth")
destino = os.path.join("modelos", f"{version}_fp16.pt")

checkpoint = torch.load(origen, map_location="cpu", weights_only=True)
pesos = {k.replace('module.', ''): v.half() if v.is_floating_point() else v for k, v in checkpoint["model"].items()}

os.makedirs("modelos", exist_ok=True)
torch.save({"model": pesos, "epoch": checkpoint.get("epoch"), "best_val_loss": checkpoint.get("best_val_loss")}, destino)
print(f"Época {checkpoint.get('epoch')} -> {destino} ({os.path.getsize(destino) / 1e6:.1f} MB)")
