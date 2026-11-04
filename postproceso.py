import subprocess
import sys

# Posprocesamiento de la red entrenada (corre en el PC: el bootstrap necesita los ángulos por muón
# de ../Datos). Usa modelos/V16_fp16.pt y resultados/V16/test_evaluation/iron_threshold.json.
# Salidas en resultados/V16/postprocessing/:
#   reliability_map.png      zona confiable del detector (confiabilidad.py)
#   bootstrap/               frecuencia de detección e incertidumbre por remuestreo de muones
#   rigid_fit/               Nivel A: barras rígidas ajustadas a los datos, con diámetro y significancia
#   rigid_fit_validation/    el mismo ajuste en escenas simuladas con la geometría conocida
# Uso:  python postproceso.py

PASOS = ["confiabilidad.py", "bootstrap_muones.py", "ajuste_barras.py", "validar_ajuste_barras.py"]

for paso in PASOS:
    print(f"===== {paso}", flush=True)
    if subprocess.run([sys.executable, paso]).returncode != 0:
        sys.exit(f"Falló {paso}")
