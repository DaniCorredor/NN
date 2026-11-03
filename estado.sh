#!/bin/bash
# Muestra en qué va correr_todo.sh. Uso: bash estado.sh

cd "$(dirname "$0")"

echo "===== PASOS ====="
cat logs/progreso.txt 2>/dev/null || echo "Todavía no se ha iniciado correr_todo.sh"

echo
echo "===== PROCESOS ACTIVOS ====="
pgrep -af "correr_todo.sh|data_generator.py|train_Unet_2.py|evaluar_test.py|inference_unet_2_real.py" || echo "Ninguno"

echo
echo "===== DATASET ====="
echo "Entrenamiento: $(ls datasets/dataset_muografia_V11 2>/dev/null | wc -l) / 10000"
echo "Prueba:        $(ls datasets/dataset_muografia_Test_V12 2>/dev/null | wc -l) / 500"

echo
echo "===== ÚLTIMAS ÉPOCAS (máximo 200, se detiene antes si deja de mejorar) ====="
grep "Fin Epoch" logs/2_entrenar.log 2>/dev/null | tail -3 || true

echo
echo "===== ERRORES ====="
grep -l "Traceback" logs/*.log 2>/dev/null || echo "Ninguno"
