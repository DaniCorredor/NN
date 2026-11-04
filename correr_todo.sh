#!/bin/bash
# Corre todo el flujo en orden: generar -> entrenar -> evaluar -> inferencia.
# Cada paso empieza solo cuando el anterior terminó bien.
# Uso:    nohup bash correr_todo.sh > /dev/null 2>&1 &
# Estado: bash estado.sh

cd "$(dirname "$0")"
mkdir -p logs
: > logs/progreso.txt

paso() { echo "$(date '+%Y-%m-%d %H:%M')  $1" >> logs/progreso.txt; }

correr() {
    # $1 = descripción, $2 = script, $3 = archivo de log
    paso "INICIO  $1"
    if python3 "$2" > "logs/$3" 2>&1; then
        paso "OK      $1"
    else
        paso "ERROR   $1  (revisa logs/$3)"
        exit 1
    fi
}

correr "1/5 Generar dataset de prueba (~5 min; el de entrenamiento V11 se reutiliza)" data_generator.py         1_generar.log
correr "2/5 Entrenar (hasta ~5 h)"          train_Unet_2.py           2_entrenar.log
correr "3/5 Evaluar en test"                evaluar_test.py           3_evaluar.log
correr "4/5 Inferencia datos reales"        inference_unet_2_real.py  4_inferencia.log
# Pesos en float16 (~74 MB) para poder subirlos a GitHub y analizarlos en el PC
paso "INICIO  5/5 Exportar pesos"
if python3 exportar_pesos.py V16 > logs/5_exportar.log 2>&1; then paso "OK      5/5 Exportar pesos"; else paso "ERROR   5/5 Exportar pesos (revisa logs/5_exportar.log)"; exit 1; fi
paso "TERMINADO. Resultados en checkpoints/V16, resultados/V16 y modelos/V16_fp16.pt"
