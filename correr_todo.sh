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

correr "1/4 Generar dataset (~40 min)"      data_generator.py         1_generar.log
correr "2/4 Entrenar (varias horas)"        train_Unet_2.py           2_entrenar.log
correr "3/4 Evaluar en test"                evaluar_test.py           3_evaluar.log
correr "4/4 Inferencia datos reales"        inference_unet_2_real.py  4_inferencia.log
paso "TERMINADO. Resultados en checkpoints/V12 y resultados/V12"
