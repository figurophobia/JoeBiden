#!/bin/bash

# Verifica si se pasó un argumento
if [ -z "$1" ]; then
    echo "Uso: $0 <programa>"
    exit 1
fi

# Ejecuta el programa 15 veces con diferentes parámetros
for params in "250 0" "2500 0" "5000 0"; do
    echo "Ejecutando con parámetros: $params"
    for i in {1..15}; do
        echo "Ejecución $i:"
        $1 $params
    done
done