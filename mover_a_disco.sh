#!/bin/bash
# Mueve lo pesado del proyecto al disco grande y deja enlaces simbólicos en su lugar, así los
# scripts siguen usando las mismas rutas:
#   datasets/            -> carpeta completa (no se sube a git)
#   checkpoints/*/*.pth  -> cada archivo de pesos completo (no se sube a git); las curvas y el
#                           historial se quedan en home porque sí se suben a GitHub
# Uso (en el servidor, desde la carpeta NN2):
#   bash mover_a_disco.sh          -> solo muestra qué movería y cuánto ocupa
#   bash mover_a_disco.sh mover    -> lo mueve
set -e
cd "$(dirname "$0")"
DESTINO=/disco1/d.corredorl_files/NN

echo "===== Espacio en home ====="
df -h "$HOME" | tail -1
echo
echo "===== Lo que se movería ====="
[ -d datasets ] && [ ! -L datasets ] && du -sh datasets
find checkpoints -name "*.pth" -type f -exec du -ch {} + 2>/dev/null | tail -1 | sed 's/total/checkpoints (*.pth)/'
echo
echo "===== Lo más pesado de toda tu carpeta personal ====="
du -sh "$HOME"/* "$HOME"/.cache 2>/dev/null | sort -rh | head -8

if [ "$1" != "mover" ]; then
    echo
    echo "Nada se movió. Para moverlo:  bash mover_a_disco.sh mover"
    exit 0
fi

if pgrep -f "train_Unet_2.py|data_generator.py|evaluar_test.py|inference_unet_2_real.py" > /dev/null; then
    echo "Hay un proceso de la red corriendo: espera a que termine." && exit 1
fi
mkdir -p "$DESTINO"

mover() {
    # $1 = archivo o carpeta. Copia, verifica, borra el original y deja un enlace
    local origen="$1" copia="$DESTINO/$1"
    mkdir -p "$(dirname "$copia")"
    if [ -d "$origen" ]; then rsync -a "$origen/" "$copia/"; else rsync -a "$origen" "$copia"; fi
    if diff -rq "$origen" "$copia" > /dev/null; then
        rm -rf "$origen"
        ln -s "$copia" "$origen"
        echo "  OK  $origen -> $copia"
    else
        echo "  La copia de $origen no coincide: el original NO se borró" && exit 1
    fi
}

[ -d datasets ] && [ ! -L datasets ] && mover datasets
find checkpoints -name "*.pth" -type f | while read -r f; do mover "$f"; done

echo
echo "===== Espacio en home después ====="
df -h "$HOME" | tail -1
