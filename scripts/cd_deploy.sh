#!/usr/bin/env bash
# Despliegue continuo. Lo invoca GitHub Actions por ssh con una clave RESTRINGIDA a este script.
#
# Requisitos que se ha propuesto cumplir, porque es el piloto de lo que luego hará DR Solutions:
#
#   1. ATÓMICO respecto al SHA. Se despliega un commit concreto, no "lo último de main", que
#      cambia entre que el pipeline empieza y termina.
#   2. VERIFICA ANTES DE REINICIAR. Si el código nuevo ni siquiera importa, no se toca el servicio.
#   3. COMPRUEBA LA SALUD DESPUÉS. Un servicio que arranca no es un servicio que funciona.
#   4. REVIERTE SOLO. Si la comprobación falla, vuelve al SHA anterior y lo deja funcionando.
#      Un pipeline que despliega pero no revierte solo traslada el trabajo manual a la peor hora.
#   5. NO SE PISA. Un cerrojo impide que dos despliegues simultáneos se mezclen.
set -uo pipefail

DEST=/opt/wavelab
LOCK=/var/lock/wavelab-deploy.lock
SHA_NUEVO=${1:-}
SALUD_URL=http://127.0.0.1:8000/api/estado
INTENTOS=40           # ~2 min: el calentamiento de 9 años tarda unos 120 s con CPUQuota=50%

log() { printf '[cd] %s\n' "$*"; }
fail() { log "FALLO: $*"; exit 1; }

exec 9>"$LOCK"
flock -n 9 || fail "hay otro despliegue en marcha"

[ -n "$SHA_NUEVO" ] || fail "hace falta el SHA a desplegar"
[ -d "$DEST/.git" ] || fail "$DEST no es un repositorio"

cd "$DEST"
SHA_VIEJO=$(git rev-parse HEAD)
log "actual $SHA_VIEJO -> solicitado $SHA_NUEVO"
[ "$SHA_VIEJO" = "$SHA_NUEVO" ] && { log "ya está desplegado; nada que hacer"; exit 0; }

comprobar_salud() {
    for _ in $(seq 1 $INTENTOS); do
        if curl -sf --max-time 5 "$SALUD_URL" | grep -q '"ok":true'; then return 0; fi
        sleep 3
    done
    return 1
}

desplegar() {
    local sha=$1
    git fetch --quiet origin || return 1
    git reset --hard --quiet "$sha" || return 1
    chown -R wavelab:wavelab "$DEST"
    uv sync --frozen 2>&1 | tail -2 || return 1
    chown -R wavelab:wavelab "$DEST"
    # Comprobación EN FRÍO: si el código nuevo ni siquiera importa, no se toca el servicio vivo.
    sudo -u wavelab "$DEST/.venv/bin/python" -c "import wavelab.server.app" || return 1
    systemctl restart wavelab || return 1
    comprobar_salud
}

log "desplegando $SHA_NUEVO"
if desplegar "$SHA_NUEVO"; then
    log "OK: $(git rev-parse --short HEAD) en marcha y respondiendo"
    curl -s --max-time 5 "$SALUD_URL" | head -c 200; echo
    exit 0
fi

log "la comprobación de salud ha fallado; REVIRTIENDO a $SHA_VIEJO"
if desplegar "$SHA_VIEJO"; then
    fail "revertido a $SHA_VIEJO, que sí responde. El commit $SHA_NUEVO NO está desplegado."
fi
fail "CRÍTICO: ni el commit nuevo ni el anterior arrancan. Hace falta intervención manual."
