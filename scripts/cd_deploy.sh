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
# El calentamiento de nueve años tarda ~120 s con CPUQuota=50%. El plazo se pone MUY por encima
# a propósito: un plazo ajustado convierte cada despliegue en una moneda al aire, y ya pasó —
# se agotó por dos segundos, revirtió un despliegue correcto y declaró un incidente inexistente.
INTENTOS=120          # 6 minutos

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
    # Se espera a `listo`, no a `ok`. `ok` solo dice que el proceso vive; `listo` dice que ya
    # puede trabajar. Confundirlos es dar por bueno un servicio que aún no sirve para nada.
    local visto_ok=0
    for i in $(seq 1 $INTENTOS); do
        local r
        r=$(curl -sf --max-time 5 "$SALUD_URL" 2>/dev/null || true)
        if echo "$r" | grep -q '"listo":true'; then
            log "sano tras $((i*3))s"
            return 0
        fi
        if echo "$r" | grep -q '"ok":true'; then
            [ "$visto_ok" = 0 ] && log "responde y está calentando…"
            visto_ok=1
        fi
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
