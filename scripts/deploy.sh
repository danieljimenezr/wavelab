#!/usr/bin/env bash
# Despliegue en /opt/wavelab. Separado de /var/lib/wavelab a propósito: esto se puede
# reemplazar entero (por un CI/CD, por ejemplo) sin rozar los datos irrecuperables.
set -euo pipefail
REPO=${1:-https://github.com/danieljimenezr/wavelab.git}
DEST=/opt/wavelab

# El intérprete NO puede vivir bajo /root: el venv lo enlaza y el usuario wavelab, que no tiene
# permiso para leer /root (modo 700), obtendría "Permission denied" al arrancar el servicio.
export UV_PYTHON_INSTALL_DIR=/opt/python

command -v uv >/dev/null 2>&1 || {
    echo "== instalando uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
}
uv --version

# /opt/wavelab pertenece al usuario wavelab pero git corre como root: sin esto, cada despliegue
# (y cada pipeline de CI/CD) falla con "dubious ownership".
git config --global --add safe.directory /opt/wavelab 2>/dev/null || true

echo "== código =="
if [ -d "$DEST/.git" ]; then
    git -C "$DEST" fetch --quiet origin && git -C "$DEST" reset --hard --quiet origin/main
    echo "  actualizado a $(git -C "$DEST" rev-parse --short HEAD)"
else
    rm -rf "$DEST"; git clone --quiet "$REPO" "$DEST"
    echo "  clonado en $(git -C "$DEST" rev-parse --short HEAD)"
fi

echo "== entorno =="
cd "$DEST"
uv python install 3.13 >/dev/null 2>&1 || true
chmod -R a+rX /opt/python 2>/dev/null || true
rm -rf "$DEST/.venv"
uv sync --frozen 2>&1 | tail -2
chown -R wavelab:wavelab "$DEST"
chmod -R a+rX /opt/python

echo "== comprobación =="
sudo -u wavelab "$DEST/.venv/bin/python" -c "
import wavelab, wavelab.collect
from wavelab.core.causality import CausalityError
print('  wavelab importa OK')"
