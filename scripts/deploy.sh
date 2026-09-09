#!/usr/bin/env bash
# Deployment into /opt/wavelab. Kept separate from /var/lib/wavelab on purpose: this can be
# replaced wholesale (by a CI/CD pipeline, for instance) without going anywhere near the
# unrecoverable data.
set -euo pipefail
REPO=${1:-https://github.com/danieljimenezr/wavelab.git}
DEST=/opt/wavelab

# The interpreter can NOT live under /root: the venv links to it, and the wavelab user, which has
# no permission to read /root (mode 700), would get "Permission denied" when the service starts.
export UV_PYTHON_INSTALL_DIR=/opt/python

command -v uv >/dev/null 2>&1 || {
    echo "== installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
}
uv --version

# /opt/wavelab belongs to the wavelab user but git runs as root: without this, every deployment
# (and every CI/CD run) fails with "dubious ownership".
git config --global --add safe.directory /opt/wavelab 2>/dev/null || true

echo "== code =="
if [ -d "$DEST/.git" ]; then
    git -C "$DEST" fetch --quiet origin && git -C "$DEST" reset --hard --quiet origin/main
    echo "  updated to $(git -C "$DEST" rev-parse --short HEAD)"
else
    rm -rf "$DEST"; git clone --quiet "$REPO" "$DEST"
    echo "  cloned at $(git -C "$DEST" rev-parse --short HEAD)"
fi

echo "== environment =="
cd "$DEST"
uv python install 3.13 >/dev/null 2>&1 || true
chmod -R a+rX /opt/python 2>/dev/null || true
rm -rf "$DEST/.venv"
uv sync --frozen 2>&1 | tail -2
chown -R wavelab:wavelab "$DEST"
chmod -R a+rX /opt/python

echo "== check =="
sudo -u wavelab "$DEST/.venv/bin/python" -c "
import wavelab, wavelab.collect
from wavelab.core.causality import CausalityError
print('  wavelab imports OK')"
