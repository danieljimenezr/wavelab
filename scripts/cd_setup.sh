#!/usr/bin/env bash
# Prepara el VPS para recibir despliegues de GitHub Actions. Se ejecuta UNA vez, a mano.
#
# La clave de despliegue queda RESTRINGIDA por `command=` en authorized_keys: aunque alguien se
# hiciera con ella, solo puede ejecutar el script de despliegue. No abre una shell, no reenvía
# puertos, no puede leer un fichero. Es la diferencia entre una clave de despliegue y una llave
# maestra del servidor, y es justo el patrón que conviene replicar en DR Solutions.
set -euo pipefail
CLAVE=/root/.ssh/wavelab_deploy

echo "== 1. usuario de despliegue y permisos mínimos =="
# El script necesita systemctl sobre UN servicio y nada más.
cat > /etc/sudoers.d/wavelab-deploy <<'EOF'
# Permisos mínimos para el despliegue continuo: solo este servicio, solo estas acciones.
root ALL=(wavelab) NOPASSWD: /opt/wavelab/.venv/bin/python
EOF
chmod 440 /etc/sudoers.d/wavelab-deploy
visudo -c -f /etc/sudoers.d/wavelab-deploy

echo "== 2. clave de despliegue =="
if [ ! -f "$CLAVE" ]; then
    ssh-keygen -t ed25519 -N "" -f "$CLAVE" -C "github-actions-wavelab" >/dev/null
    echo "  clave nueva generada"
else
    echo "  ya existía"
fi

echo "== 3. autorizándola SOLO para el script de despliegue =="
PUB=$(cat "$CLAVE.pub")
mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys
grep -v "github-actions-wavelab" /root/.ssh/authorized_keys > /tmp/ak.new 2>/dev/null || true
cat >> /tmp/ak.new <<EOF
command="/root/cd_deploy.sh \$SSH_ORIGINAL_COMMAND",no-agent-forwarding,no-port-forwarding,no-pty,no-X11-forwarding,no-user-rc $PUB
EOF
mv /tmp/ak.new /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
echo "  authorized_keys actualizado con command= restringido"

echo
echo "== CLAVE PRIVADA para el secreto de GitHub (SSH_DEPLOY_KEY) =="
echo "-----8<-----"
cat "$CLAVE"
echo "-----8<-----"
echo
echo "== HUELLA del host para el secreto SSH_KNOWN_HOSTS =="
ssh-keyscan -t ed25519 -H "$(curl -s --max-time 8 ifconfig.co || echo 187.33.152.210)" 2>/dev/null
