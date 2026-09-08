#!/usr/bin/env bash
# Deshace scripts/provision_vps.sh por completo. Existe ANTES de aprovisionar nada:
# un rollback que no está escrito no es un rollback.
#
# NO borra la imagen de datos por defecto: contiene liquidaciones y noticias que son
# IRRECUPERABLES (no hay fuente histórica gratuita y no se rellenan hacia atrás).
# Para borrarla de verdad hay que pasar --purge explícitamente.
set -uo pipefail
IMG=/var/lib/wavelab.img; MNT=/var/lib/wavelab; PURGE=${1:-}

echo "== parando y deshabilitando unidades =="
systemctl disable --now wavelab.service wavelab-validate.timer 2>/dev/null || true
systemctl stop wavelab-validate.service 2>/dev/null || true
rm -f /etc/systemd/system/wavelab.service \
      /etc/systemd/system/wavelab-validate.service \
      /etc/systemd/system/wavelab-validate.timer \
      /etc/systemd/system/wavelab.slice
systemctl daemon-reload
rm -f /etc/logrotate.d/wavelab
echo "  unidades y logrotate eliminados"

echo "== desmontando y quitando de fstab =="
umount "$MNT" 2>/dev/null || true
sed -i "\|^${IMG} |d" /etc/fstab
echo "  /etc/fstab limpio"

if [ "$PURGE" = "--purge" ]; then
    rm -f "$IMG"; rmdir "$MNT" 2>/dev/null || true
    userdel wavelab 2>/dev/null || true
    rm -rf /opt/wavelab
    echo "  PURGADO: imagen, usuario y /opt/wavelab eliminados"
else
    echo "  imagen CONSERVADA en $IMG (usa --purge para borrarla)"
    echo "  motivo: contiene datos irrecuperables (liquidaciones, noticias punto-en-el-tiempo)"
fi

echo; echo "== estado =="; df -h / | tail -1
pm2 jlist 2>/dev/null | python3 -c "
import sys,json
for a in json.load(sys.stdin): print(f\"  {a['name']:<24} {a['pm2_env']['status']}\")" 2>/dev/null || true
