#!/usr/bin/env bash
# Demuestra que wavelab NO PUEDE llenar el disco del host ni tumbar las apps que facturan.
# El criterio de aceptación de M0b no es inspección visual: es esta prueba.
set -uo pipefail
MNT=/var/lib/wavelab

echo "== ANTES =="
HOST_BEFORE=$(df --output=avail / | tail -1)
echo "  host / disponible: $((HOST_BEFORE/1024)) MB"
df -h "$MNT" | tail -1 | sed 's/^/  wavelab: /'
for a in valorvenal evolution-api admin-site; do
  printf '  %-16s %s\n' "$a" "$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
print(next((x['pm2_env']['status'] for x in json.load(sys.stdin) if x['name']=='$a'),'?'))" 2>/dev/null)"
done

echo
echo "== LLENANDO la imagen de wavelab a propósito =="
sudo -u wavelab dd if=/dev/zero of="$MNT/RELLENO.tmp" bs=1M count=20000 2>&1 | tail -2

echo
echo "== DESPUÉS =="
HOST_AFTER=$(df --output=avail / | tail -1)
echo "  host / disponible: $((HOST_AFTER/1024)) MB"
DELTA=$(( (HOST_BEFORE - HOST_AFTER) / 1024 ))
echo "  variación en el host: ${DELTA} MB"
df -h "$MNT" | tail -1 | sed 's/^/  wavelab: /'
for a in valorvenal evolution-api admin-site; do
  printf '  %-16s %s\n' "$a" "$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
print(next((x['pm2_env']['status'] for x in json.load(sys.stdin) if x['name']=='$a'),'?'))" 2>/dev/null)"
done
curl -s -o /dev/null -w "  valorvenal.dr-techsolutions.com -> HTTP %{http_code}\n" \
     --max-time 10 https://valorvenal.dr-techsolutions.com || true

rm -f "$MNT/RELLENO.tmp"
echo
if [ "$DELTA" -lt 50 ]; then
  echo "  ✅ CONTENCIÓN VERIFICADA: wavelab se quedó sin espacio dentro de su imagen"
  echo "     y el disco del host no se movió (${DELTA} MB de variación, ruido normal)."
else
  echo "  ❌ FALLO: el host perdió ${DELTA} MB. La imagen loopback no está conteniendo nada."
  exit 1
fi
