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
# ionice idle + nice 19 + O_DIRECT: escribe 8 GB sin robarle E/S a las apps que facturan
# ni desalojar su caché de página.
# ionice idle + nice 19 + O_DIRECT: escribe sin robarle E/S a las apps que facturan
# ni desalojar su caché de página.
DD_OUT=$(ionice -c 3 nice -n 19 runuser -u wavelab -- \
    dd if=/dev/zero of="$MNT/RELLENO.tmp" bs=1M count=12000 oflag=direct 2>&1)
echo "$DD_OUT" | tail -3 | sed 's/^/  /'
WROTE_MB=$(echo "$DD_OUT" | grep -oE '^[0-9]+\+[0-9]+ records out' | grep -oE '^[0-9]+' | head -1)
WROTE_MB=${WROTE_MB:-0}

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

INNER_PCT=$(df --output=pcent "$MNT" | tail -1 | tr -dc '0-9')
rm -f "$MNT/RELLENO.tmp"

echo
FAIL=0
# (1) El escritor tiene que TOPAR. Se pidieron 12 GB en una imagen de 8: si dd los escribió
#     todos, la imagen no está conteniendo nada.
if [ "$WROTE_MB" -lt 11000 ]; then
  echo "  ✅ el escritor topó a los ${WROTE_MB} MB de los 12000 pedidos (ENOSPC)"
else
  echo "  ❌ dd escribió ${WROTE_MB} MB: no topó con ningún límite"; FAIL=1
fi
# (2) El sistema de ficheros interno llegó al 100%: el tope es suyo, no del host.
if [ "${INNER_PCT:-0}" -ge 99 ]; then
  echo "  ✅ la imagen de wavelab llegó al ${INNER_PCT}%: el límite lo puso ELLA"
else
  echo "  ❌ la imagen se quedó al ${INNER_PCT}%: topó en otro sitio"; FAIL=1
fi
# (3) El host no se mueve. Con la imagen reservada, escribir dentro no le quita ni un byte.
if [ "$DELTA" -lt 100 ]; then
  echo "  ✅ el disco del host no se movió (${DELTA} MB, ruido de medición)"
else
  echo "  ❌ el host perdió ${DELTA} MB: la imagen no estaba reservada, sino dispersa"; FAIL=1
fi

# ---------------------------------------------------------------- 2) memoria
echo
echo "== MEMORIA: proceso que reserva RAM sin parar, con el presupuesto del servicio =="
U=wl-mem-$$-$RANDOM
systemd-run --quiet --slice=wavelab.slice --uid=wavelab --unit=$U \
    -p MemoryMax=768M -p MemorySwapMax=128M -p OOMScoreAdjust=500 -p RuntimeMaxSec=45 \
    /usr/bin/python3 -c "
b=[]
while True: b.append(bytearray(20*1024*1024))
" 2>/dev/null
for i in $(seq 1 25); do
  sleep 1
  [ "$(systemctl is-active $U 2>/dev/null)" != "active" ] && break
done
RESULT=$(systemctl show $U -p Result --value 2>/dev/null)
SWAP_USED=$(free -m | sed -n 3p | awk '{print $3}')
systemctl reset-failed $U 2>/dev/null || true
echo "  resultado de la unidad: $RESULT (tras ${i}s)"
echo "  swap del host: ${SWAP_USED} MB usados"

# (4) Tiene que MORIR, no colgarse. Un servicio estrangulado que se arrastra para siempre
#     nunca dispara Restart=always: te quedas con un gráfico congelado y ningún error.
if [ "$RESULT" = "oom-kill" ]; then
  echo "  ✅ el cgroup lo mató (oom-kill) en ${i}s: morir y reiniciar, no colgarse"
else
  echo "  ❌ resultado '$RESULT' en vez de oom-kill: se estranguló sin morir"; FAIL=1
fi
# (5) El swap del host es la vía de daño real: agotarlo hace que la máquina entera se arrastre.
if [ "${SWAP_USED:-9999}" -lt 600 ]; then
  echo "  ✅ el swap del host apenas se tocó (${SWAP_USED} MB)"
else
  echo "  ❌ el swap del host está en ${SWAP_USED} MB: falta acotar MemorySwapMax"; FAIL=1
fi

# ---------------------------------------------------------------- 3) producción
echo
echo "== PRODUCCIÓN, tras ambas pruebas =="
for a in valorvenal evolution-api admin-site evolution-outbox-worker sara-restaurant-worker baja-webhook; do
  st=$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json;print(next((x['pm2_env']['status'] for x in json.load(sys.stdin) if x['name']=='$a'),'?'))" 2>/dev/null)
  printf '  %-26s %s\n' "$a" "$st"
  [ "$st" != "online" ] && FAIL=1
done
for d in valorvenal.dr-techsolutions.com dr-techsolutions.com admin.dr-techsolutions.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$d" || echo 000)
  # Lo que se comprueba es "¿sigue sirviendo?", no "¿devuelve exactamente 200?".
  # Cualquier 2xx o 3xx vale: admin.dr-techsolutions.com responde 307 (redirect de auth de Next.js)
  # y eso es funcionamiento normal. Fallo real = 4xx, 5xx o 000 (sin conexión).
  case "$code" in
    2??|3??) printf '  %-34s HTTP %s  ok\n' "$d" "$code" ;;
    *)       printf '  %-34s HTTP %s  <-- FALLO\n' "$d" "$code"; FAIL=1 ;;
  esac
done

echo
[ "$FAIL" -eq 0 ] && echo "  CONTENCIÓN VERIFICADA" || { echo "  CONTENCIÓN NO VERIFICADA"; exit 1; }
