#!/usr/bin/env bash
# Publica Assay en assay.dr-techsolutions.com.
#
# Idempotente y REVERSIBLE: scripts/despublicar.sh lo deshace sin tocar nada más.
set -euo pipefail
DOMINIO=${1:-assay.dr-techsolutions.com}
IP_VPS=187.33.152.210

echo "== 1. comprobando el DNS =="
RES=$(dig +short "$DOMINIO" A | tail -1 || true)
if [ "$RES" != "$IP_VPS" ]; then
    echo "  ✗ $DOMINIO resuelve a '${RES:-nada}', no a $IP_VPS"
    echo
    echo "  Añade este registro en tu proveedor de DNS y vuelve a ejecutar:"
    echo "      tipo  A"
    echo "      nombre assay            (o assay.dr-techsolutions.com)"
    echo "      valor $IP_VPS"
    echo "      TTL   300"
    echo
    echo "  Caddy pedirá el certificado TLS solo en cuanto el DNS propague."
    exit 1
fi
echo "  ✓ $DOMINIO -> $IP_VPS"

echo "== 2. estado ANTES (para comparar) =="
systemctl is-active caddy | sed 's/^/  caddy: /'
pm2 jlist 2>/dev/null | python3 -c "
import sys,json;a=json.load(sys.stdin)
print('  producción: '+str(sum(1 for x in a if x['pm2_env']['status']=='online'))+'/'+str(len(a))+' online')" || true

echo "== 3. copia de seguridad del Caddyfile =="
cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak.$(date +%Y%m%d%H%M%S)"
ls -1t /etc/caddy/Caddyfile.bak.* | head -1 | sed 's/^/  /'

echo "== 4. añadiendo el bloque =="
if grep -q "^${DOMINIO} {" /etc/caddy/Caddyfile; then
    echo "  ya existe; se reemplaza"
    python3 - "$DOMINIO" <<'PY'
import re, sys, pathlib
dom = sys.argv[1]
p = pathlib.Path("/etc/caddy/Caddyfile"); s = p.read_text()
# Elimina el bloque completo del dominio, contando llaves para no cortar de más.
i = s.find(f"\n{dom} {{")
if i >= 0:
    j, prof = i + len(f"\n{dom} {{"), 1
    while j < len(s) and prof:
        if s[j] == "{": prof += 1
        elif s[j] == "}": prof -= 1
        j += 1
    s = s[:i] + s[j:]
    p.write_text(s)
PY
fi
printf '\n' >> /etc/caddy/Caddyfile
cat /root/caddy_assay.conf >> /etc/caddy/Caddyfile

echo "== 5. validando la configuración ANTES de recargar =="
if ! caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | tail -3; then
    echo "  ✗ configuración inválida: restaurando la copia y abortando"
    cp "$(ls -1t /etc/caddy/Caddyfile.bak.* | head -1)" /etc/caddy/Caddyfile
    exit 1
fi

echo "== 6. modo público en el servicio =="
mkdir -p /etc/systemd/system/wavelab.service.d
cat > /etc/systemd/system/wavelab.service.d/publico.conf <<'EOF'
[Service]
# Bloquea en la propia aplicación el gráfico, el WebSocket y el histórico. Es la segunda
# capa: la lista blanca de Caddy es la primera, y una lista blanca mal escrita es un fallo
# de una sola línea.
Environment=WAVELAB_PUBLIC=1
EOF
systemctl daemon-reload && systemctl restart wavelab

echo "== 7. recargando Caddy (sin cortar las webs existentes) =="
systemctl reload caddy

echo "== 8. comprobación =="
sleep 8
for d in dr-techsolutions.com valorvenal.dr-techsolutions.com "$DOMINIO"; do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$d" || echo 000)
    printf '  %-36s HTTP %s\n' "$d" "$c"
done
echo "  privado bloqueado: HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "https://$DOMINIO/api/decide?tf=4h" || echo 000) (debe ser 404)"
pm2 jlist 2>/dev/null | python3 -c "
import sys,json;a=json.load(sys.stdin)
print('  producción: '+str(sum(1 for x in a if x['pm2_env']['status']=='online'))+'/'+str(len(a))+' online')" || true
