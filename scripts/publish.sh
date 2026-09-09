#!/usr/bin/env bash
# Publishes Assay at assay.dr-techsolutions.com.
#
# Idempotent and REVERSIBLE: scripts/unpublish.sh undoes it without touching anything else.
set -euo pipefail
DOMAIN=${1:-assay.dr-techsolutions.com}
VPS_IP=187.33.152.210

echo "== 1. checking DNS =="
RESOLVED=$(dig +short "$DOMAIN" A | tail -1 || true)
if [ "$RESOLVED" != "$VPS_IP" ]; then
    echo "  ✗ $DOMAIN resolves to '${RESOLVED:-nothing}', not to $VPS_IP"
    echo
    echo "  Add this record at your DNS provider and run this again:"
    echo "      type  A"
    echo "      name  assay            (or assay.dr-techsolutions.com)"
    echo "      value $VPS_IP"
    echo "      TTL   300"
    echo
    echo "  Caddy will request the TLS certificate on its own once DNS propagates."
    exit 1
fi
echo "  ✓ $DOMAIN -> $VPS_IP"

echo "== 2. state BEFORE (to compare against) =="
systemctl is-active caddy | sed 's/^/  caddy: /'
pm2 jlist 2>/dev/null | python3 -c "
import sys,json;a=json.load(sys.stdin)
print('  production: '+str(sum(1 for x in a if x['pm2_env']['status']=='online'))+'/'+str(len(a))+' online')" || true

echo "== 3. backing up the Caddyfile =="
cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak.$(date +%Y%m%d%H%M%S)"
ls -1t /etc/caddy/Caddyfile.bak.* | head -1 | sed 's/^/  /'

echo "== 4. adding the block =="
if grep -q "^${DOMAIN} {" /etc/caddy/Caddyfile; then
    echo "  already present; replacing it"
    python3 - "$DOMAIN" <<'PY'
import re, sys, pathlib
dom = sys.argv[1]
p = pathlib.Path("/etc/caddy/Caddyfile"); s = p.read_text()
# Remove the domain's whole block, counting braces so we don't cut off too much.
i = s.find(f"\n{dom} {{")
if i >= 0:
    j, depth = i + len(f"\n{dom} {{"), 1
    while j < len(s) and depth:
        if s[j] == "{": depth += 1
        elif s[j] == "}": depth -= 1
        j += 1
    s = s[:i] + s[j:]
    p.write_text(s)
PY
fi
printf '\n' >> /etc/caddy/Caddyfile
cat /root/caddy_assay.conf >> /etc/caddy/Caddyfile

echo "== 5. preparing the log file =="
# Caddy runs as the `caddy` user and cannot CREATE files in /var/log/caddy if the directory is
# owned by root. It has to be created beforehand with the right owner.
#
# And `caddy validate` does NOT catch this: it validates the SYNTAX, not whether the process will
# be able to open the files the configuration mentions. A valid configuration can still be
# unusable, so validating is no substitute for reloading and checking.
OWNER=$(stat -c '%U' /var/log/caddy 2>/dev/null || echo caddy)
install -o "$OWNER" -g "$OWNER" -m 644 /dev/null /var/log/caddy/assay.log 2>/dev/null \
  || { mkdir -p /var/log/caddy && touch /var/log/caddy/assay.log \
       && chown "$OWNER:$OWNER" /var/log/caddy/assay.log; }
ls -l /var/log/caddy/assay.log | sed 's/^/  /'

echo "== 6. validating the configuration BEFORE reloading =="
if ! caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 | tail -3; then
    echo "  ✗ invalid configuration: restoring the backup and aborting"
    cp "$(ls -1t /etc/caddy/Caddyfile.bak.* | head -1)" /etc/caddy/Caddyfile
    exit 1
fi

echo "== 7. public mode on the service =="
mkdir -p /etc/systemd/system/wavelab.service.d
cat > /etc/systemd/system/wavelab.service.d/public.conf <<'EOF'
[Service]
# Blocks the chart, the WebSocket and the history in the application itself. This is the second
# layer: Caddy's allowlist is the first, and a badly written allowlist is a one-line failure.
Environment=WAVELAB_PUBLIC=1
EOF
systemctl daemon-reload && systemctl restart wavelab

echo "== 8. reloading Caddy (without cutting off the existing sites) =="
if ! systemctl reload caddy; then
    echo "  ✗ Caddy REJECTED the configuration and is still running the previous one."
    echo "    Your sites are unaffected. Reason:"
    journalctl -u caddy --no-pager -n 20 | grep -oP '"error":"[^"]+' | tail -1 | sed 's/^/      /'
    echo "    Restoring the previous Caddyfile."
    cp "$(ls -1t /etc/caddy/Caddyfile.bak.* | head -1)" /etc/caddy/Caddyfile
    exit 1
fi

echo "== 9. verification =="
sleep 8
for d in dr-techsolutions.com valorvenal.dr-techsolutions.com "$DOMAIN"; do
    c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "https://$d" || echo 000)
    printf '  %-36s HTTP %s\n' "$d" "$c"
done
echo "  private blocked: HTTP $(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "https://$DOMAIN/api/decide?tf=4h" || echo 000) (must be 404)"
pm2 jlist 2>/dev/null | python3 -c "
import sys,json;a=json.load(sys.stdin)
print('  production: '+str(sum(1 for x in a if x['pm2_env']['status']=='online'))+'/'+str(len(a))+' online')" || true
