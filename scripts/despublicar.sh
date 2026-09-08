#!/usr/bin/env bash
# Retira Assay de internet. No borra datos ni toca las demás webs.
set -uo pipefail
DOMINIO=${1:-assay.dr-techsolutions.com}
cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak.$(date +%Y%m%d%H%M%S)"
python3 - "$DOMINIO" <<'PY'
import sys, pathlib
dom = sys.argv[1]
p = pathlib.Path("/etc/caddy/Caddyfile"); s = p.read_text()
i = s.find(f"\n{dom} {{")
if i >= 0:
    j, prof = i + len(f"\n{dom} {{"), 1
    while j < len(s) and prof:
        if s[j] == "{": prof += 1
        elif s[j] == "}": prof -= 1
        j += 1
    p.write_text(s[:i] + s[j:])
    print("  bloque eliminado del Caddyfile")
else:
    print("  no estaba")
PY
rm -f /etc/systemd/system/wavelab.service.d/publico.conf
systemctl daemon-reload && systemctl restart wavelab
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 && systemctl reload caddy
echo "  retirado. Assay vuelve a ser accesible solo por túnel ssh."
