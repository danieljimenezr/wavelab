#!/usr/bin/env bash
# Takes Assay off the internet. Deletes no data and touches none of the other sites.
set -uo pipefail
DOMAIN=${1:-assay.dr-techsolutions.com}
cp /etc/caddy/Caddyfile "/etc/caddy/Caddyfile.bak.$(date +%Y%m%d%H%M%S)"
python3 - "$DOMAIN" <<'PY'
import sys, pathlib
dom = sys.argv[1]
p = pathlib.Path("/etc/caddy/Caddyfile"); s = p.read_text()
i = s.find(f"\n{dom} {{")
if i >= 0:
    j, depth = i + len(f"\n{dom} {{"), 1
    while j < len(s) and depth:
        if s[j] == "{": depth += 1
        elif s[j] == "}": depth -= 1
        j += 1
    p.write_text(s[:i] + s[j:])
    print("  block removed from the Caddyfile")
else:
    print("  it was not there")
PY
# publico.conf is the pre-rename name of the drop-in. Nodes provisioned before the rename still
# have it, and leaving it behind would keep WAVELAB_PUBLIC=1 set after unpublishing — the service
# would stay in public mode with nothing to show for it in the Caddyfile.
rm -f /etc/systemd/system/wavelab.service.d/public.conf \
      /etc/systemd/system/wavelab.service.d/publico.conf
systemctl daemon-reload && systemctl restart wavelab
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1 && systemctl reload caddy
echo "  taken down. Assay is reachable over the ssh tunnel only again."
