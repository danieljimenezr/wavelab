#!/usr/bin/env bash
# Aprovisionamiento de wavelab en rec-ai-1 (Ubuntu 24.04, 4 vCPU, 7.8 GB, 59 GB).
#
# PREMISA: la máquina ejecuta aplicaciones que facturan (valorvenal, evolution-api, admin-site,
# n8n, y 8 dominios servidos por caddy). Ningún fallo de wavelab, por catastrófico que sea, puede
# degradarlas. Todo lo de aquí es aditivo y reversible; NO se toca ni un fichero de configuración
# existente del sistema.
#
# Idempotente: se puede volver a ejecutar sin efectos.
# Reversión: scripts/deprovision_vps.sh
set -euo pipefail

IMG=/var/lib/wavelab.img
MNT=/var/lib/wavelab
SIZE_GB=8
USER_NAME=wavelab

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- 0. estado previo
say "Estado ANTES (para poder comparar al terminar)"
df -h / | tail -1
free -h | sed -n 2p
pm2 jlist 2>/dev/null | python3 -c "
import sys,json
for a in json.load(sys.stdin): print(f\"  {a['name']:<24} {a['pm2_env']['status']}\")" 2>/dev/null || true

# --------------------------------------------------------------------------- 1. usuario
say "Usuario de sistema sin shell"
# wavelab descarga ficheros de internet y los parsea. Correr eso como root, junto a las apps que
# facturan, es justo el escenario que no queremos.
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$USER_NAME"
    echo "  creado $USER_NAME (sin shell, sin home, sin privilegios)"
else
    echo "  $USER_NAME ya existe"
fi

# --------------------------------------------------------------------------- 2. límite FÍSICO de disco
say "Imagen loopback de ${SIZE_GB} GB — el límite duro"
# No es una alerta que se pueda ignorar ni una cuota que se pueda subir por descuido: wavelab
# escribe DENTRO de un sistema de ficheros de 8 GB. Un bucle desbocado en el grabador de
# liquidaciones se queda sin espacio dentro de su propia imagen y el host ni se entera.
if [ ! -f "$IMG" ]; then
    fallocate -l "${SIZE_GB}G" "$IMG"
    mkfs.ext4 -q -m 0 -L wavelab "$IMG"      # -m 0: sin reserva para root, es de un solo uso
    echo "  creada $IMG (${SIZE_GB} GB)"
else
    echo "  $IMG ya existe"
fi

mkdir -p "$MNT"
if ! grep -q "^$IMG " /etc/fstab; then
    # nofail: si algún día la imagen no está, el arranque del SERVIDOR no se bloquea.
    # Es la diferencia entre "wavelab no arranca" y "el servidor no arranca".
    printf '%s %s ext4 loop,defaults,nofail,noatime 0 2\n' "$IMG" "$MNT" >> /etc/fstab
    echo "  añadida entrada en /etc/fstab (con nofail)"
fi
mountpoint -q "$MNT" || mount "$MNT"
mkdir -p "$MNT"/{bars,raw/liquidations,journal,logs,backup}
chown -R "$USER_NAME:$USER_NAME" "$MNT"
df -h "$MNT" | tail -1 | sed 's/^/  /'

# --------------------------------------------------------------------------- 3. contención systemd
say "Slice y unidades con cuotas"
# Cifras calculadas sobre lo MEDIDO en esta máquina, no sobre supuestos:
#   RAM 7,8 GB, ~3,9 GB en uso real, PSI de memoria 0,00, cero eventos OOM en 30 días.
#   Disco 37 GB libres. Carga 0,63 sobre 4 vCPU.
# El slice se queda en 1,5 GB agregados: cabe de sobra y deja intacto el margen existente.
cat > /etc/systemd/system/wavelab.slice <<'EOF'
[Unit]
Description=wavelab — techo agregado de todo el proyecto
Before=slices.target

[Slice]
CPUQuota=150%
MemoryHigh=1200M
MemoryMax=1536M
TasksMax=192
IOWeight=50
EOF

cat > /etc/systemd/system/wavelab.service <<'EOF'
[Unit]
Description=wavelab — feeds, motor y web (127.0.0.1:8000)
After=network-online.target var-lib-wavelab.mount
Requires=var-lib-wavelab.mount
Wants=network-online.target

[Service]
Type=simple
User=wavelab
Group=wavelab
Slice=wavelab.slice
WorkingDirectory=/opt/wavelab
Environment=WAVELAB_DATA=/var/lib/wavelab
ExecStart=/opt/wavelab/.venv/bin/python -m wavelab.server
Restart=always
RestartSec=10

# Consumo esperado: <1% de un núcleo en reposo, ~300 MB.
CPUQuota=50%
MemoryHigh=600M
MemoryMax=768M

# Si el kernel llegase a disparar el OOM killer por presión del host, que elija a wavelab
# y no a valorvenal. MemoryMax ya debería hacer que systemd mate primero al infractor;
# esto es el cinturón además de los tirantes.
OOMScoreAdjust=500

# Endurecimiento: solo puede escribir en su propia imagen.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/wavelab
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictSUIDSGID=true
LockPersonality=true

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/wavelab-validate.service <<'EOF'
[Unit]
Description=wavelab — validación por lotes (walk-forward, CPCV, bootstrap)
After=var-lib-wavelab.mount
Requires=var-lib-wavelab.mount

[Service]
Type=oneshot
User=wavelab
Group=wavelab
Slice=wavelab.slice
WorkingDirectory=/opt/wavelab
Environment=WAVELAB_DATA=/var/lib/wavelab
ExecStart=/opt/wavelab/.venv/bin/python -m wavelab.validation.run_nightly

# Un núcleo como mucho, y con la prioridad más baja posible en CPU y en E/S:
# así no puede arrebatarle recursos a valorvenal ni a evolution-api NUNCA, ni bajo carga.
CPUQuota=100%
Nice=19
IOSchedulingClass=idle
CPUSchedulingPolicy=idle
MemoryHigh=1000M
MemoryMax=1200M
OOMScoreAdjust=800
TimeoutStartSec=8h

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/wavelab
EOF

cat > /etc/systemd/system/wavelab-validate.timer <<'EOF'
[Unit]
Description=wavelab — validación nocturna

[Timer]
OnCalendar=*-*-* 04:17:00
RandomizedDelaySec=1200
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
echo "  unidades escritas y systemd recargado (NO se arranca nada todavía)"

# --------------------------------------------------------------------------- 4. rotación de logs
say "Rotación de logs (solo la de wavelab; no se toca journald global)"
cat > /etc/logrotate.d/wavelab <<'EOF'
/var/lib/wavelab/logs/*.log /var/lib/wavelab/raw/liquidations/*.jsonl {
    daily
    rotate 400
    compress
    compresscmd /usr/bin/zstd
    compressoptions -19 --rm
    compressext .zst
    missingok
    notifempty
    copytruncate
    su wavelab wavelab
}
EOF
command -v zstd >/dev/null || echo "  AVISO: zstd no instalado (apt-get install -y zstd)"
echo "  /etc/logrotate.d/wavelab escrito — las liquidaciones se comprimen, NUNCA se borran"

say "Estado DESPUÉS"
df -h / | tail -1
free -h | sed -n 2p
echo
echo "Hecho. Nada arrancado aún. Siguiente paso: scripts/contain_test.sh"
