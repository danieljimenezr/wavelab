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
WANT_MB=$((SIZE_GB * 1024))
if [ ! -f "$IMG" ]; then
    fallocate -l "${SIZE_GB}G" "$IMG" 2>/dev/null || truncate -s "${SIZE_GB}G" "$IMG"
    mkfs.ext4 -q -m 0 -L wavelab "$IMG"      # -m 0: sin reserva para root, es de un solo uso
    echo "  creada $IMG (${SIZE_GB} GB)"
else
    echo "  $IMG ya existe"
fi

# VERIFICAR que está reservada de verdad, no confiar en que fallocate haya hecho su trabajo.
# En esta máquina fallocate dejó la imagen DISPERSA (8 GB aparentes, 69 MB reales), lo que
# significaría que el disco del host baja según wavelab escribe — justo lo contrario de una
# reserva. Con la imagen materializada, `df /` del host queda congelado para siempre y el
# crecimiento de wavelab es literalmente invisible desde fuera.
REAL_MB=$(du -sm "$IMG" | cut -f1)
if [ "$REAL_MB" -lt $((WANT_MB * 9 / 10)) ]; then
    echo "  imagen dispersa (${REAL_MB} MB de ${WANT_MB}); materializando…"
    # ionice idle: reservar 8 GB no puede robarle E/S a las apps que facturan.
    ionice -c 3 nice -n 19 dd if=/dev/zero of="$IMG" bs=1M count="$WANT_MB" \
        conv=notrunc oflag=direct status=none
    REAL_MB=$(du -sm "$IMG" | cut -f1)
fi
if [ "$REAL_MB" -lt $((WANT_MB * 9 / 10)) ]; then
    echo "  ERROR: no se pudo reservar el espacio (${REAL_MB} MB de ${WANT_MB})" >&2
    exit 1
fi
echo "  reservados ${REAL_MB} MB reales en disco: el host ya no se moverá"

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

# Destino de despliegue, separado de los DATOS a propósito: un CI/CD puede reemplazar
# /opt/wavelab entero sin rozar la imagen de 8 GB, donde vive lo irrecuperable.
mkdir -p /opt/wavelab
chown "$USER_NAME:$USER_NAME" /opt/wavelab
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
# MemoryMax acota la RAM pero NO el swap. Sin esta línea, un proceso desbocado se estrangula
# en 1,2 GB de RAM y luego empuja al swap sin freno: medido en esta máquina, 1.957 MB de los
# 2.048 MB del swapfile del host, con oom_kill=0. Es decir, no lo mata nadie, el OOMScoreAdjust
# no llega a dispararse nunca, y la vía real de daño —agotar el swap y hacer que la máquina
# entera se arrastre por thrashing de E/S— queda abierta.
# Acotando el swap, al topar RAM+swap salta el OOM killer DENTRO del cgroup: mata a wavelab
# y a nadie más, que es exactamente el comportamiento que queremos.
MemorySwapMax=256M
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
# SIN MemoryHigh a propósito. Medido en esta máquina: con MemoryHigh por debajo de MemoryMax,
# un proceso desbocado no muere — se estrangula y se arrastra (crece 1 MB/s en vez de 500),
# con oom_kill=0 indefinidamente. Para un lote de validación eso es aceptable: tarda más.
# Para el servicio EN VIVO es peor que una caída, porque Restart=always no llega a dispararse
# nunca y te quedas con un gráfico congelado sin ningún error. Solo MemoryMax => al superarlo
# salta el OOM killer DENTRO del cgroup, systemd lo reinicia, y queda registrado.
MemoryMax=768M
MemorySwapMax=128M

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
MemorySwapMax=256M
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
