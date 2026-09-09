#!/usr/bin/env bash
# Provisions wavelab on rec-ai-1 (Ubuntu 24.04, 4 vCPU, 7.8 GB, 59 GB).
#
# PREMISE: this machine runs applications that bill customers (valorvenal, evolution-api,
# admin-site, n8n, and 8 domains served by caddy). No wavelab failure, however catastrophic, may
# degrade them. Everything here is additive and reversible; NOT one existing system configuration
# file is touched.
#
# Idempotent: it can be re-run with no effect.
# Rollback: scripts/deprovision_vps.sh
set -euo pipefail

IMG=/var/lib/wavelab.img
MNT=/var/lib/wavelab
SIZE_GB=8
USER_NAME=wavelab

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------- 0. prior state
say "State BEFORE (so it can be compared at the end)"
df -h / | tail -1
free -h | sed -n 2p
pm2 jlist 2>/dev/null | python3 -c "
import sys,json
for a in json.load(sys.stdin): print(f\"  {a['name']:<24} {a['pm2_env']['status']}\")" 2>/dev/null || true

# --------------------------------------------------------------------------- 1. user
say "System user with no shell"
# wavelab downloads files from the internet and parses them. Running that as root, alongside the
# apps that bill customers, is precisely the scenario we do not want.
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$USER_NAME"
    echo "  created $USER_NAME (no shell, no home, no privileges)"
else
    echo "  $USER_NAME already exists"
fi

# --------------------------------------------------------------------------- 2. PHYSICAL disk limit
say "${SIZE_GB} GB loopback image — the hard limit"
# This is not an alert that can be ignored or a quota that can be raised by accident: wavelab
# writes INSIDE an 8 GB filesystem. A runaway loop in the liquidation recorder runs out of space
# inside its own image and the host never even notices.
WANT_MB=$((SIZE_GB * 1024))
if [ ! -f "$IMG" ]; then
    fallocate -l "${SIZE_GB}G" "$IMG" 2>/dev/null || truncate -s "${SIZE_GB}G" "$IMG"
    mkfs.ext4 -q -m 0 -L wavelab "$IMG"      # -m 0: no root reserve, this has a single use
    echo "  created $IMG (${SIZE_GB} GB)"
else
    echo "  $IMG already exists"
fi

# VERIFY it really is reserved rather than trusting fallocate to have done its job.
# On this machine fallocate left the image SPARSE (8 GB apparent, 69 MB real), which would mean
# the host's disk goes down as wavelab writes — the exact opposite of a reservation. With the
# image materialised, the host's `df /` is frozen for good and wavelab's growth is literally
# invisible from outside.
REAL_MB=$(du -sm "$IMG" | cut -f1)
if [ "$REAL_MB" -lt $((WANT_MB * 9 / 10)) ]; then
    echo "  sparse image (${REAL_MB} MB out of ${WANT_MB}); materialising…"
    # ionice idle: reserving 8 GB must not steal I/O from the apps that bill customers.
    ionice -c 3 nice -n 19 dd if=/dev/zero of="$IMG" bs=1M count="$WANT_MB" \
        conv=notrunc oflag=direct status=none
    REAL_MB=$(du -sm "$IMG" | cut -f1)
fi
if [ "$REAL_MB" -lt $((WANT_MB * 9 / 10)) ]; then
    echo "  ERROR: could not reserve the space (${REAL_MB} MB out of ${WANT_MB})" >&2
    exit 1
fi
echo "  reserved ${REAL_MB} MB for real on disk: the host will not move again"

mkdir -p "$MNT"
if ! grep -q "^$IMG " /etc/fstab; then
    # nofail: if the image is ever missing, the SERVER's boot does not block.
    # It is the difference between "wavelab does not start" and "the server does not start".
    printf '%s %s ext4 loop,defaults,nofail,noatime 0 2\n' "$IMG" "$MNT" >> /etc/fstab
    echo "  added an /etc/fstab entry (with nofail)"
fi
mountpoint -q "$MNT" || mount "$MNT"
mkdir -p "$MNT"/{bars,raw/liquidations,journal,logs,backup}
chown -R "$USER_NAME:$USER_NAME" "$MNT"

# Deployment target, kept apart from the DATA on purpose: a CI/CD pipeline can replace the whole
# of /opt/wavelab without going near the 8 GB image, where the unrecoverable things live.
mkdir -p /opt/wavelab
chown "$USER_NAME:$USER_NAME" /opt/wavelab
df -h "$MNT" | tail -1 | sed 's/^/  /'

# --------------------------------------------------------------------------- 3. systemd containment
say "Slice and units with quotas"
# Figures computed from what was MEASURED on this machine, not from assumptions:
#   RAM 7.8 GB, ~3.9 GB actually in use, memory PSI 0.00, zero OOM events in 30 days.
#   Disk 37 GB free. Load 0.63 across 4 vCPU.
# The slice stops at 1.5 GB in aggregate: plenty of room, and the existing headroom stays intact.
cat > /etc/systemd/system/wavelab.slice <<'EOF'
[Unit]
Description=wavelab — aggregate ceiling for the whole project
Before=slices.target

[Slice]
CPUQuota=150%
MemoryHigh=1200M
MemoryMax=1536M
# MemoryMax bounds RAM but NOT swap. Without this line a runaway process throttles at 1.2 GB of
# RAM and then pushes into swap unchecked: measured on this machine, 1,957 MB of the host
# swapfile's 2,048 MB, with oom_kill=0. That is, nobody kills it, OOMScoreAdjust never gets a
# chance to fire, and the real path to damage — exhausting swap and making the whole machine crawl
# through I/O thrashing — stays wide open.
# With swap bounded, hitting RAM+swap fires the OOM killer INSIDE the cgroup: it kills wavelab and
# nobody else, which is exactly the behaviour we want.
MemorySwapMax=256M
TasksMax=192
IOWeight=50
EOF

cat > /etc/systemd/system/wavelab.service <<'EOF'
[Unit]
Description=wavelab — feeds, engine and web (127.0.0.1:8000)
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
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=10

# Expected consumption: <1% of a core at rest, ~300 MB.
CPUQuota=50%
# NO MemoryHigh, on purpose. Measured on this machine: with MemoryHigh below MemoryMax, a runaway
# process does not die — it throttles and crawls (growing 1 MB/s instead of 500), with oom_kill=0
# indefinitely. For a validation batch that is acceptable: it just takes longer. For the LIVE
# service it is worse than a crash, because Restart=always never gets a chance to fire and you are
# left with a frozen chart and no error at all. MemoryMax alone => on exceeding it the OOM killer
# fires INSIDE the cgroup, systemd restarts the service, and it is on the record.
MemoryMax=768M
MemorySwapMax=128M

# If the kernel ever does fire the OOM killer because of host-wide pressure, let it pick wavelab
# and not valorvenal. MemoryMax should already make systemd kill the offender first;
# this is the belt to go with the braces.
OOMScoreAdjust=500

# Hardening: it can only write inside its own image.
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

cat > /etc/systemd/system/wavelab-collect.service <<'EOF'
[Unit]
Description=wavelab — shadow recorders (liquidations and derivatives)
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
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/wavelab/.venv/bin/python -m wavelab.collect
Restart=always
RestartSec=15

# Very little work: two idle WebSockets and a REST poll every 5 minutes.
CPUQuota=25%
MemoryMax=384M
MemorySwapMax=64M
OOMScoreAdjust=500

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
Description=wavelab — batch validation (walk-forward, CPCV, bootstrap)
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

# One core at most, and at the lowest possible priority on CPU and on I/O:
# that way it can NEVER take resources from valorvenal or evolution-api, not even under load.
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
Description=wavelab — nightly validation

[Timer]
OnCalendar=*-*-* 04:17:00
RandomizedDelaySec=1200
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
echo "  units written and systemd reloaded (NOTHING started yet)"

# --------------------------------------------------------------------------- 4. log rotation
say "Log rotation (wavelab's only; global journald is not touched)"
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
command -v zstd >/dev/null || echo "  WARNING: zstd not installed (apt-get install -y zstd)"
echo "  /etc/logrotate.d/wavelab written — liquidations are compressed, NEVER deleted"

say "State AFTER"
df -h / | tail -1
free -h | sed -n 2p
echo
echo "Done. Nothing started yet. Next step: scripts/contain_test.sh"
