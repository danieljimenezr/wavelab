#!/usr/bin/env bash
# Undoes scripts/provision_vps.sh completely. It exists BEFORE anything is provisioned:
# a rollback that has not been written is not a rollback.
#
# It does NOT delete the data image by default: it holds liquidations and news that are
# IRRECOVERABLE (there is no free historical source and they cannot be backfilled).
# To actually delete it you have to pass --purge explicitly.
set -uo pipefail
IMG=/var/lib/wavelab.img; MNT=/var/lib/wavelab; PURGE=${1:-}

echo "== stopping and disabling units =="
systemctl disable --now wavelab.service wavelab-validate.timer 2>/dev/null || true
systemctl stop wavelab-validate.service 2>/dev/null || true
rm -f /etc/systemd/system/wavelab.service \
      /etc/systemd/system/wavelab-validate.service \
      /etc/systemd/system/wavelab-validate.timer \
      /etc/systemd/system/wavelab.slice
systemctl daemon-reload
rm -f /etc/logrotate.d/wavelab
echo "  units and logrotate removed"

echo "== unmounting and removing from fstab =="
umount "$MNT" 2>/dev/null || true
sed -i "\|^${IMG} |d" /etc/fstab
echo "  /etc/fstab clean"

if [ "$PURGE" = "--purge" ]; then
    rm -f "$IMG"; rmdir "$MNT" 2>/dev/null || true
    userdel wavelab 2>/dev/null || true
    rm -rf /opt/wavelab
    echo "  PURGED: image, user and /opt/wavelab removed"
else
    echo "  image KEPT at $IMG (use --purge to delete it)"
    echo "  reason: it holds unrecoverable data (liquidations, point-in-time news)"
fi

echo; echo "== state =="; df -h / | tail -1
pm2 jlist 2>/dev/null | python3 -c "
import sys,json
for a in json.load(sys.stdin): print(f\"  {a['name']:<24} {a['pm2_env']['status']}\")" 2>/dev/null || true
