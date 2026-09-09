#!/usr/bin/env bash
# Proves that wavelab CANNOT fill the host's disk or bring down the apps that bill customers.
# The M0b acceptance criterion is not visual inspection: it is this test.
set -uo pipefail
MNT=/var/lib/wavelab

echo "== BEFORE =="
HOST_BEFORE=$(df --output=avail / | tail -1)
echo "  host / available: $((HOST_BEFORE/1024)) MB"
df -h "$MNT" | tail -1 | sed 's/^/  wavelab: /'
for a in valorvenal evolution-api admin-site; do
  printf '  %-16s %s\n' "$a" "$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
print(next((x['pm2_env']['status'] for x in json.load(sys.stdin) if x['name']=='$a'),'?'))" 2>/dev/null)"
done

echo
echo "== FILLING wavelab's image on purpose =="
# ionice idle + nice 19 + O_DIRECT: writes without stealing I/O from the apps that bill customers
# and without evicting their page cache.
DD_OUT=$(ionice -c 3 nice -n 19 runuser -u wavelab -- \
    dd if=/dev/zero of="$MNT/FILL.tmp" bs=1M count=12000 oflag=direct 2>&1)
echo "$DD_OUT" | tail -3 | sed 's/^/  /'
WROTE_MB=$(echo "$DD_OUT" | grep -oE '^[0-9]+\+[0-9]+ records out' | grep -oE '^[0-9]+' | head -1)
WROTE_MB=${WROTE_MB:-0}

echo
echo "== AFTER =="
HOST_AFTER=$(df --output=avail / | tail -1)
echo "  host / available: $((HOST_AFTER/1024)) MB"
DELTA=$(( (HOST_BEFORE - HOST_AFTER) / 1024 ))
echo "  change on the host: ${DELTA} MB"
df -h "$MNT" | tail -1 | sed 's/^/  wavelab: /'
for a in valorvenal evolution-api admin-site; do
  printf '  %-16s %s\n' "$a" "$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json
print(next((x['pm2_env']['status'] for x in json.load(sys.stdin) if x['name']=='$a'),'?'))" 2>/dev/null)"
done
curl -s -o /dev/null -w "  valorvenal.dr-techsolutions.com -> HTTP %{http_code}\n" \
     --max-time 10 https://valorvenal.dr-techsolutions.com || true

INNER_PCT=$(df --output=pcent "$MNT" | tail -1 | tr -dc '0-9')
rm -f "$MNT/FILL.tmp"

echo
FAIL=0
# (1) The writer has to HIT A WALL. We asked for 12 GB inside an 8 GB image: if dd wrote all of
#     them, the image is containing nothing.
if [ "$WROTE_MB" -lt 11000 ]; then
  echo "  ✅ the writer hit the wall at ${WROTE_MB} MB of the 12000 requested (ENOSPC)"
else
  echo "  ❌ dd wrote ${WROTE_MB} MB: it hit no limit at all"; FAIL=1
fi
# (2) The inner filesystem reached 100%: the ceiling is its own, not the host's.
if [ "${INNER_PCT:-0}" -ge 99 ]; then
  echo "  ✅ wavelab's image reached ${INNER_PCT}%: IT set the limit"
else
  echo "  ❌ the image stopped at ${INNER_PCT}%: it hit a wall somewhere else"; FAIL=1
fi
# (3) The host does not move. With the image reserved, writing inside it costs the host nothing.
if [ "$DELTA" -lt 100 ]; then
  echo "  ✅ the host's disk did not move (${DELTA} MB, measurement noise)"
else
  echo "  ❌ the host lost ${DELTA} MB: the image was sparse, not reserved"; FAIL=1
fi

# ---------------------------------------------------------------- 2) memory
echo
echo "== MEMORY: a process that allocates RAM without stopping, on the service's budget =="
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
echo "  unit result: $RESULT (after ${i}s)"
echo "  host swap: ${SWAP_USED} MB used"

# (4) It has to DIE, not hang. A throttled service that crawls along forever never fires
#     Restart=always: you are left with a frozen chart and no error.
if [ "$RESULT" = "oom-kill" ]; then
  echo "  ✅ the cgroup killed it (oom-kill) in ${i}s: die and restart, not hang"
else
  echo "  ❌ result '$RESULT' instead of oom-kill: it throttled without dying"; FAIL=1
fi
# (5) The host's swap is the real path to damage: exhausting it makes the whole machine crawl.
if [ "${SWAP_USED:-9999}" -lt 600 ]; then
  echo "  ✅ the host's swap was barely touched (${SWAP_USED} MB)"
else
  echo "  ❌ the host's swap is at ${SWAP_USED} MB: MemorySwapMax needs bounding"; FAIL=1
fi

# ---------------------------------------------------------------- 3) production
echo
echo "== PRODUCTION, after both tests =="
for a in valorvenal evolution-api admin-site evolution-outbox-worker sara-restaurant-worker baja-webhook; do
  st=$(pm2 jlist 2>/dev/null | python3 -c "
import sys,json;print(next((x['pm2_env']['status'] for x in json.load(sys.stdin) if x['name']=='$a'),'?'))" 2>/dev/null)
  printf '  %-26s %s\n' "$a" "$st"
  [ "$st" != "online" ] && FAIL=1
done
for d in valorvenal.dr-techsolutions.com dr-techsolutions.com admin.dr-techsolutions.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "https://$d" || echo 000)
  # What is being checked is "is it still serving?", not "does it return exactly 200?".
  # Any 2xx or 3xx counts: admin.dr-techsolutions.com answers 307 (Next.js auth redirect) and that
  # is normal operation. A real failure = 4xx, 5xx or 000 (no connection).
  case "$code" in
    2??|3??) printf '  %-34s HTTP %s  ok\n' "$d" "$code" ;;
    *)       printf '  %-34s HTTP %s  <-- FAILURE\n' "$d" "$code"; FAIL=1 ;;
  esac
done

echo
[ "$FAIL" -eq 0 ] && echo "  CONTAINMENT VERIFIED" || { echo "  CONTAINMENT NOT VERIFIED"; exit 1; }
