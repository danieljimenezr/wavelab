#!/usr/bin/env bash
# Continuous deployment. GitHub Actions invokes it over ssh with a key RESTRICTED to this script.
#
# The requirements it sets out to meet, because this is the pilot for what DR Solutions will do later:
#
#   1. ATOMIC with respect to the SHA. It deploys one specific commit, not "the latest on main",
#      which changes between the pipeline starting and finishing.
#   2. VERIFY BEFORE RESTARTING. If the new code does not even import, the service is left alone.
#   3. CHECK HEALTH AFTERWARDS. A service that starts is not a service that works.
#   4. ROLL BACK BY ITSELF. If the health check fails, it returns to the previous SHA and leaves it
#      running. A pipeline that deploys but does not roll back just moves the manual work to the
#      worst possible hour.
#   5. NO TRAMPLING. A lock keeps two simultaneous deployments from getting mixed up.
set -uo pipefail

DEST=/opt/wavelab
LOCK=/var/lock/wavelab-deploy.lock
NEW_SHA=${1:-}
HEALTH_URL=http://127.0.0.1:8000/api/status
# The nine-year warm-up takes ~120 s with CPUQuota=50%. The deadline is set WAY above that on
# purpose: a tight deadline turns every deployment into a coin flip, and that already happened —
# it ran out by two seconds, rolled back a perfectly good deployment and declared an incident that
# never existed.
ATTEMPTS=120          # 6 minutes

log() { printf '[cd] %s\n' "$*"; }
fail() { log "FAILED: $*"; exit 1; }

exec 9>"$LOCK"
flock -n 9 || fail "another deployment is already running"

[ -n "$NEW_SHA" ] || fail "the SHA to deploy is required"
[ -d "$DEST/.git" ] || fail "$DEST is not a repository"

cd "$DEST"
OLD_SHA=$(git rev-parse HEAD)
log "current $OLD_SHA -> requested $NEW_SHA"
[ "$OLD_SHA" = "$NEW_SHA" ] && { log "already deployed; nothing to do"; exit 0; }

check_health() {
    # We wait for `ready`, not for `ok`. `ok` only says the process is alive; `ready` says it can
    # actually do work. Confusing the two means signing off on a service that is not yet useful.
    local seen_ok=0
    for i in $(seq 1 $ATTEMPTS); do
        local r
        r=$(curl -sf --max-time 5 "$HEALTH_URL" 2>/dev/null || true)
        if echo "$r" | grep -q '"ready":true'; then
            log "healthy after $((i*3))s"
            return 0
        fi
        if echo "$r" | grep -q '"ok":true'; then
            [ "$seen_ok" = 0 ] && log "responding and warming up…"
            seen_ok=1
        fi
        sleep 3
    done
    return 1
}

deploy() {
    local sha=$1
    git fetch --quiet origin || return 1
    git reset --hard --quiet "$sha" || return 1
    chown -R wavelab:wavelab "$DEST"
    uv sync --frozen 2>&1 | tail -2 || return 1
    chown -R wavelab:wavelab "$DEST"
    # COLD check: if the new code does not even import, the live service is left alone.
    sudo -u wavelab "$DEST/.venv/bin/python" -c "import wavelab.server.app" || return 1
    systemctl restart wavelab || return 1
    check_health
}

log "deploying $NEW_SHA"
if deploy "$NEW_SHA"; then
    log "OK: $(git rev-parse --short HEAD) running and responding"
    curl -s --max-time 5 "$HEALTH_URL" | head -c 200; echo
    exit 0
fi

log "the health check failed; ROLLING BACK to $OLD_SHA"
if deploy "$OLD_SHA"; then
    fail "rolled back to $OLD_SHA, which does respond. Commit $NEW_SHA is NOT deployed."
fi
fail "CRITICAL: neither the new commit nor the previous one starts. Manual intervention required."
