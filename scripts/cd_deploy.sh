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
        if echo "$r" | grep -q '"failed":true'; then
            # The third state, and the only one worth giving up on early. Warm-up crashed, so
            # `ready` will NEVER arrive and every remaining attempt is dead time. Until
            # /api/status grew `failed`, a crashed start looked EXACTLY like a slow one on the
            # wire, so this loop sat out all six minutes and then rolled back with "the health
            # check failed" and no cause. The cause was in the process's own stdout the whole
            # time; now it is in the payload too, so print it and stop.
            log "WARM-UP FAILED on the new revision; not waiting out the deadline"
            log "  $(echo "$r" | sed -n 's/.*"error":"\([^"]*\)".*/\1/p')"
            return 1
        fi
        if echo "$r" | grep -q '"ok":true'; then
            [ "$seen_ok" = 0 ] && log "responding and warming up…"
            seen_ok=1
        fi
        # A heartbeat every 30 s. Warmup can take minutes, and a deploy that prints nothing for
        # that long looks wedged to a human and looks idle to whatever NAT sits on the ssh
        # connection — a silent pipe is exactly what gets collected, and the deploy then reports
        # failure for a rollout that actually worked.
        [ $((i % 10)) -eq 0 ] && log "still waiting for ready ($((i*3))s of $((ATTEMPTS*3))s)"
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

# This script is installed by cd_setup.sh, NOT by the deploy, so it does not update itself the way
# everything else does. That drift is invisible: it only surfaces the day a deploy fails for a
# reason the newer version would have handled. It has already happened once — the copy on the
# server was still polling /api/estado after the codebase was translated, so the health check could
# never have passed, and the rollout survived only because ssh died before the rollback ran.
#
# So: after a deploy that WORKED, adopt the version that came with it. Not before — a bash script
# is read incrementally as it runs, and overwriting the one currently executing is how you get a
# syntax error halfway through your own deployment. Next run uses it; this run finishes as itself.
adopt_new_self() {
    local incoming="$DEST/scripts/cd_deploy.sh"
    [ -r "$incoming" ] || return 0
    cmp -s "$incoming" "$0" && return 0
    bash -n "$incoming" || { log "WARNING: the incoming cd_deploy.sh does not parse; keeping this one"; return 0; }
    install -m 700 -o root -g root "$incoming" "$0" \
        && log "cd_deploy.sh updated itself; the next deploy runs the new one"
}

log "deploying $NEW_SHA"
if deploy "$NEW_SHA"; then
    log "OK: $(git rev-parse --short HEAD) running and responding"
    curl -s --max-time 5 "$HEALTH_URL" | head -c 200; echo
    adopt_new_self
    exit 0
fi

log "the health check failed; ROLLING BACK to $OLD_SHA"
if deploy "$OLD_SHA"; then
    fail "rolled back to $OLD_SHA, which does respond. Commit $NEW_SHA is NOT deployed."
fi
fail "CRITICAL: neither the new commit nor the previous one starts. Manual intervention required."
