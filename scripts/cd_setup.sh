#!/usr/bin/env bash
# Prepares the VPS to receive deployments from GitHub Actions. Run ONCE, by hand.
#
# The deploy key is RESTRICTED by `command=` in authorized_keys: even if someone got hold of it,
# all it can do is run the deploy script. It opens no shell, forwards no ports, and cannot read a
# file. That is the difference between a deploy key and a master key to the server, and it is
# exactly the pattern worth replicating at DR Solutions.
set -euo pipefail
KEY=/root/.ssh/wavelab_deploy

echo "== 1. deploy user and least privilege =="
# The script needs systemctl over ONE service and nothing else.
cat > /etc/sudoers.d/wavelab-deploy <<'EOF'
# Least privilege for continuous deployment: only this service, only these actions.
root ALL=(wavelab) NOPASSWD: /opt/wavelab/.venv/bin/python
EOF
chmod 440 /etc/sudoers.d/wavelab-deploy
visudo -c -f /etc/sudoers.d/wavelab-deploy

echo "== 2. deploy key =="
if [ ! -f "$KEY" ]; then
    ssh-keygen -t ed25519 -N "" -f "$KEY" -C "github-actions-wavelab" >/dev/null
    echo "  new key generated"
else
    echo "  already existed"
fi

echo "== 3. authorising it for the deploy script ONLY =="
PUB=$(cat "$KEY.pub")
mkdir -p /root/.ssh && touch /root/.ssh/authorized_keys
grep -v "github-actions-wavelab" /root/.ssh/authorized_keys > /tmp/ak.new 2>/dev/null || true
cat >> /tmp/ak.new <<EOF
command="/root/cd_deploy.sh \$SSH_ORIGINAL_COMMAND",no-agent-forwarding,no-port-forwarding,no-pty,no-X11-forwarding,no-user-rc $PUB
EOF
mv /tmp/ak.new /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
echo "  authorized_keys updated with a restricted command="

echo
echo "== PRIVATE KEY for the GitHub secret (SSH_DEPLOY_KEY) =="
echo "-----8<-----"
cat "$KEY"
echo "-----8<-----"
echo
echo "== HOST FINGERPRINT for the SSH_KNOWN_HOSTS secret =="
ssh-keyscan -t ed25519 -H "$(curl -s --max-time 8 ifconfig.co || echo 187.33.152.210)" 2>/dev/null
