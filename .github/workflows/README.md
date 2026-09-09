# The pipeline, and why it looks like this

This file is the **pilot** for migrating DR Solutions. The decisions were made on the assumption
that it gets copied, not that it works once.

## The four decisions that matter

**1. The deploy key is not a key to the server.** In `authorized_keys` it carries
`command="/root/cd_deploy.sh $SSH_ORIGINAL_COMMAND"` plus `no-pty,no-port-forwarding,
no-agent-forwarding`. Even if someone got hold of it, **all it can do is run the deployment**: it
opens no shell, forwards no ports, reads no files. That is the difference between a deploy key and
root access, and it costs nothing.

**2. Automatic rollback lives on the SERVER, not in the pipeline.** If the GitHub runner dies
halfway through a deployment, the rollback has to happen anyway. `cd_deploy.sh` checks health after
restarting and, if it fails, goes back to the previous SHA on its own and leaves it running. A
pipeline that deploys but does not roll back just moves the manual work to the worst possible hour.

**3. `cancel-in-progress: false` on deployments.** Cancelling a deployment halfway leaves the server
in an indeterminate state — dependencies half installed, service stopped. That is worse than making
the next one wait. On *tests* cancelling is the right call; on deployments, never.

**4. Network tests are excluded.** The ones marked `net` depend on Binance answering. A pipeline
that fails because of someone else's API teaches the team to ignore failures, and a pipeline that
gets ignored is worse than no pipeline at all.

## Secrets

| secret | what it is |
|---|---|
| `SSH_DEPLOY_KEY` | ed25519 private key, restricted by `command=` |
| `SSH_KNOWN_HOSTS` | host fingerprint, so verification does not have to be turned off |
| `VPS_HOST` | the server's IP |

`StrictHostKeyChecking=no` is never used. Turning off host verification turns any DNS or route
hijack into handing the private key to the attacker.

## To replicate it in another repository

1. `bash scripts/cd_setup.sh` on the target server (it generates the key and restricts it).
2. Pipe the private key into the secret without it passing across a screen:
   `ssh root@host 'cat /root/.ssh/wavelab_deploy' | gh secret set SSH_DEPLOY_KEY --repo owner/repo`
3. Copy `cd_deploy.sh`, adapting the service name and the health URL.
4. Copy this workflow.

Step 2 matters: a key printed to a terminal, into a log or into a chat is compromised from that
moment on. It happened while this very pipeline was being built and the key had to be regenerated.
