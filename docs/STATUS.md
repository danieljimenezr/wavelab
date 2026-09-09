# Deployment status

## Running on rec-ai-1 (187.33.152.210)

| Component | State | Notes |
|---|---|---|
| `wavelab-collect.service` | **active** | OKX + Bybit liquidations, derivatives every 5 min. 42 MB of RAM |
| Bar history | **4,756,158 1m bars** | 2017-08 → 2026-09, 110 months, 292 MB, 99.82% coverage |
| `wavelab.service` (engine + web) | **active** | 430,949 1m bars warmed up, API on `127.0.0.1:8000`. 223 MB of RAM |
| `wavelab-validate.timer` | pending | Will be enabled in M6b |

## Containment verified

8 GB loopback image on `/var/lib/wavelab`, `wavelab.slice` with 1.5 vCPU / 1536 MB /
256 MB of swap. Acceptance test: `bash /root/contain_test.sh`.

- Disk: `dd` asked for 12 GB and hit ENOSPC at 7,947. The host's `df /` moved by **0 MB**.
- Memory: `Result=oom-kill` in 3 s. The host's swap barely touched.
- Production: 6/6 pm2 apps online, three domains still answering during and after.

## Commands

```bash
# on the node
systemctl status wavelab-collect
journalctl -u wavelab-collect -f
bash /root/deploy.sh                    # git pull + uv sync + permissions
bash /root/contain_test.sh              # full containment test
bash /root/deprovision_vps.sh           # roll back (--purge to delete data too)

# from the Mac
ssh root@187.33.152.210
ssh -L 8000:localhost:8000 root@187.33.152.210   # tunnel; then open http://localhost:8000
```

## Data: what there is and where

```
/var/lib/wavelab/
  bars/BTCUSDT/YYYY-MM.parquet      110 months, 292 MB, zstd
  raw/liquidations/okx-*.jsonl      OKX (primary) — append-only, NEVER deleted
  raw/liquidations/bybit-*.jsonl    Bybit (secondary)
  raw/derivatives/*.jsonl.gz        funding, open interest, long/short, taker
  logs/hydrate.log
```

## Performance measured on the node

Full hydration: **233 MB in 96 s (2.4 MB/s)** — nine years of 1m bars in a minute and a half.
Over REST it would have been ~1,050 calls and more than half an hour.
