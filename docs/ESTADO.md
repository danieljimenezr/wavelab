# Estado del despliegue

## En marcha en rec-ai-1 (187.33.152.210)

| Componente | Estado | Notas |
|---|---|---|
| `wavelab-collect.service` | **activo** | Liquidaciones OKX + Bybit, derivados cada 5 min. 42 MB de RAM |
| Histórico de velas | **4.756.158 velas de 1m** | 2017-08 → 2026-09, 110 meses, 292 MB, cobertura 99,82% |
| `wavelab.service` (motor) | pendiente | Falta el motor (M2-M4) |
| `wavelab-validate.timer` | pendiente | Se activará en M6b |

## Contención verificada

Imagen loopback de 8 GB en `/var/lib/wavelab`, `wavelab.slice` con 1,5 vCPU / 1536 MB /
256 MB de swap. Prueba de aceptación: `bash /root/contain_test.sh`.

- Disco: `dd` pidió 12 GB, topó a los 7.947 con ENOSPC. `df /` del host **0 MB de variación**.
- Memoria: `Result=oom-kill` en 3 s. Swap del host apenas tocado.
- Producción: 6/6 apps de pm2 online, tres dominios respondiendo durante y después.

## Comandos

```bash
# en el nodo
systemctl status wavelab-collect
journalctl -u wavelab-collect -f
bash /root/deploy.sh                    # git pull + uv sync + permisos
bash /root/contain_test.sh              # prueba de contención completa
bash /root/deprovision_vps.sh           # revertir (--purge para borrar datos)

# desde el Mac
ssh root@187.33.152.210
ssh -L 8000:localhost:8000 root@187.33.152.210   # túnel para la interfaz (cuando exista)
```

## Datos: qué hay y dónde

```
/var/lib/wavelab/
  bars/BTCUSDT/YYYY-MM.parquet      110 meses, 292 MB, zstd
  raw/liquidations/okx-*.jsonl      OKX (primaria) — append-only, NUNCA se borra
  raw/liquidations/bybit-*.jsonl    Bybit (secundaria)
  raw/derivatives/*.jsonl.gz        funding, open interest, long/short, taker
  logs/hydrate.log
```

## Rendimiento medido en el nodo

Hidratación completa: **233 MB en 96 s (2,4 MB/s)** — nueve años de velas de 1m en minuto y medio.
Por REST habrían sido ~1.050 llamadas y más de media hora.
