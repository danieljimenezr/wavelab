# Mediciones reales (no razonadas)

El plan admitía que sus cifras de rendimiento eran *razonadas, no medidas*. Estas están medidas.

## M5 — Apple M5, 10 núcleos, 32 GB, macOS 26.6.2, CPython 3.13.15

Fecha: 2026-09-08 · numpy 2.5.3 · TA-Lib 0.7.1

| n velas | Batería TA-Lib (18 indicadores) | ATR-ZigZag (Python puro, O(n)) |
|--------:|--------------------------------:|-------------------------------:|
|   5.000 |    0,28 ms  (p95 0,32) |     1,19 ms  (p95 1,21) |
|  20.000 |    1,27 ms  (p95 1,42) |     4,88 ms  (p95 5,91) |
| 100.000 |    6,93 ms  (p95 7,26) |    24,51 ms  (p95 29,27) |
| 500.000 |   34,05 ms  (p95 34,32) |   122,30 ms  (p95 122,80) |

### Correcciones a lo que decía el plan

- La síntesis afirmaba «~5 ms sobre 500k velas en Python puro» para el ZigZag. **Es 122 ms: 25× optimista.**
- El crítico de completitud estimó «200-400 ms» para lo mismo. **Es 2-3× pesimista.**
- Sobre la ventana real de trabajo (5.000 velas) el presupuesto de 5 ms del plan se cumple con **18× de
  margen** en la batería y **4×** en el ZigZag.

### Consecuencias de diseño

1. **numba queda descartado definitivamente**, y por medición, no por corazonada. Recomputar entero cada
   vela es holgadamente viable.
2. El re-parseo offline sobre 9 años de 1m (~4,8 M velas) costará ~1,2 s de ZigZag y ~0,3 s de batería:
   perfectamente asumible como trabajo por lotes.
3. **Proyección al VPS** (x86_64, ~2-3× más lento en monohilo que un M5): la batería sobre 5.000 velas
   quedaría en ~0,8 ms y el ZigZag en ~3,5 ms. Sigue sin ser un problema. Pendiente de medir *in situ*
   en M0b y de anotar aquí la tabla real del nodo.
