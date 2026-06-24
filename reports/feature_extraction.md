# Sprint 2 — Feature Extraction & EDA Report

**Fecha:** 2026-06-23  
**Seed:** 42

---

## Representaciones extraídas

| Representación | Dims | Archivos | Cache |
|---|---|---|---|
| eGeMAPSv02 Functionals | 88 | 1440 | `data/processed/features/egemaps.parquet` |
| wav2vec2-base (mean pooling) | 768 | 1440 | `data/processed/features/wav2vec.parquet` |

## EDA — Hallazgos (development pool, n=864)

### Top 3 features eGeMAPSv02 (ANOVA)
equivalentSoundLevel_dBp, loudness_sma3_meanRisingSlope, loudness_sma3_amean

### PCA
- eGeMAPSv02: 40 componentes para 95% varianza (40/88)
- wav2vec2:   96 componentes para 95% varianza (96/768)

## Decisiones para Sprint 3

- eGeMAPS se usa directamente (88 dims, manejable).
- wav2vec se explorará con PCA reducción dentro de CV.
- El contrato anti-leakage se mantiene: scaler y PCA se ajustan solo en train de cada fold.
