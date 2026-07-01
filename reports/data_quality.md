# Sprint 1 — Data Quality Report

**Fecha:** 2026-06-30  
**Seed:** 42  
**Status:** ✅ PASS WITH WARNINGS (1)

---

## Dataset

| Métrica | Valor |
|---|---|
| Total archivos | 1440 |
| Actores | 24 (12F / 12M) |
| Emociones (8 clases) | neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Cuadrantes (4 clases) | Q1 / Q2 / Q3 / Q4 (Russell 1980) |

## Particiones

| Partición | Archivos | Actores |
|---|---|---|
| development | 864 | 18 actores, grupos de utterance no reservados |
| test_speaker_dependent | 216 | 18 actores conocidos, grupos completos reservados |
| test_speaker_independent | 360 | 6 actores: [2, 4, 6, 19, 21, 23] |

## Trimming

| Parámetro | Valor |
|---|---|
| frame_length | 25 ms |
| hop_length | 10 ms |
| threshold_db | -35 dB |
| padding | 150 ms |
| trim_ratio medio | 0.391 |
| Archivos con trim_ratio > 0.5 | 83 |

## Auditorías

| Verificación | Estado |
|---|---|
| Sanity metadata | PASS WITH WARNINGS |
| Sanity splits | PASS |
| Sanity audio | PASS |
| Leakage audit | PASS |
| Reproducibilidad | PASS |
