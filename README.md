# Speech Emotion Recognition — RAVDESS

Proyecto final de la materia I302 (Machine Learning / Deep Learning) — UdeSA.

## Objetivo

Desarrollar un sistema de **Speech Emotion Recognition (SER)** sobre la base RAVDESS, comparando representaciones acústicas expertas (eGeMAPSv02) con embeddings de modelos SSL (wav2vec2), bajo protocolo speaker-independent.

---

## Estructura del proyecto

```
.
├── configs/
│   └── config.yaml              # Única fuente de verdad (paths, seeds, params)
├── models/						 # Modelos guardados
├── data/
│   ├── raw/                     # wav crudo (gitignored)
│   └── processed/
│       ├── metadata.parquet     # Metadata + métricas de trimming
│       ├── splits.parquet       # Particiones + folds CV
│       ├── audio_trimmed/       # Audios recortados (gitignored)
│       └── features/
│           ├── egemaps.parquet
│           └── wav2vec.parquet
├── notebooks/
├── reports/
├── src/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── feature_selection/
│   ├── features/
│   ├── models/
│   ├── utils/
│   └── validation/
└── README.md
```

---

## Sprints

| Sprint                                     | Notebook                          | Estado                                   |
| ------------------------------------------ | --------------------------------- | ---------------------------------------- |
| 1 — Datos, audio, targets y particiones   | `01_data_audio_splits.ipynb`    | ✅                                       |
| 2 — Extracción de representaciones y EDA | `02_features_eda.ipynb`         | ✅                                       |
| 3 — Baselines y refinamiento (CV)         | `03_baselines_refinement.ipynb` | 🟨 Código completo; corridas pendientes |
| 4 — Evaluación final                     | `04_final_evaluation.ipynb`     | 🔲                                       |

---

## Setup

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
# Instalar dependencias y el paquete local
pip install -r requirements.txt
pip install -e .

# Abrir el notebook de baselines
jupyter notebook notebooks/03_baselines_refinement.ipynb

# Dentro del notebook se pueden activar por separado:
# RUN_BASELINES
# RUN_FEATURE_IMPORTANCE
# RUN_RFECV
# RUN_PCA
#
# Salidas consolidadas:
# reports/cv_results.csv
# reports/feature_selection.csv
```

---

## Principios clave

- **Anti-leakage**: toda transformación aprendida (scaler, PCA, RFECV) se ajusta solo en train de cada fold.
- **Test aislado**: los test finales nunca participan en EDA, selección de features ni tuning.
- **Reproducibilidad**: seed fijo en `configs/config.yaml`. Dos ejecuciones producen resultados idénticos.
- **Modularidad**: lógica en `src/`, notebooks solo importan y visualizan.
