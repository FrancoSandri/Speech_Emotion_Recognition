# Speech Emotion Recognition — RAVDESS

Proyecto final de la materia I302 (Machine Learning / Deep Learning) — UdeSA.

## Objetivo

Desarrollar un sistema de **Speech Emotion Recognition (SER)** sobre la base RAVDESS, comparando representaciones acústicas expertas (eGeMAPSv02) con embeddings de modelos SSL (wav2vec2), bajo protocolos speaker-dependent e speaker-independent.

---

## Estructura del proyecto

```
.
├── configs/
│   └── config.yaml              # Única fuente de verdad (paths, seeds, params)
├── data/
│   ├── raw/                     # RAVDESS crudo (gitignored)
│   └── processed/
│       ├── metadata.parquet     # Metadata + métricas de trimming
│       ├── splits.parquet       # Particiones + folds CV
│       ├── audio_trimmed/       # Audios recortados (gitignored)
│       └── features/
│           ├── egemaps.parquet
│           └── wav2vec.parquet
├── notebooks/
│   ├── 01_data_audio_splits.ipynb   # Sprint 1
│   ├── 02_features_eda.ipynb        # Sprint 2
│   ├── 03_baselines_refinement.ipynb # Sprint 3
│   └── 04_final_evaluation.ipynb    # Sprint 4
├── reports/
│   ├── figures/
│   ├── data_quality.md
│   ├── feature_extraction.md
│   └── model_selection.md
├── models/                      # Modelos finales (gitignored)
├── src/
│   ├── config/
│   │   └── contracts.py         # Contratos compartidos (columnas, nombres)
│   ├── data/
│   │   ├── metadata.py          # Parseo RAVDESS, targets, utterance_group_id
│   │   ├── audio_cleaning.py    # Trimming sin normalización de amplitud
│   │   └── splits.py            # Particiones y folds CV
│   ├── evaluation/
│   │   ├── metrics.py           # Métricas globales y evaluación 8→4
│   │   └── reporting.py         # Consolidación y persistencia de CV
│   ├── experiments/
│   │   ├── cross_validation.py          # Loop CV anti-leakage
│   │   ├── baselines.py                 # Matriz común de baselines
│   │   └── representation_refinement.py # Feature importance, RFECV y PCA
│   ├── feature_selection/
│   │   ├── importance.py        # Importancias agregadas por outer fold
│   │   ├── rfecv.py             # RFECV nested con grupos internos
│   │   └── pca.py               # Scaler + PCA dentro del pipeline
│   ├── features/
│   │   ├── extract_egemaps.py   # eGeMAPSv02 vía openSMILE
│   │   ├── extract_wav2vec.py   # wav2vec2 frozen mean pooling
│   │   └── feature_store.py     # Acceso unificado a features
│   ├── models/
│   │   ├── random_forest.py      # RF sin escalado
│   │   ├── linear_probe.py       # Scaler + Logistic Regression
│   │   └── mlp.py                # Scaler + MLP shallow
│   ├── utils/
│   │   ├── config.py            # Carga de config.yaml
│   │   ├── io.py                # I/O de artefactos
│   │   ├── logging.py           # Logger estándar
│   │   └── reproducibility.py   # Seed global
│   └── validation/
│       ├── leakage.py           # Auditoría anti-leakage
│       ├── sanity.py            # Validaciones de metadata, splits y audio
│       └── schemas.py           # Schemas de columnas requeridas
└── README.md
```

---

## Sprints

| Sprint | Notebook | Estado |
|--------|----------|--------|
| 1 — Datos, audio, targets y particiones | `01_data_audio_splits.ipynb` | ✅ |
| 2 — Extracción de representaciones y EDA | `02_features_eda.ipynb` | ✅ |
| 3 — Baselines y refinamiento (CV) | `03_baselines_refinement.ipynb` | 🟨 Código completo; corridas pendientes |
| 4 — Evaluación final | `04_final_evaluation.ipynb` | 🔲 |

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

## Diagnóstico e interpretación posterior al Sprint 3

El notebook:

```text
notebooks/03b_diagnostics_interpretability_projection.ipynb
```

consolida en una única ejecución:

- diagnóstico OOF por actor, emoción y fold;
- interpretación eGeMAPS por familias acústicas;
- ablaciones `family-only` y `leave-one-family-out`;
- proyección supervisada LDA regularizada;
- matrices de confusión normalizadas baseline vs LDA.

Los flags de ejecución están desactivados por defecto para reutilizar los CSV ya calculados. El análisis usa únicamente `development` y los folds persistidos.
