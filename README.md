# Speech Emotion Recognition — RAVDESS

Proyecto final de la materia I302 (Aprendizaje Automático y Aprendizaje Profundo) — Universidad de San Andrés.

**Autores:** Alejandro García Giacchetta, Franco Agustín Sandri

Presentado en el AI Fest (julio 2026). Informe completo en [`GarciaG_Sandri_Informe_PF.pdf`](./GarciaG_Sandri_Informe_PF.pdf) y poster en [`GarciaG_Sandri_Poster_PF.pdf`](./GarciaG_Sandri_Poster_PF.pdf).

## Objetivo

Desarrollar un sistema de **Speech Emotion Recognition (SER)** sobre la base RAVDESS, comparando representaciones acústicas expertas (eGeMAPSv02) con embeddings de modelos autosupervisados (wav2vec 2.0), bajo un protocolo estrictamente **speaker-independent**: el modelo nunca ve en entrenamiento a los mismos hablantes sobre los que después se evalúa, para asegurar que aprenda patrones emocionales y no identidades de voz.

## Resultados principales

| Configuración | Balanced Accuracy (CV) | Balanced Accuracy (test) |
|---|---|---|
| eGeMAPS + Regresión Logística | 0.493 ± 0.069 | 0.555 |
| wav2vec (última capa) + LR | 0.552 ± 0.071 | — |
| **wav2vec multicapa (avg. mean-std) + LR** | **0.736 ± 0.049** | **0.828** |

- La mejor configuración promedia estadísticas temporales (media y desvío) de las 13 capas del encoder wav2vec congelado y clasifica con una regresión logística simple — superando a variantes con atención temporal y mezclas aprendidas más complejas.
- Mejora de **+0.305** en balanced accuracy sobre el baseline obligatorio (Random Forest + eGeMAPS).
- Principal fuente de error: confusión tristeza–calma (recall 0.50), coherente con su bajo *arousal* compartido.
- En transferencia zero-shot a otros dominios (EMO-DB, RAVDESS Song), wav2vec transfiere mejor que eGeMAPS y responde mejor a la adaptación de la cabeza de clasificación manteniendo el encoder congelado.
- RFECV y los mecanismos de atención temporal no aportaron mejoras robustas ni estables entre folds.

Detalle completo de metodología, tablas y figuras en el informe.

## Notebooks (pipeline)

| Notebook | Contenido |
|---|---|
| `01_data_audio_splits.ipynb` | Preprocesamiento de audio y armado de splits speaker-independent |
| `02_features_eda.ipynb` | Extracción de features (eGeMAPS, wav2vec) y análisis exploratorio |
| `03a_baselines_refinement.ipynb` | Baselines (RF, LR, MLP) y refinamientos (PCA, RFECV, LDA) |
| `03b_diagnostics_interpretability_projection.ipynb` | Diagnóstico e interpretabilidad de representaciones |
| `03c_temporal_pooling_layer_mixture.ipynb` | Pooling temporal multicapa, mezcla aprendida y atención |
| `04_modelo_final.ipynb` | Entrenamiento y evaluación del modelo final sobre test |
| `05_domain_transfer_head_adaptation.ipynb` | Transferencia zero-shot y adaptación de cabeza a otros dominios |

## Estructura del proyecto

```
.
├── configs/
│   └── config.yaml              # Única fuente de verdad (paths, seeds, params)
├── models/                      # Modelos guardados
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
├── GarciaG_Sandri_Informe_PF.pdf
├── GarciaG_Sandri_Poster_PF.pdf
└── README.md
```

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
jupyter notebook notebooks/03a_baselines_refinement.ipynb

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

Para reproducir el resto del pipeline, correr los notebooks en el orden numérico indicado en la tabla de arriba.

## Principios clave

- **Anti-leakage**: toda transformación aprendida (scaler, PCA, RFECV) se ajusta solo en train de cada fold.
- **Test aislado**: los datos de test finales nunca participan en EDA, selección de features ni tuning.
- **Reproducibilidad**: seed fijo en `configs/config.yaml`. Dos ejecuciones producen resultados idénticos.
- **Modularidad**: lógica en `src/`, notebooks solo importan y visualizan.

## Trabajo futuro

Ampliar la evaluación a más corpora y hablantes, explorar fine-tuning parcial del encoder, incorporar representaciones log-Mel y estudiar regularización explícita de identidad.