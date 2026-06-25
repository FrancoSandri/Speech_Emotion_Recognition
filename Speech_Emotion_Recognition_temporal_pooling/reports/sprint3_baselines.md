# Sprint 3 — Implementación de baselines

## Alcance implementado

Se incorporó una infraestructura común para evaluar:

- eGeMAPSv02 y wav2vec mean pooling;
- speaker-dependent y speaker-independent;
- emoción original, cuadrante directo y evaluación 8→4;
- Random Forest, Logistic Regression y MLP shallow.

## Contrato metodológico

- Los folds se leen desde `splits.parquet` y no se reconstruyen.
- Todos los experimentos utilizan `partition == development`.
- Random Forest recibe las features sin escalado.
- Logistic Regression y MLP incluyen `StandardScaler` dentro del pipeline.
- El early stopping del MLP utiliza una fracción interna de outer train.
- Las métricas se calculan mediante una única implementación global.
- El target 8→4 conserva métricas originales de ocho clases y métricas colapsadas de cuatro cuadrantes.

## Módulos

- `src/models/random_forest.py`
- `src/models/linear_probe.py`
- `src/models/mlp.py`
- `src/experiments/baselines.py`
- `src/experiments/cross_validation.py`
- `src/evaluation/metrics.py`
- `src/evaluation/reporting.py`
- `notebooks/03_baselines_refinement.ipynb`

## Validación funcional

Se ejecutaron corridas de humo completas sobre los cinco folds para:

- los tres modelos con eGeMAPS, cuadrantes y protocolo speaker-independent;
- Logistic Regression con wav2vec, ocho emociones y protocolo speaker-dependent;
- Logistic Regression con eGeMAPS en el escenario 8→4.

Las corridas verificaron dimensiones, entrenamiento, predicción, agregación de métricas y preservación de métricas originales en 8→4. Estos resultados no constituyen selección de modelo y no se guardaron en `cv_results.csv`.

## Ejecución

El notebook no dispara automáticamente la matriz completa. Para generarla se debe cambiar:

```python
RUN_BASELINES = True
```

La salida se consolida en:

```text
reports/cv_results.csv
```
