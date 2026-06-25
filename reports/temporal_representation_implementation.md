# Implementación — Representaciones temporales y selección de capas

## Alcance

Se incorporó un notebook independiente para comparar, sobre los folds persistidos de `development`:

- última capa wav2vec con `mean` y `mean + std`;
- promedio uniforme de capas;
- mezcla escalar aprendida de capas;
- attentive statistics pooling sobre secuencias congeladas;
- Logistic Regression Elastic Net sobre eGeMAPS.

No se modifica el encoder wav2vec ni se accede a los test finales.

## Notebook

```text
notebooks/03c_temporal_pooling_layer_mixture.ipynb
```

Todos los bloques costosos están desactivados por defecto. Orden recomendado:

1. `RUN_FEATURE_EXTRACTION = True` una única vez.
2. `RUN_STATIC_POOLING = True`.
3. `RUN_LAYER_MIXTURE = True`.
4. `RUN_ATTENTIVE_POOLING = True`.
5. `RUN_EGEMAPS_ELASTIC_NET = True`.

Para la primera corrida se recomienda mantener:

```python
RUN_SECONDARY_TARGETS = False
RUN_SPEAKER_DEPENDENT = False
```

Así se prioriza `speaker-independent · emotion_original` antes de ampliar la matriz.

## Módulos nuevos

```text
src/features/wav2vec_temporal.py
src/models/layer_mixture.py
src/models/attentive_statistics_pooling.py
src/models/elastic_net.py
src/experiments/temporal_representation.py
```

Se reutilizan `feature_store`, `run_cv`, métricas, mappings, folds y el linear probe de sprints previos.

## Artefactos

```text
data/processed/features/wav2vec_layer_statistics.npz
data/processed/features/wav2vec_sequences/<file_id>.pt
reports/temporal_representation_results.csv
reports/temporal_oof_predictions.csv
reports/layer_mixture_weights.csv
reports/attention_seed_results.csv
reports/attention_weights_selected.npz
reports/egemaps_elastic_net_results.csv
reports/egemaps_elastic_net_family_importance.csv
```

## Presentación

El notebook organiza los resultados mediante:

- error bars para media y desvío;
- scatter de rendimiento frente a estabilidad;
- curva de pesos por capa;
- curvas de macro F1 por outer fold;
- boxplot de variabilidad entre semillas;
- matrices OOF normalizadas y diferenciales;
- curvas temporales de atención y RMS;
- comparación L2/Elastic Net por familias acústicas;
- scatter final de rendimiento, estabilidad y dimensionalidad.

## Validaciones realizadas

- compilación de todos los módulos;
- smoke test sintético de mezcla aprendida de capas;
- smoke test sintético de attentive statistics pooling;
- smoke test sintético de Elastic Net nested;
- validación sintáctica de todas las celdas del notebook.

La extracción real no se ejecutó en este paquete porque el snapshot no contiene los audios trimmed.
