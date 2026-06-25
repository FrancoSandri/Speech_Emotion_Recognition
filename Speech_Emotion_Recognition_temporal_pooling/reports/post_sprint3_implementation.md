# Diagnóstico, interpretación acústica y proyección supervisada

## Alcance

Se agregó un único notebook posterior al Sprint 3:

```text
notebooks/03b_diagnostics_interpretability_projection.ipynb
```

El notebook mantiene congeladas las referencias de Sprint 3 y prioriza una presentación visual compacta.

## Nuevos módulos

```text
src/evaluation/diagnostics.py
src/features/egemaps_families.py
src/feature_selection/supervised_projection.py
src/experiments/diagnostics_interpretability.py
```

### Diagnóstico OOF

Calcula métricas por actor, recall actor × emoción y métricas por fold a partir de predicciones out-of-fold. No accede a los test finales.

### Familias eGeMAPS

Las 88 features se asignan exactamente a siete familias:

```text
F0 y prosodia
loudness y energía
MFCC
formantes
espectro
calidad vocal
voicing y temporalidad
```

La importancia se agrega mediante suma absoluta y media por feature. La media por feature evita que una familia domine únicamente por contener más variables.

### Ablaciones

Se reutiliza el motor común de cross-validation para ejecutar:

```text
all-features
family-only
leave-one-family-out
```

Los resultados se guardan en:

```text
reports/egemaps_family_results.csv
```

### LDA regularizada

Se implementó:

```text
StandardScaler
→ LDA shrinkage
→ Logistic Regression
```

La transformación usa scores discriminantes regularizados y produce `n_clases - 1` dimensiones. Esto permite mantener una proyección supervisada explícita también en representaciones de alta dimensión.

Los resultados se agregan al mismo:

```text
reports/cv_results.csv
```

con:

```text
refinement = lda_shrinkage
```

## Visualizaciones principales

El notebook presenta:

- referencia Sprint 3 como media ± desvío;
- macro F1 por actor;
- heatmap actor × emoción;
- recall por emoción entre actores;
- rendimiento por fold;
- heatmap de importancia por familia y método;
- estabilidad por familia como media ± desvío;
- delta de ablaciones contra eGeMAPS completo;
- rendimiento vs estabilidad de las familias;
- all-features vs PCA 95% vs LDA;
- matrices de confusión normalizadas baseline, LDA y diferencia;
- scatter final de rendimiento, estabilidad y dimensionalidad.

## Ejecución

Flags disponibles:

```python
RUN_OOF_DIAGNOSTICS = False
RUN_FAMILY_ABLATIONS = False
RUN_LDA = False
RUN_LDA_WAV2VEC = True
```

Se recomienda ejecutar cada bloque por separado. LDA sobre wav2vec es opcional y se mantiene en un flag independiente por su mayor costo computacional.

## Validación realizada

- Los módulos compilan correctamente.
- Las 88 features eGeMAPS quedan cubiertas una sola vez por el mapping de familias.
- Las ablaciones completaron un smoke test real sobre los cinco folds speaker-independent.
- LDA completó un smoke test real sobre eGeMAPS.
- El notebook completo ejecutó sin errores con los flags desactivados.
