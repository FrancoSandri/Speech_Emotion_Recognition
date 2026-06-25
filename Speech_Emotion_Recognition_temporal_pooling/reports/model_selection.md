# Referencia congelada de Sprint 3

Fecha de congelamiento: completar al ejecutar el notebook 03 con los resultados definitivos.

## Configuración primaria

```text
representation = wav2vec
protocol       = speaker_independent
target         = emotion_original
model          = logistic_regression
refinement     = none
```

## Baseline interpretable

```text
representation = egemaps
protocol       = speaker_independent
target         = emotion_original
model          = logistic_regression
refinement     = none
```

## Regla de selección

La métrica primaria es macro F1 medio speaker-independent. Si dos configuraciones difieren menos de 0.01, se prioriza la alternativa más simple y estable.

Las métricas efectivas deben copiarse desde `reports/cv_results.csv` después de completar Sprint 3. Los nuevos experimentos de diagnóstico, ablación y LDA no modifican retroactivamente estas referencias.

## Test isolation

Los dos test finales permanecen cerrados. Este documento debe existir antes de cualquier evaluación final.
