# Sprint — Representaciones temporales y selección de capas wav2vec

## Objetivo

Evaluar si la representación global de wav2vec mejora al incorporar dispersión temporal, información de capas intermedias y atención ligera, manteniendo el encoder congelado y los test finales aislados.

## Experimentos

1. `last layer + mean` frente a `last layer + mean/std`.
2. `last layer` frente a `average layers`.
3. Mezcla escalar aprendida de capas con tres semillas.
4. Attentive statistics pooling con media/desvío ponderados y tres semillas.
5. Logistic Regression Elastic Net sobre eGeMAPS, interpretada por familias acústicas.

## Criterio principal

```text
macro F1 medio speaker-independent · emotion_original
```

Se reportan además desvío entre outer folds, variabilidad entre semillas, peor fold, balanced accuracy y matrices OOF normalizadas.

## Reglas de aceptación

Una configuración se considera candidata si:

- mejora al menos `0.01` macro F1 medio, o
- conserva la media dentro de `0.01` y reduce el desvío entre folds al menos `20%`.

Ningún fold puede caer más de `0.03` frente a `last_mean`; para modelos entrenables, el desvío medio entre semillas debe ser menor a `0.02`.

## Aislamiento metodológico

- Hidden states: extracción congelada, sin targets ni folds.
- Scalers, layer mixture, atención, Elastic Net y early stopping: fit exclusivamente dentro de outer train.
- Outer validation: solo predicción y evaluación.
- Test final: no se carga en este sprint.

## Salida

El sprint debe concluir si la evidencia útil proviene de:

- variación temporal;
- capas intermedias;
- selección temporal aprendida;
- regularización acústica interpretable.

El entrenamiento adversarial de hablante queda condicionado a que la atención mejore el pooling pero persista la sensibilidad al actor.
