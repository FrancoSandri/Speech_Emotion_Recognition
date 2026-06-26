# Sprint — Atención temporal sobre representaciones wav2vec multicapa

## Objetivo

Definir el mejor modelo SSL antes de incorporar una rama CNN-LSTM sobre log-Mel. El notebook existente compara:

```text
average_mean_std
learned_layers_mean_std
average_attention_statistics
learned_layers_attention_statistics
```

La hipótesis es que la atención temporal puede aportar valor cuando opera sobre una secuencia multicapa informativa, y no únicamente sobre la última capa de wav2vec.

## Implementación

Se modificó exclusivamente:

```text
notebooks/03c_temporal_pooling_layer_mixture.ipynb
```

No se creó ningún notebook adicional.

Los hidden states multicapa se guardan como:

```text
data/processed/features/wav2vec_multilayer_sequences/<file_id>.pt
```

Cada tensor tiene forma:

```text
[n_layers, n_frames, hidden_size]
```

El encoder wav2vec permanece congelado y la extracción no utiliza targets ni folds.

## Modelos nuevos

### `average_attention_statistics`

```text
promedio uniforme de capas por frame
→ attentive statistics pooling
→ LayerNorm
→ Dropout(0.50)
→ Linear
```

### `learned_layers_attention_statistics`

```text
mezcla softmax aprendida de capas por frame
→ attentive statistics pooling
→ LayerNorm
→ Dropout(0.50)
→ Linear
```

Los logits de la mezcla de capas se inicializan en cero, por lo que el modelo comienza desde pesos uniformes.

## Entrenamiento

- outer folds persistidos;
- inner `StratifiedGroupKFold` para early stopping;
- speaker-independent agrupado por `actor_id`;
- tres seeds: `13`, `42`, `73`;
- ensemble de probabilidades por outer fold;
- gradient clipping de `1.0`;
- encoder wav2vec sin fine-tuning.

## Presentación

El notebook muestra:

1. macro F1 medio y desvío entre folds;
2. curvas por outer fold;
3. variabilidad entre seeds;
4. pesos aprendidos entre capas;
5. concentración temporal de la atención;
6. matrices OOF normalizadas y diferenciales;
7. ejemplos de atención contra RMS;
8. síntesis rendimiento–estabilidad–complejidad.

## Orden de ejecución

```python
RUN_FEATURE_EXTRACTION = True
```

Después:

```python
RUN_STATIC_POOLING = True
RUN_LAYER_MIXTURE = True
```

Finalmente, por separado:

```python
RUN_AVERAGE_ATTENTION = True
RUN_LEARNED_LAYER_ATTENTION = True
```

Se recomienda ejecutar un bloque costoso por vez.

## Criterio de selección

Una atención reemplaza a `average_mean_std` cuando:

- mejora al menos `0.01` macro F1 medio; o
- mantiene una diferencia inferior a `0.01` y reduce al menos `20%` el desvío entre folds;
- ningún fold cae más de `0.03`;
- el desvío medio entre seeds es inferior a `0.02`.

Cuando las dos variantes de atención difieren menos de `0.01`, se prioriza `average_attention_statistics` por simplicidad.

## Cambio de alcance

Elastic Net y eGeMAPS fueron retirados del notebook temporal. La rama eGeMAPS permanece documentada en los sprints anteriores, pero no participa en esta selección del mejor modelo SSL.
