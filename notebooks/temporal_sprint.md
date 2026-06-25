# Sprint — Representaciones temporales y selección de capas para SER

## 1. Propósito

Evaluar si la generalización speaker-independent puede mejorarse modificando la forma en que se construye la representación global de wav2vec, sin realizar fine-tuning del encoder SSL.

El sprint estudiará, de forma incremental:

```text
estadísticas temporales
selección de capas wav2vec
mezcla aprendida de capas
pooling temporal con atención
regularización interpretable de eGeMAPS
```

La hipótesis principal es:

> El mean pooling de la última capa de wav2vec contiene información emocional útil, pero elimina variaciones temporales relevantes y puede conservar una representación excesivamente dependiente del hablante. Incorporar dispersión temporal, información de capas intermedias y ponderación temporal aprendida puede mejorar la robustez sin modificar el encoder preentrenado.

Este sprint no incluye todavía:

```text
fine-tuning completo de wav2vec
autoencoders
VAE
CNN sobre embeddings globales
speaker-adversarial training
evaluación sobre test final
```

El entrenamiento adversarial de hablante queda definido como extensión condicionada a los resultados.

---

# 2. Referencia congelada

Toda comparación utilizará como referencia:

```text
wav2vec last_hidden_state
→ mean pooling temporal
→ StandardScaler
→ Logistic Regression
```

La referencia se evalúa exclusivamente mediante los folds persistidos del conjunto `development`.

Configuración principal:

```text
representation = wav2vec_mean
protocol = speaker_independent
target = emotion_original
model = logistic_regression
```

Configuraciones secundarias:

```text
emotion_quadrant
emotion_original_eval_quadrant
speaker_dependent
```

El criterio principal seguirá siendo:

```text
macro F1 medio speaker-independent
```

También se reportará:

```text
desvío estándar entre outer folds
balanced accuracy
weighted F1
accuracy
peor fold
recall por clase
```

---

# 3. Notebook

Se implementará un único notebook:

```text
notebooks/03c_temporal_pooling_layer_mixture.ipynb
```

El notebook deberá:

```text
cargar folds y referencias existentes
extraer o cargar artefactos wav2vec congelados
ejecutar experimentos mediante funciones de src
mostrar comparaciones visuales
generar predicciones OOF
documentar conclusiones
```

No deberá:

```text
reimplementar modelos completos
reconstruir folds
usar test final
guardar modelos por fold
mostrar tablas extensas
duplicar métricas o loops de cross-validation
```

Flags recomendados:

```python
RUN_FEATURE_EXTRACTION = False
RUN_STATIC_POOLING = False
RUN_LAYER_ANALYSIS = False
RUN_LEARNED_LAYER_MIXTURE = False
RUN_ATTENTIVE_POOLING = False
RUN_EGEMAPS_ELASTIC_NET = False
```

---

# 4. Artefactos wav2vec requeridos

El Parquet actual contiene únicamente:

```text
mean(last_hidden_state)
```

Para este sprint se necesitan dos nuevos tipos de artefactos congelados.

## 4.1 Estadísticas por capa

Para cada audio y cada capa wav2vec:

```text
mean(z_t^l)
std(z_t^l)
```

Forma conceptual:

```text
n_files × n_layers × hidden_size
```

Guardar como:

```text
data/processed/features/wav2vec_layer_statistics.npz
```

El artefacto debe contener:

```text
file_ids
layer_means
layer_stds
model_name
model_revision
sample_rate
n_layers
hidden_size
```

No debe depender de targets ni folds.

## 4.2 Secuencia de última capa

Para attentive statistics pooling se necesita:

```text
z_t = last_hidden_state
```

Debido a que la longitud temporal cambia entre audios, se almacenará una secuencia por archivo:

```text
data/processed/features/wav2vec_sequences/
    <file_id>.pt
```

Cada archivo contendrá:

```text
tensor float16 o float32
shape = [n_frames, hidden_size]
```

El encoder wav2vec permanecerá:

```text
eval mode
requires_grad = False
```

La extracción se realiza una sola vez.

---

# 5. Experimento A — Poolings no entrenables

## 5.1 Mean pooling

Referencia actual:

```text
μ = mean_t(z_t)
```

Dimensión:

```text
768
```

## 5.2 Statistics pooling

Nueva representación:

```text
μ = mean_t(z_t)
σ = std_t(z_t)

z_global = concat(μ, σ)
```

Dimensión:

```text
1536
```

Pipeline:

```text
wav2vec last layer
→ mean + std
→ StandardScaler
→ Logistic Regression
```

## Pregunta experimental

> ¿La variación temporal de los embeddings contiene información emocional que se pierde al conservar únicamente la media?

## Comparación

```text
last layer + mean
vs
last layer + mean/std
```

## Criterio de interés

Statistics pooling se considera útil si:

```text
Δ macro F1 >= 0.01
```

o si:

```text
|Δ macro F1| < 0.01
y
reduce el desvío entre folds al menos 20%
```

Además, ningún outer fold debería caer más de:

```text
0.03 macro F1
```

respecto de la referencia.

---

# 6. Experimento B — Capas wav2vec

## Objetivo

Determinar si las capas intermedias contienen información emocional más útil o estable que la última capa.

Se evaluarán tres estrategias.

## 6.1 Última capa

```text
z_global = pool(layer_12)
```

Es la referencia actual.

## 6.2 Promedio uniforme de capas

Para cada audio:

```text
μ_mix = mean_l(μ_l)
```

Para statistics pooling:

```text
σ_mix = mean_l(σ_l)
```

Representaciones:

```text
average_layers + mean
average_layers + mean/std
```

No contiene parámetros entrenables.

## 6.3 Mezcla aprendida de capas

Se aprenderán pesos escalares:

```text
α = softmax(w)
```

y:

```text
μ_mix = Σ_l α_l μ_l
σ_mix = Σ_l α_l σ_l
```

La misma distribución de pesos se aplicará inicialmente a media y desvío.

Representación final:

```text
mean:
    z_global = μ_mix

mean/std:
    z_global = concat(μ_mix, σ_mix)
```

El modelo completo será:

```text
layer statistics congeladas
→ scalar layer mixture
→ cabeza lineal de clasificación
```

La cantidad de parámetros entrenables será mínima:

```text
n_layers pesos escalares
+
clasificador lineal
```

## Preguntas experimentales

```text
¿La última capa es la mejor para emoción?
¿Las capas intermedias aportan mayor estabilidad?
¿Los pesos aprendidos se concentran en un rango consistente de capas?
```

## Visualizaciones

### Rendimiento por estrategia

Errorbar plot:

```text
x: macro F1 medio
barra: ± desvío entre folds
y:
    last layer
    average layers
    learned mixture
```

Paneles:

```text
mean
mean/std
```

### Pesos aprendidos

Curva:

```text
x: índice de capa
y: peso softmax medio
banda: ± desvío entre folds y seeds
```

No se calcularán índices de correlación o estabilidad de rankings.

La interpretación será visual:

```text
capas tempranas
capas intermedias
capas profundas
```

---

# 7. Experimento C — Attentive statistics pooling

## Objetivo

Aprender qué frames contienen mayor evidencia emocional, en lugar de asignar el mismo peso temporal a toda la secuencia.

## Arquitectura

Entrada:

```text
z_t ∈ R^768
```

Cálculo de atención:

```text
h_t = tanh(W z_t + b)
e_t = vᵀ h_t
a_t = softmax(e_t)
```

Media ponderada:

```text
μ_att = Σ_t a_t z_t
```

Desvío ponderado:

```text
σ_att = sqrt(
    Σ_t a_t (z_t - μ_att)²
)
```

Representación:

```text
z_global = concat(μ_att, σ_att)
```

Clasificación:

```text
attentive statistics pooling
→ dropout
→ capa lineal
→ logits emocionales
```

## Configuración inicial

```yaml
attentive_pooling:
  input_dim: 768
  attention_hidden_dim: 128
  dropout: 0.10
  learning_rate: 0.001
  weight_decay: 0.0001
  max_epochs: 100
  patience: 10
  batch_size: 16
  seeds: [13, 42, 73]
```

No se implementará:

```text
multi-head attention
transformer adicional
CNN temporal
encoder wav2vec entrenable
```

## Early stopping

Por cada outer fold:

```text
outer train
├── inner train
└── inner validation
    └── early stopping

outer validation
└── evaluación final
```

La separación interna deberá respetar:

```text
speaker-independent → actor_id
speaker-dependent → utterance_group_id
```

Outer validation nunca participa en early stopping.

## Variación por seed

Para los modelos entrenables:

```text
3 seeds por outer fold
```

Las probabilidades de las tres semillas se promediarán para construir una única predicción OOF por muestra.

Se reportarán separadamente:

```text
desvío entre outer folds
desvío entre seeds dentro de cada fold
```

No se mezclarán ambas fuentes de variabilidad.

## Visualizaciones

### Comparación con poolings estáticos

Scatter:

```text
x: macro F1 medio
y: desvío entre folds
punto:
    mean
    mean/std
    attentive mean/std
```

### Curva por fold

```text
x: outer fold
y: macro F1
series:
    mean
    mean/std
    attentive statistics
```

### Matrices de confusión

```text
baseline mean
attentive statistics
diferencia attention − baseline
```

Normalizadas por clase verdadera.

### Atención temporal

Mostrar un máximo de tres ejemplos:

```text
audio corregido por atención
audio correctamente clasificado por ambos modelos
error persistente
```

Visualización:

```text
eje x: tiempo
curva 1: peso de atención
curva 2: RMS normalizado o energía
```

El objetivo no será interpretar causalmente cada frame, sino comprobar si la atención se concentra en regiones acústicamente activas o emocionalmente informativas.

---

# 8. Experimento D — Elastic Net sobre eGeMAPS

## Objetivo

Mantener una rama acústica interpretable y comprobar si una regularización estructurada produce coeficientes más consistentes que Logistic Regression L2.

Pipeline:

```text
StandardScaler
→ Logistic Regression Elastic Net
```

Configuración:

```text
solver = saga
penalty = elasticnet
class_weight = balanced
max_iter = 5000
```

Búsqueda interna pequeña:

```text
C ∈ {0.1, 1.0, 10.0}
l1_ratio ∈ {0.1, 0.5, 0.9}
```

La búsqueda se realiza exclusivamente dentro de outer train usando grouped inner CV.

## Comparaciones

```text
eGeMAPS + LogReg L2
eGeMAPS + LogReg Elastic Net
```

Targets principales:

```text
emotion_original
emotion_quadrant
```

## Interpretación por familias

Los coeficientes absolutos se agregarán usando las familias ya definidas:

```text
F0 y prosodia
loudness y energía
MFCC
formantes
espectro
calidad vocal
voicing y temporalidad
```

Se reportará:

```text
importancia media por familia
desvío entre folds
proporción de coeficientes no nulos
```

## Visualizaciones

### Rendimiento

Errorbar plot:

```text
L2 vs Elastic Net
macro F1 medio ± desvío
```

### Heatmap de familias

```text
filas: familias acústicas
columnas:
    L2
    Elastic Net
valor:
    importancia normalizada media
```

### Regularización

Barplot:

```text
familia
→ porcentaje de coeficientes no nulos
```

No se utilizará Elastic Net para eliminar manualmente familias completas.

Su función será:

```text
regularizar predictores correlacionados
estabilizar interpretación
mantener una referencia acústica experta
```

---

# 9. Matriz experimental

## Wav2vec

Configuraciones mínimas:

```text
1. last layer + mean
2. last layer + mean/std
3. average layers + mean
4. average layers + mean/std
5. learned layer mixture + mean
6. learned layer mixture + mean/std
7. last layer + attentive statistics pooling
```

No se combinará inicialmente:

```text
learned layer mixture
+
attentive statistics pooling
```

Esto evita mezclar selección de capa y selección temporal en un mismo experimento.

Solo se habilitará esa combinación si ambos componentes muestran evidencia favorable por separado.

## eGeMAPS

```text
1. Logistic Regression L2
2. Logistic Regression Elastic Net
```

## Targets

Prioridad:

```text
1. emotion_original
2. emotion_quadrant
3. emotion_original_eval_quadrant
```

El escenario 8→4 podrá ejecutarse únicamente para:

```text
baseline
mejor representación wav2vec
```

para evitar una matriz experimental redundante.

## Protocolos

```text
speaker-independent → principal
speaker-dependent → análisis secundario
```

---

# 10. Reutilización del código existente

Se reutilizarán:

```text
src/evaluation/metrics.py
src/evaluation/cross_validation.py
src/features/feature_store.py
src/features/extract_wav2vec.py
src/features/egemaps_families.py
src/models/linear_probe.py
src/experiments/baselines.py
folds persistidos
mappings de targets
```

No se creará un segundo sistema de métricas ni un segundo formato de resultados.

---

# 11. Módulos nuevos o extendidos

```text
src/features/wav2vec_temporal.py
src/models/layer_mixture.py
src/models/attentive_statistics_pooling.py
src/models/elastic_net.py
src/experiments/temporal_representation.py
```

## `wav2vec_temporal.py`

Responsabilidades:

```text
extraer hidden states congelados
calcular mean/std por capa
guardar estadísticas por capa
guardar secuencia de última capa
cargar artefactos por file_id
validar dimensión y alineación
```

## `layer_mixture.py`

Responsabilidades:

```text
average layers
scalar learned mixture
softmax de pesos
clasificador lineal
```

## `attentive_statistics_pooling.py`

Responsabilidades:

```text
máscara temporal
pesos de atención
media ponderada
desvío ponderado
clasificador
```

## `elastic_net.py`

Responsabilidades:

```text
construir pipeline
definir grilla pequeña
devolver estimador compatible con grouped inner CV
```

## `temporal_representation.py`

Responsabilidades:

```text
orquestar configuraciones
reutilizar folds
generar predicciones OOF
consolidar métricas
registrar seeds
```

---

# 12. Resultados persistidos

```text
reports/temporal_representation_results.csv
reports/temporal_oof_predictions.csv
reports/layer_mixture_weights.csv
reports/attention_seed_results.csv
reports/egemaps_elastic_net_results.csv
reports/egemaps_elastic_net_family_importance.csv
```

`temporal_representation_results.csv` deberá contener:

```text
representation
pooling
layer_strategy
protocol
target
model
fold
seed
macro_f1
balanced_accuracy
weighted_f1
accuracy
n_features
training_time
```

Para configuraciones deterministas:

```text
seed = -1
```

Para modelos entrenables:

```text
una fila por seed
+
una fila agregada por fold
```

---

# 13. Presentación visual del notebook

El notebook evitará tablas extensas.

## Figura 1 — Pooling estático

```text
mean vs mean/std
macro F1 medio ± desvío
```

## Figura 2 — Estrategia de capas

```text
last vs average vs learned mixture
```

## Figura 3 — Pesos por capa

```text
curva media ± desvío
```

## Figura 4 — Rendimiento y estabilidad

```text
scatter:
x = macro F1 medio
y = desvío entre folds
```

## Figura 5 — Curvas por fold

```text
baseline
best static pooling
layer mixture
attention
```

## Figura 6 — Matrices de confusión

```text
baseline
mejor modelo
diferencia
```

## Figura 7 — Atención temporal

```text
tres audios representativos
```

## Figura 8 — Elastic Net eGeMAPS

```text
L2 vs Elastic Net
+
heatmap de familias
```

## Figura final — Síntesis

Scatter global:

```text
x: macro F1 speaker-independent
y: desvío entre folds
tamaño: dimensionalidad
etiqueta: representación
```

Configuraciones incluidas:

```text
wav2vec mean
wav2vec mean/std
average layers
learned mixture
attentive statistics
eGeMAPS L2
eGeMAPS Elastic Net
```

---

# 14. Contrato anti-leakage

Los hidden states wav2vec pueden extraerse previamente porque:

```text
encoder congelado
sin targets
sin transformaciones aprendidas
```

Toda transformación aprendida deberá ajustarse dentro de outer train:

```text
StandardScaler
Elastic Net
layer mixture
attention pooling
early stopping
selección de hiperparámetros
```

El test final no se utiliza para:

```text
seleccionar pooling
seleccionar capas
elegir semillas
ajustar atención
ajustar Elastic Net
comparar configuraciones
```

---

# 15. Criterios de decisión

## Mejora de rendimiento

```text
Δ macro F1 medio >= 0.01
```

## Mejora de estabilidad

```text
|Δ macro F1| < 0.01
y
reducción de desvío entre folds >= 20%
```

## Protección del peor fold

```text
ningún fold cae más de 0.03
respecto del baseline wav2vec mean
```

## Modelos entrenables

Además:

```text
desvío medio entre seeds < 0.02
```

Si la atención mejora la media pero presenta alta variabilidad entre seeds, no se considerará una configuración estable.

---

# 16. Extensión opcional: speaker-adversarial pooling

No se implementará en este sprint.

Se habilitará únicamente si:

```text
attentive statistics mejora frente a mean
pero
la varianza sigue concentrada por actor
```

Arquitectura futura:

```text
attentive representation
├── cabeza de emoción
└── cabeza de actor
    └── gradient reversal
```

La hipótesis será:

> La representación temporal aprendida contiene evidencia emocional útil, pero todavía conserva información identitaria que limita la generalización speaker-independent.

Esta extensión deberá desarrollarse en un sprint independiente para no mezclar:

```text
selección temporal
invariancia de hablante
```

---

# 17. Condiciones de aprobación

```text
[ ] La referencia wav2vec mean se reproduce con los folds persistidos.
[ ] Mean/std pooling fue evaluado.
[ ] Last layer y average layers fueron comparados.
[ ] Learned layer mixture fue entrenado dentro de cada outer fold.
[ ] Los pesos de capas fueron visualizados.
[ ] Attentive statistics pooling fue evaluado con tres seeds.
[ ] Se separó variabilidad entre folds y entre seeds.
[ ] Se generaron predicciones OOF para las configuraciones principales.
[ ] Se generaron matrices de confusión normalizadas.
[ ] Elastic Net fue comparado contra LogReg L2.
[ ] La importancia Elastic Net fue agregada por familias eGeMAPS.
[ ] Ningún test final fue utilizado.
[ ] Se redactó una conclusión sobre pooling, capas y temporalidad.
[ ] Se decidió formalmente si speaker-adversarial queda justificado.
```

---

# 18. Criterio de cierre

El sprint deberá responder:

```text
¿Mean/std mejora frente a mean?
¿Las capas intermedias superan a la última capa?
¿La mezcla aprendida concentra peso en capas específicas?
¿La atención temporal mejora media o estabilidad?
¿La variabilidad entre seeds es aceptable?
¿Elastic Net estabiliza la interpretación de eGeMAPS?
¿La extensión speaker-adversarial queda metodológicamente justificada?
```

La salida principal no será necesariamente un modelo final nuevo.

El resultado esperado es identificar, con experimentos controlados, cuál de estas fuentes aporta evidencia real:

```text
variación temporal
nivel de representación wav2vec
selección temporal aprendida
regularización acústica experta
```
