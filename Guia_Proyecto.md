# Plan de ejecución en cuatro sprints

## Propósito

Construir desde cero un pipeline de Speech Emotion Recognition sobre RAVDESS que sea:

```text
interpretable
modular
reproducible
reutilizable
estadísticamente válido
resistente a data leakage
```

El proyecto comparará inicialmente:

```text
eGeMAPSv02
wav2vec frozen + mean pooling
```

sobre:

```text
emotion_original
emotion_quadrant
emotion_original_eval_quadrant
```

y bajo dos protocolos:

```text
speaker-dependent
speaker-independent
```

Los cuatro sprints son:

```text
Sprint 1 — Datos, audio, targets y particiones
Sprint 2 — Extracción de representaciones y EDA
Sprint 3 — Baselines y refinamiento dentro de cross-validation
Sprint 4 — Congelamiento de configuración y evaluación final
```

---

# Principios comunes a todos los sprints

## Código fuente

La lógica reutilizable debe residir en `src/`.

```text
src/
├── data/
├── datasets/
├── evaluation/
├── experiments/
├── features/
├── feature_selection/
├── models/
├── utils/
└── validation/
```

Los directorios `__pycache__` no forman parte de la arquitectura y deben estar excluidos mediante `.gitignore`.

---

## Notebooks

Se utilizará un notebook principal por sprint:

```text
notebooks/01_data_audio_splits.ipynb
notebooks/02_features_eda.ipynb
notebooks/03_baselines_refinement.ipynb
notebooks/04_final_evaluation.ipynb
```

Los notebooks deben:

```text
importar funciones de src
mostrar resultados
crear visualizaciones pertinentes
explicar decisiones
documentar conclusiones
```

No deben:

```text
reimplementar clases completas
contener funciones de entrenamiento duplicadas
hardcodear paths
reconstruir splits
modificar silenciosamente configuraciones
```

---

## Configuración

Se recomienda una única configuración principal:

```text
configs/config.yaml
```

Debe contener:

```text
paths
seed global
mapping emocional
parámetros de trimming
definición de test
cross-validation
extractores
modelos
métricas
refinamientos habilitados
```

Los valores efectivos deben cargarse desde esta configuración y no repetirse en notebooks.

---

## Contrato anti-leakage

Toda transformación aprendida debe seguir:

```text
definir particiones
↓
seleccionar train del fold
↓
fit exclusivamente sobre train
↓
transform validation o test
```

Aplica a:

```text
StandardScaler
imputación
RFECV
PCA
autoencoder
selección por importancia
calibración probabilística
attention pooling
optimización de hiperparámetros
```

El test final no participa en:

```text
EDA supervisado
selección de representación
selección de features
elección de hiperparámetros
early stopping
comparación de alternativas
```

---

## Criterio general de aprobación

Cada sprint debe cerrar como:

```text
PASS
PASS WITH WARNINGS
FAIL
```

Un sprint queda en `FAIL` si incumple una condición bloqueante. No debe iniciarse el siguiente sprint hasta resolverla.

---

# Sprint 1 — Datos, audio, targets y particiones

## 1. Objetivo

Construir la base inmutable del proyecto:

```text
metadata
targets
audio recortado
test finales
conjunto de desarrollo
folds de cross-validation
validaciones de leakage
```

Este sprint debe garantizar que todos los experimentos posteriores utilicen exactamente los mismos archivos y particiones.

---

## 2. Preguntas que debe responder

```text
¿Todos los audios fueron descubiertos correctamente?
¿La metadata coincide con la codificación RAVDESS?
¿Los targets están completos y bien mapeados?
¿El trimming elimina solo silencios extremos?
¿Las repeticiones casi equivalentes permanecen juntas?
¿Los actores del test independent están realmente aislados?
¿Los test finales permanecen fuera de todo desarrollo?
```

---

## 3. Diseño de particiones

Para mantener válidos simultáneamente ambos protocolos, se construirán tres particiones globales y disjuntas.

### 3.1 Test speaker-independent

Contiene todos los audios de un conjunto fijo de actores.

Condiciones:

```text
los actores no aparecen en desarrollo
los actores no aparecen en test dependent
los actores no participan en EDA supervisado
```

Recomendación inicial:

```text
6 actores
3 masculinos
3 femeninos
```

La selección debe ser determinística y quedar almacenada en configuración y manifest.

---

### 3.2 Test speaker-dependent

Contiene grupos de utterances pertenecientes a los actores de desarrollo.

Cada grupo se define como:

```text
utterance_group_id =
actor_id + emotion_original + intensity + statement
```

Las dos repeticiones de la misma combinación deben permanecer juntas.

El test dependent debe:

```text
contener todos o prácticamente todos los actores de desarrollo
preservar representación de las emociones
preservar ambos sexos
no compartir utterance_group_id con desarrollo
```

---

### 3.3 Development pool

Contiene todos los archivos que no pertenecen a ninguno de los test finales.

Será el único conjunto utilizado para:

```text
EDA supervisado
cross-validation
selección de features
ajuste de hiperparámetros
selección de representación
```

De esta manera:

```text
development_pool
∩ test_speaker_dependent
= vacío

development_pool
∩ test_speaker_independent
= vacío

test_speaker_dependent
∩ test_speaker_independent
= vacío
```

---

## 4. Cross-validation

Los folds deben calcularse una sola vez y persistirse.

### 4.1 Speaker-dependent CV

```text
dataset = development_pool
groups = utterance_group_id
splitter = StratifiedGroupKFold
```

Las repeticiones equivalentes nunca pueden separarse entre train y validation.

### 4.2 Speaker-independent CV

```text
dataset = development_pool
groups = actor_id
splitter = StratifiedGroupKFold
```

Si no fuera viable:

```text
GroupKFold
```

La degradación a `GroupKFold` debe quedar documentada y acompañada por auditoría de balance.

### 4.3 Reutilización

Los mismos folds se reutilizarán para:

```text
eGeMAPSv02
wav2vec
RF
MLP
LogReg
emotion_original
emotion_quadrant
emotion_original_eval_quadrant
RFECV
PCA
autoencoder
```

No se reconstruyen folds por modelo o representación.

---

## 5. Limpieza del audio

Aplicar trimming únicamente sobre los extremos.

Procedimiento:

```text
cargar audio
convertir a mono cuando corresponda
calcular RMS por frames
detectar primera y última región activa
agregar padding
guardar audio sin modificar amplitud
```

Parámetros iniciales:

```text
frame_length = 25 ms
hop_length = 10 ms
threshold_db = entre -30 y -40 dB respecto del máximo
padding = entre 100 y 200 ms
```

Los parámetros se fijan inspeccionando únicamente muestras del `development_pool`.

No aplicar:

```text
RMS normalization
LUFS normalization
peak normalization agresiva
compresión
noise reduction
eliminación de silencios internos
```

---

## 6. Módulos requeridos

```text
src/data/metadata.py
src/data/audio_cleaning.py
src/data/splits.py

src/validation/schemas.py
src/validation/sanity.py
src/validation/leakage.py

src/config/contracts.py

src/utils/config.py
src/utils/io.py
src/utils/logging.py
src/utils/reproducibility.py
```

### Responsabilidades

#### `src/data/metadata.py`

```text
descubrir archivos
parsear nombres RAVDESS
construir file_id
crear targets
crear utterance_group_id
```

#### `src/data/audio_cleaning.py`

```text
detectar región activa
aplicar padding
guardar audio recortado
devolver métricas de trimming
```

#### `src/data/splits.py`

```text
crear test independent
crear test dependent
crear development_pool
crear folds dependent
crear folds independent
```

#### `src/validation/leakage.py`

```text
validar separación de actores
validar separación de utterance_group_id
validar ausencia de file overlap
validar que test no aparece en folds
```

#### `src/config/contracts.py`

Debe definir únicamente contratos compartidos:

```text
columnas requeridas
nombres oficiales de targets
nombres oficiales de protocolos
paths oficiales
mapping emoción → cuadrante
```

No debe contener lógica de procesamiento.

---

## 7. Notebook

```text
notebooks/01_data_audio_splits.ipynb
```

Secciones:

```text
1. Configuración y seed
2. Descubrimiento de audios
3. Construcción de metadata
4. Validaciones RAVDESS
5. Creación de targets
6. Definición de test finales
7. Construcción de folds
8. Ejecución del trimming
9. Auditoría del trimming
10. Auditoría de leakage
11. Conclusiones del sprint
```

---

## 8. Artefactos permitidos

```text
data/processed/metadata.parquet
data/processed/splits.parquet
data/processed/audio_trimmed/Actor_x/*.wav

reports/data_quality.md
reports/figures/trim_duration_comparison.png
reports/figures/split_class_distribution.png
```

`metadata.parquet` debe contener las métricas de trimming; no crear un CSV adicional por audio.

`data/processed/splits.parquet` debe contener, como mínimo:

```text
file_id
partition
fold_speaker_dependent
fold_speaker_independent
```

Valores posibles de `partition`:

```text
development
test_speaker_dependent
test_speaker_independent
```

---

## 9. Condiciones de aprobación

### 9.1 Metadata — bloqueante

Debe cumplirse:

```text
file_id único
file_path_raw existente
file_path_trimmed existente
actor_id válido
emotion_original válida
emotion_quadrant no nulo
sex consistente con actor
sin archivos duplicados
sin archivos sin parsear
```

Cualquier archivo no parseado implica `FAIL`, salvo que sea excluido explícitamente con justificación.

---

### 9.2 Audio — bloqueante

Debe cumplirse:

```text
todos los audios procesados pueden abrirse
ningún audio queda vacío
ningún audio cambia de amplitud por normalización
sample rate documentado
trim_start < trim_end
duration_trimmed > 0
```

Los clips con trimming excesivo deben revisarse.

Umbral inicial de advertencia:

```text
trim_ratio > 0.50
```

No implica necesariamente error, pero requiere inspección.

---

### 9.3 Splits — bloqueante

Debe cumplirse:

```text
cero file overlap entre particiones
cero actores compartidos entre development y test independent
cero utterance_group_id compartidos entre development y test dependent
cero archivos de test presentes en folds
todos los folds contienen todas las clases cuando sea matemáticamente posible
```

---

### 9.4 Reproducibilidad — bloqueante

Ejecutar dos veces con la misma configuración debe producir:

```text
mismos file_id
mismas particiones
mismos folds
mismos límites de trimming
```

---

### 9.5 Interpretabilidad del código — bloqueante

No se aprueba si:

```text
el notebook implementa el parser completo
el notebook reconstruye splits manualmente
los paths están repetidos en múltiples celdas
una función mezcla metadata, trimming y splitting
existen scripts duplicados por target
```

---

## 10. Criterio de Done

```text
[ ] Metadata construida y validada.
[ ] Targets original y quadrant creados.
[ ] Test speaker-independent congelado.
[ ] Test speaker-dependent congelado.
[ ] Development pool definido.
[ ] Folds dependent persistidos.
[ ] Folds independent persistidos.
[ ] Audio recortado sin normalización de amplitud.
[ ] Auditoría de trimming completada.
[ ] Auditoría de leakage en PASS.
[ ] Notebook ejecutable de principio a fin.
[ ] data_quality.md redactado.
```

---

# Sprint 2 — Extracción de representaciones y EDA

## 1. Objetivo

Extraer representaciones acústicas alineadas a partir del audio recortado y realizar un EDA limitado, interpretable y libre de contaminación del test.

Representaciones obligatorias:

```text
eGeMAPSv02
wav2vec frozen + mean pooling
```

---

## 2. Preguntas que debe responder

```text
¿Las features fueron extraídas para todos los audios?
¿Las dos representaciones están alineadas por file_id?
¿Existen features constantes, NaN o valores infinitos?
¿Qué estructura muestran las features dentro de development?
¿La geometría está dominada por emoción, actor o ambos?
¿Qué efecto tuvo el trimming sobre la representación?
```

---

## 3. Extracción eGeMAPSv02

Debe utilizarse:

```text
eGeMAPSv02
Functionals
```

Salida:

```text
una fila por file_id
88 columnas acústicas
```

Los nombres originales de OpenSMILE deben preservarse.

No renombrar columnas a etiquetas genéricas como:

```text
feature_1
feature_2
```

porque se perdería interpretabilidad acústica.

---

## 4. Extracción wav2vec

En Stage 1, wav2vec se utiliza como extractor congelado.

Procedimiento:

```text
audio trimmed
↓
mono
↓
resampling a 16 kHz
↓
processor
↓
wav2vec en eval()
↓
hidden states z_t
↓
mean pooling temporal
↓
embedding global
```

Debe quedar documentado:

```text
modelo exacto
versión o revision
layer utilizado
dimensión del embedding
sample rate
método de pooling
```

Todos los parámetros del modelo deben tener:

```text
requires_grad = False
```

No existe fine-tuning en este sprint.

---

## 5. EDA permitido

El EDA supervisado se realiza únicamente sobre:

```text
partition == development
```

No deben utilizarse los dos test finales para:

```text
PCA coloreado por clases
selección de features
comparaciones por actor
inspección de separabilidad
decisiones de preprocesamiento
```

Sobre test solo se permiten:

```text
conteos
schema
dimensionalidad
NaN
infinitos
alineación de file_id
```

---

## 6. EDA de dataset

Analizar en development:

```text
balance por emotion_original
balance por emotion_quadrant
cantidad por actor
cantidad por sexo
cantidad por intensity
cantidad por statement
duración raw vs trimmed
```

Auditar también cada fold:

```text
número de muestras
actores
clases
sexo
intensidad
statement
```

---

## 7. EDA de eGeMAPSv02

Analizar:

```text
NaN
infinitos
features constantes
varianza
outliers
correlaciones altas
distribuciones por clase
distribuciones por actor
```

Visualizaciones limitadas:

```text
un heatmap de correlación resumido
PCA por emoción
PCA por actor
boxplots de familias acústicas representativas
```

Familias sugeridas:

```text
F0
loudness
MFCC
formantes
jitter/shimmer
HNR
spectral
```

No crear automáticamente 88 histogramas o boxplots.

---

## 8. EDA de wav2vec

El embedding se interpreta geométricamente, no dimensión por dimensión.

Analizar:

```text
PCA
varianza explicada
distancias intra-clase
distancias inter-clase
distribución por actor
distribución por emoción
```

No asignar significado acústico directo a una dimensión latente sin evidencia adicional.

---

## 9. Módulos requeridos

```text
src/datasets/audio.py
src/datasets/tabular.py

src/features/egemaps.py
src/features/wav2vec.py
src/features/feature_store.py

src/validation/schemas.py
src/validation/sanity.py
```

### Responsabilidades

#### `src/datasets/audio.py`

```text
cargar waveform
convertir mono
resamplear
devolver waveform y sample rate
```

#### `src/datasets/tabular.py`

```text
unir metadata y features
seleccionar feature columns
filtrar por partition y fold
```

#### `src/features/egemaps.py`

```text
configurar OpenSMILE
extraer una muestra
extraer un batch
devolver DataFrame indexado por file_id
```

#### `src/features/wav2vec.py`

```text
cargar extractor congelado
extraer secuencia
aplicar mean pooling
procesar batch
```

#### `src/features/feature_store.py`

```text
guardar representación
cargar representación
validar schema
validar correspondencia con metadata
```

---

## 10. Notebook

```text
notebooks/02_features_eda.ipynb
```

Secciones:

```text
1. Carga del contrato de datos
2. Validación del audio trimmed
3. Extracción o carga de eGeMAPSv02
4. Extracción o carga de wav2vec
5. Validación de alineación
6. EDA del dataset
7. EDA eGeMAPSv02
8. EDA wav2vec
9. Comparación conceptual de representaciones
10. Conclusiones del sprint
```

La extracción puede ejecutarse desde el notebook, pero la implementación debe residir en `src/features/`.

---

## 11. Artefactos permitidos

```text
data/features/egemaps_v02.parquet
data/features/wav2vec_mean.parquet

reports/eda_summary.md
reports/figures/eda_class_balance.png
reports/figures/egemaps_pca_emotion.png
reports/figures/egemaps_pca_actor.png
reports/figures/wav2vec_pca_emotion.png
reports/figures/wav2vec_pca_actor.png
reports/figures/egemaps_correlation.png
```

No guardar:

```text
un parquet por fold
un parquet por target
una copia escalada de cada representación
una imagen por feature
```

El escalado se realiza dinámicamente dentro de pipelines posteriores.

---

## 12. Condiciones de aprobación

### 12.1 Integridad — bloqueante

Para cada representación:

```text
un único registro por file_id
mismo conjunto de file_id que metadata
sin filas duplicadas
dimensión constante
sin columnas completamente nulas
```

---

### 12.2 eGeMAPSv02 — bloqueante

Debe cumplirse:

```text
88 features acústicas
nombres originales preservados
sin infinitos
NaN tratados o justificados
extracción desde file_path_trimmed
```

Si una feature es constante, debe documentarse. No eliminarla todavía de manera global; la eliminación debe realizarse dentro del pipeline cuando corresponda.

---

### 12.3 wav2vec — bloqueante

Debe cumplirse:

```text
modelo en eval mode
gradientes deshabilitados
sample rate correcto
pooling documentado
misma layer para todas las muestras
embedding dimension consistente
```

---

### 12.4 Test isolation — bloqueante

Debe comprobarse que:

```text
ningún gráfico supervisado usa test
ningún PCA supervisado se ajusta con test
ninguna estadística por clase incluye test
```

---

### 12.5 Interpretabilidad — bloqueante

No se aprueba si:

```text
la extracción depende de código copiado en el notebook
eGeMAPS y wav2vec usan listas distintas de audios
los joins se realizan por posición en lugar de file_id
se guardan versiones escaladas globalmente
```

---

## 13. Criterio de Done

```text
[ ] eGeMAPSv02 extraído desde audio trimmed.
[ ] wav2vec mean extraído desde audio trimmed.
[ ] Representaciones alineadas por file_id.
[ ] Schemas validados.
[ ] EDA realizado solo sobre development.
[ ] Folds auditados.
[ ] Figuras principales generadas.
[ ] EDA resumido en eda_summary.md.
[ ] Notebook ejecutable sin lógica duplicada.
```

---

# Sprint 3 — Baselines y refinamiento dentro de cross-validation

## 1. Objetivo

Entrenar y comparar baselines reproducibles, y evaluar técnicas de refinamiento sin utilizar los test finales.

Este sprint debe responder qué combinación de:

```text
representación
target
modelo
protocolo
refinamiento
```

presenta la mejor relación entre:

```text
generalización
estabilidad
interpretabilidad
simplicidad
```

---

## 2. Matriz experimental mínima

### Representaciones

```text
egemaps_v02
wav2vec_mean
```

La concatenación queda como experimento posterior, solo si las dos ramas individuales están estabilizadas.

### Targets

```text
emotion_original
emotion_quadrant
emotion_original_eval_quadrant
```

### Protocolos

```text
speaker_dependent
speaker_independent
```

### Modelos

```text
Random Forest
MLP shallow
Logistic Regression recomendado
```

---

## 3. Implementación de baselines

### Random Forest

Debe usar features sin StandardScaler.

Se recomienda una configuración inicial fija y moderada. No hacer un grid search masivo.

### Logistic Regression

Debe usar:

```text
StandardScaler
↓
LogisticRegression
```

dentro de un único `Pipeline`.

### MLP

Debe usar:

```text
StandardScaler
↓
MLP
```

El scaler se ajusta exclusivamente con train del fold.

El MLP debe ser shallow:

```text
una o dos capas ocultas
dropout o regularización moderada
early stopping
seed controlada
```

La partición utilizada para early stopping debe salir de train del fold, nunca de outer validation.

---

## 4. Evaluación 8→4

Para `emotion_original_eval_quadrant`:

```text
entrenar con emotion_original
predecir una de las 8 emociones
mapear la predicción a cuadrante
evaluar contra emotion_quadrant
```

Se deben conservar, como mínimo:

```text
métricas originales de 8 clases
métricas colapsadas de 4 cuadrantes
```

No entrenar directamente en cuadrantes y presentarlo como 8→4.

---

## 5. Cross-validation

Para cada protocolo se usarán exclusivamente los folds persistidos en Sprint 1.

Por fold:

```text
train
validation
```

Nunca reconstruir:

```text
train_test_split
KFold
GroupKFold
```

dentro de cada modelo.

Outputs por fold:

```text
macro_F1
balanced_accuracy
UAR
weighted_F1
accuracy
n_features
tiempo de entrenamiento opcional
```

Reporte agregado:

```text
media
desvío estándar
mínimo
máximo
```

Métrica primaria:

```text
macro_F1 speaker-independent
```

---

## 6. Feature importance

Aplicar prioritariamente sobre eGeMAPSv02.

Métodos:

```text
RF impurity importance
permutation importance sobre outer validation
coeficientes absolutos de LogReg
```

La importancia debe calcularse dentro de cada fold.

Reportar:

```text
ranking promedio
desvío de ranking
frecuencia en top-k
```

No calcular un ranking global sobre todo development antes de cross-validation para luego evaluar con los mismos datos.

---

## 7. RFECV

RFECV debe aplicarse principalmente a eGeMAPSv02.

Diseño:

```text
outer fold
├── outer train
│   └── inner grouped CV
│       └── RFECV
└── outer validation
    └── evaluación del subset seleccionado
```

Grupos internos:

```text
speaker-independent → actor_id
speaker-dependent → utterance_group_id
```

Estimador recomendado:

```text
Logistic Regression regularizada
```

Scoring recomendado:

```text
f1_macro
```

Debe reportarse:

```text
cantidad de features seleccionadas por outer fold
features seleccionadas por fold
selection_frequency
resultado contra all_features
```

RFECV nunca se ajusta sobre outer validation.

---

## 8. PCA

PCA debe implementarse dentro del pipeline:

```text
StandardScaler
↓
PCA
↓
modelo
```

Evaluar como mínimo:

```text
90% de varianza explicada
95% de varianza explicada
```

El PCA se ajusta únicamente con train del fold.

Debe reportarse:

```text
número de componentes
varianza explicada
performance
comparación con representación original
```

---

## 9. Autoencoder opcional

Solo se habilita si:

```text
los baselines están completos
RFECV está completo
PCA está completo
el pipeline está libre de leakage
```

Por outer fold:

```text
outer train
├── inner train
└── inner validation para early stopping

outer validation
└── evaluación final del fold
```

El autoencoder nunca usa outer validation para early stopping.

Debe ser pequeño y regularizado.

No se aprueba un autoencoder que mejore solo speaker-dependent y no sea analizado en speaker-independent.

---

## 10. Selección de configuración

La configuración no se elige por el mejor fold.

Criterio principal:

```text
macro_F1 medio speaker-independent
```

Desempates:

```text
menor desvío
menor complejidad
menor cantidad de features
mayor interpretabilidad
mayor estabilidad de selección
```

Debe predefinirse una tolerancia práctica. Por ejemplo:

```text
si dos configuraciones difieren menos de 0.01 macro_F1,
preferir la más simple
```

La tolerancia exacta debe quedar en configuración.

---

## 11. Módulos requeridos

```text
src/models/random_forest.py
src/models/mlp.py
src/models/linear_probe.py
src/models/autoencoder.py

src/evaluation/metrics.py
src/evaluation/cross_validation.py
src/evaluation/reporting.py

src/feature_selection/importance.py
src/feature_selection/rfecv.py
src/feature_selection/pca.py

src/experiments/baselines.py
src/experiments/representation_refinement.py

src/validation/leakage.py
```

### Responsabilidades

#### `src/models/*`

Deben construir estimadores. No deben leer archivos ni guardar reportes.

#### `src/evaluation/cross_validation.py`

Debe:

```text
recibir folds persistidos
ejecutar train/validation
calcular métricas
devolver resultados normalizados
```

#### `src/experiments/baselines.py`

Debe orquestar:

```text
representaciones
targets
modelos
protocolos
```

sin duplicar la implementación del modelo.

#### `src/feature_selection/rfecv.py`

Debe encapsular:

```text
inner grouped CV
RFECV
feature names
ranking
selection frequency
```

---

## 12. Notebook

```text
notebooks/03_baselines_refinement.ipynb
```

Secciones:

```text
1. Carga de representaciones y folds
2. Validación previa
3. Baselines eGeMAPSv02
4. Baselines wav2vec
5. Comparación de targets
6. Comparación de protocolos
7. Feature importance
8. RFECV
9. PCA
10. Autoencoder opcional
11. Selección de configuración
12. Conclusiones
```

El notebook no debe ejecutar automáticamente todos los experimentos costosos cada vez que se abre. Debe permitir cargar resultados consolidados ya calculados.

---

## 13. Artefactos permitidos

```text
reports/cv_results.csv
reports/feature_selection.csv
reports/model_selection.md

reports/figures/cv_model_comparison.png
reports/figures/protocol_comparison.png
reports/figures/feature_selection_stability.png
```

`cv_results.csv` debe concentrar todos los folds:

```text
representation
protocol
target
model
refinement
fold
n_features
macro_f1
balanced_accuracy
uar
weighted_f1
accuracy
```

`feature_selection.csv` debe concentrar:

```text
protocol
target
model
fold
feature
selected
rank
selection_frequency
```

No guardar:

```text
un directorio por fold
un modelo por fold
un CSV por experimento
un JSON por métrica
predicciones de todos los folds salvo necesidad analítica explícita
```

---

## 14. Condiciones de aprobación

### 14.1 Folds — bloqueante

Debe comprobarse que todas las configuraciones comparables utilizan exactamente los mismos folds.

---

### 14.2 Fit scope — bloqueante

Debe existir evidencia programática de que:

```text
scaler.fit usa outer train
PCA.fit usa outer train
RFECV.fit usa outer train con inner CV
autoencoder.fit usa outer train
outer validation solo se transforma y evalúa
```

---

### 14.3 Test isolation — bloqueante

Los módulos de este sprint no deben cargar archivos con:

```text
partition != development
```

Se recomienda que el dataset loader rechace test por defecto.

---

### 14.4 Resultados — bloqueante

Debe existir al menos una corrida completa para:

```text
2 representaciones
2 protocolos
3 targets
RF
MLP
```

LogReg es recomendado y necesario para RFECV, salvo deuda técnica documentada.

---

### 14.5 Interpretabilidad — bloqueante

No se aprueba si:

```text
cada modelo implementa su propio loop de CV
las métricas se calculan de manera distinta por modelo
RFECV se ejecuta antes del outer split
PCA se ajusta globalmente
el notebook contiene una segunda implementación del entrenamiento
```

---

### 14.6 Selección final — bloqueante

`model_selection.md` debe registrar, antes de abrir test:

```text
configuración primaria
configuración baseline de referencia
regla utilizada para seleccionarlas
métricas CV
riesgos observados
```

---

## 15. Criterio de Done

```text
[ ] RF evaluado en ambas representaciones.
[ ] MLP evaluado en ambas representaciones.
[ ] Ambos protocolos evaluados.
[ ] Tres condiciones de target evaluadas.
[ ] Resultados por fold consolidados.
[ ] Feature importance agregada por fold.
[ ] RFECV nested ejecutado.
[ ] PCA dentro de folds ejecutado.
[ ] Autoencoder ejecutado o diferido explícitamente.
[ ] Configuración final congelada.
[ ] Ningún test final fue abierto.
[ ] model_selection.md aprobado.
```

---

# Sprint 4 — Congelamiento de configuración y evaluación final

## 1. Objetivo

Entrenar la configuración seleccionada sobre todo el `development_pool` y evaluarla una única vez sobre los test finales.

Este sprint transforma la evidencia de cross-validation en una conclusión final del proyecto.

---

## 2. Condición previa

Antes de cargar cualquier test debe existir:

```text
reports/model_selection.md
```

con una configuración inmutable:

```text
representación
target
modelo
hiperparámetros
preprocesamiento
feature selection o reducción
seed
```

También debe registrarse un hash o snapshot de la configuración.

---

## 3. Configuraciones finales

Se recomienda fijar:

### Configuración primaria

Seleccionada por:

```text
macro_F1 speaker-independent
estabilidad
simplicidad
interpretabilidad
```

Esta será la base de la conclusión principal.

### Baseline de referencia

Una configuración simple, por ejemplo:

```text
eGeMAPSv02 + RF
```

o la configuración baseline que haya sido predefinida en Sprint 3.

Su función es comprobar si el refinamiento realmente agrega valor.

---

## 4. Entrenamiento final

La configuración se reentrena usando todo el `development_pool`.

Las transformaciones deben ajustarse nuevamente sobre development:

```text
scaler
RFECV o selector final
PCA
autoencoder
clasificador
```

Cuando RFECV resulte ganador:

```text
usar la metodología RFECV seleccionada
ajustarla nuevamente sobre development
usar grouped CV interno
obtener el subset final
reentrenar el clasificador
```

No seleccionar manualmente features luego de observar test.

---

## 5. Evaluación speaker-independent

Evaluar sobre:

```text
test_speaker_independent
```

Este resultado representa la generalización a actores no vistos.

Debe considerarse la evidencia principal del Proyecto.

---

## 6. Evaluación speaker-dependent

Evaluar sobre:

```text
test_speaker_dependent
```

Representa un escenario de actores conocidos con utterances no vistas.

Debe utilizar:

```text
la misma configuración
el mismo target
la misma representación
```

que la evaluación independent cuando se quiera atribuir la diferencia al protocolo.

Pueden informarse configuraciones optimizadas por protocolo como análisis secundario, pero no deben sustituir esta comparación controlada.

---

## 7. Extensión temporal opcional

Attention pooling solo puede evaluarse antes de abrir los test.

Si se implementa, el orden dentro del Sprint 4 será:

```text
1. desarrollar attention pooling usando solo development
2. compararlo mediante los folds de Sprint 3
3. decidir si reemplaza mean pooling
4. congelar configuración
5. abrir test
```

Si los test ya fueron abiertos, attention pooling debe diferirse a otro stage.

Nunca usar test para decidir:

```text
número de heads
dimensión de attention
dropout
learning rate
número de epochs
```

---

## 8. Modelo final

Guardar únicamente el modelo primario final:

```text
models/primary_model.joblib
```

o, para PyTorch:

```text
models/primary_model.pt
```

Debe incluirse o acompañarse por:

```text
preprocesamiento
orden de features
mapping de clases
configuración efectiva
```

No guardar modelos de cada fold.

---

## 9. Módulos requeridos

```text
src/experiments/final_evaluation.py

src/evaluation/predictions.py
src/evaluation/reporting.py

src/models/attention_pooling.py  # opcional

src/pipeline.py
src/validation/leakage.py
```

### `src/experiments/final_evaluation.py`

Debe:

```text
cargar configuración congelada
cargar development
ajustar pipeline final
evaluar test una vez
guardar modelo primario
devolver métricas y predicciones
```

### `src/pipeline.py`

Debe orquestar los bloques, pero no reimplementar:

```text
feature extraction
cross-validation
modelos
métricas
```

---

## 10. Notebook

```text
notebooks/04_final_evaluation.ipynb
```

Secciones:

```text
1. Verificación de configuración congelada
2. Verificación de test isolation
3. Reentrenamiento sobre development
4. Evaluación speaker-independent
5. Evaluación speaker-dependent
6. Comparación CV vs test
7. Confusion matrices
8. Recall por clase
9. Análisis de errores
10. Interpretación acústica y metodológica
11. Conclusiones del Proyecto
```

El test no debe abrirse en celdas exploratorias anteriores a la evaluación final.

---

## 11. Artefactos permitidos

```text
models/primary_model.joblib
o
models/primary_model.pt

reports/test_metrics.json
reports/test_predictions.csv
reports/classification_report.md
reports/final_summary.md

reports/figures/confusion_matrix_independent.png
reports/figures/confusion_matrix_dependent.png
reports/figures/cv_vs_test.png
```

`test_predictions.csv` debe contener como mínimo:

```text
file_id
protocol
target_true
target_pred
correct
```

Probabilidades por clase pueden agregarse solo si son necesarias para análisis.

---

## 12. Condiciones de aprobación

### 12.1 Configuración congelada — bloqueante

Debe verificarse que `model_selection.md` sea anterior a la generación de métricas de test.

No se permite editar hiperparámetros luego de observar test.

---

### 12.2 Entrenamiento final — bloqueante

Debe cumplirse:

```text
fit únicamente sobre development
test usado solo en predict/evaluate
preprocesamiento incluido en pipeline
orden de features preservado
```

---

### 12.3 Predicciones — bloqueante

Debe existir exactamente una predicción por muestra del test correspondiente.

```text
sin file_id duplicados
sin file_id ausentes
sin clases desconocidas
```

---

### 12.4 Comparación CV-test — bloqueante

El reporte debe contrastar:

```text
CV mean ± std
test score
```

Una caída fuerte debe documentarse y analizarse. No habilita automáticamente una nueva ronda de tuning.

---

### 12.5 Repetición del test — bloqueante

El test no debe ejecutarse repetidamente para seleccionar modelos.

Una segunda ejecución solo es válida por:

```text
bug técnico demostrado
error de schema
corrupción de artefactos
```

En ese caso debe invalidarse formalmente la corrida anterior y documentarse.

---

### 12.6 Interpretación — bloqueante

`final_summary.md` debe responder:

```text
¿eGeMAPSv02 o wav2vec generaliza mejor?
¿Qué diferencia existe entre dependent e independent?
¿El target de cuadrantes mejora la robustez?
¿8→4 supera el entrenamiento directo en cuadrantes?
¿RFECV redujo dimensionalidad sin perder performance?
¿PCA o autoencoder aportaron evidencia útil?
¿Qué clases concentran los errores?
¿Existe justificación para attention pooling?
```

---

## 13. Criterio de Done

```text
[ ] Configuración final congelada antes de abrir test.
[ ] Pipeline reentrenado sobre development.
[ ] Test speaker-independent evaluado una vez.
[ ] Test speaker-dependent evaluado una vez.
[ ] CV y test comparados.
[ ] Predicciones finales guardadas.
[ ] Confusion matrices generadas.
[ ] Modelo primario guardado en models/.
[ ] Análisis de errores redactado.
[ ] Limitaciones documentadas.
[ ] Conclusión final del Proyecto redactada.
```

---

# Matriz de dependencias

```text
Sprint 1
Datos y splits válidos
        ↓
Sprint 2
Representaciones válidas
        ↓
Sprint 3
Baselines y refinamiento válidos
        ↓
Sprint 4
Test final y cierre
```

No se permite:

```text
extraer resultados antes de congelar splits
entrenar antes de validar features
hacer RFECV antes de definir folds
abrir test antes de congelar configuración
implementar attention pooling después de observar test
```

---

# Criterio global de aprobación del Proyecto

Proyecto queda aprobada cuando:

```text
los datos son trazables
los dos test finales permanecieron aislados
los folds son reproducibles
las representaciones están alineadas
los modelos usan pipelines sin leakage
RFECV/PCA/AE fueron evaluados dentro de folds
la configuración final fue congelada antes de test
el test se utilizó una sola vez
el modelo final y su preprocesamiento son reproducibles
las conclusiones diferencian performance de generalización
```
