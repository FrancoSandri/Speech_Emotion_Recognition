# Sprint 3 — Refinamientos de representación

## Alcance implementado

Se incorporaron tres análisis sobre la infraestructura común de outer cross-validation:

1. **Feature importance sobre eGeMAPSv02**
   - Random Forest impurity importance.
   - Permutation importance calculada sobre outer validation.
   - Coeficientes absolutos de Logistic Regression.
   - Ranking y frecuencia de aparición en top-k por outer fold.

2. **RFECV nested sobre eGeMAPSv02**
   - Estimador `StandardScaler → LogisticRegression`.
   - Inner `StratifiedGroupKFold` dentro de cada outer train.
   - Grupos por actor para speaker-independent.
   - Grupos por `utterance_group_id` para speaker-dependent.
   - Outer validation se usa solamente para evaluación.
   - Se registran subset, ranking, frecuencia de selección y métricas por fold.

3. **PCA dentro de outer CV**
   - Pipeline `StandardScaler → PCA → LogisticRegression`.
   - Umbrales de 90% y 95% de varianza explicada.
   - PCA se reajusta en cada outer train.
   - Se registra el número efectivo de componentes y la varianza explicada.

## Artefactos consolidados

```text
reports/cv_results.csv
reports/feature_selection.csv
```

Los refinamientos agregan o reemplazan filas por configuración y fold, evitando crear un archivo por experimento.

## Configuración práctica

RFECV usa por defecto:

```text
inner_folds = 3
step = 5
min_features_to_select = 10
n_jobs = 1
```

El `step=5` reduce el costo respecto de eliminar una sola feature por iteración y mantiene suficiente resolución para 88 variables eGeMAPS.

## Validación funcional realizada

Se ejecutaron smoke tests sobre los artefactos reales para comprobar:

- PCA sobre los cinco outer folds speaker-independent;
- importancia de features preservando los 88 nombres acústicos;
- RFECV con grupos de actores;
- RFECV speaker-dependent en el escenario 8→4;
- compatibilidad del motor de baselines después de extender `run_cv`.

Los smoke tests no se guardaron como resultados experimentales definitivos.

## Ejecución desde el notebook

Las corridas están separadas mediante flags:

```python
RUN_BASELINES = False
RUN_FEATURE_IMPORTANCE = False
RUN_RFECV = False
RUN_PCA = False
```

Esto permite ejecutar primero los baselines y después cada refinamiento sin recalcular automáticamente toda la matriz al abrir el notebook.

## Decisión sobre autoencoder

El autoencoder permanece diferido. Solo corresponde incorporarlo después de contar con resultados completos de baselines, RFECV y PCA, y si existe evidencia de que una reducción no lineal puede aportar valor suficiente para justificar mayor complejidad.
