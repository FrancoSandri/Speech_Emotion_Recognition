# Verificación técnica del snapshot actualizado

**Fecha:** 2026-06-23

## Alcance

Se revisaron configuración, código fuente, notebooks, metadata, splits y representaciones. Se priorizaron cambios mínimos útiles para Sprint 3. No se agregaron tests ni se modificaron el trimming o la extracción eGeMAPS.

## Estado de los artefactos

| Artefacto | Estado |
|---|---|
| `metadata.parquet` | 1440 IDs únicos, paths relativos, sexo consistente |
| `splits.parquet` | 1440 IDs, regenerado desde configuración |
| `egemaps.parquet` | 1440 × 88, sin NaN/Inf |
| `wav2vec.parquet` | 1440 × 768, sin NaN/Inf |
| Alineación por `file_id` | Exacta entre las cuatro tablas |

## Particiones regeneradas

| Partición | Muestras | Grupos | Actores |
|---|---:|---:|---:|
| development | 864 | 432 | 18 |
| test speaker-dependent | 216 | 108 | 18 conocidos |
| test speaker-independent | 360 | 180 | 6 no vistos |

Actores del test speaker-independent: `[2, 4, 6, 19, 21, 23]`, con balance 3 female / 3 male.

Comprobaciones:

- cero actores independent compartidos;
- cero `utterance_group_id` compartidos entre development y test dependent;
- ambas repeticiones permanecen en la misma partición;
- tests con fold `-1`;
- development con cinco folds completos;
- regeneración idéntica aun alterando el orden de las filas de metadata.

## Cambios mínimos aplicados

### `src/features/feature_store.py`

- joins uno-a-uno por `file_id`;
- error ante IDs faltantes, extras o duplicados;
- development como partición por defecto;
- validación numérica y de valores finitos;
- una única tabla alineada para los experimentos.

### `src/evaluation/metrics.py`

Métricas globales estandarizadas:

- macro F1;
- balanced accuracy;
- UAR;
- weighted F1;
- accuracy.

También se incorporó:

- resumen CV con media, desvío, mínimo y máximo;
- mapping emoción → cuadrante;
- evaluación formal 8→4.

### `src/experiments/cross_validation.py`

- usa únicamente folds persistidos;
- crea un estimador nuevo por fold;
- ajusta el pipeline solo sobre outer train;
- devuelve resultados normalizados por fold;
- soporta los tres targets, incluido 8→4;
- conserva métricas originales de ocho clases en el escenario 8→4.

### Splits y documentación

- `build_splits_dataframe()` propaga todos los parámetros del holdout agrupado;
- los splits ya no dependen del orden del DataFrame;
- se eliminaron referencias obsoletas a la estrategia `rep2`;
- se corrigieron comentarios sobre la paridad actor-sexo;
- se agregó un `setup.py` mínimo para que `pip install -e .` funcione;
- los extractores del notebook 02 cargan caché por defecto (`overwrite=False`).

## Validación funcional de Sprint 3

Se ejecutó un smoke test real con:

```text
eGeMAPS + StandardScaler + LogisticRegression
speaker-independent CV
```

Se validaron tanto clasificación directa de cuadrantes como 8→4. Las cinco corridas produjeron resultados completos, predicciones para las 864 muestras de development y todas las métricas esperadas. Esta ejecución fue únicamente una prueba técnica del pipeline y no constituye selección de modelo.

## Pendiente operativo

Los notebooks 01 y 02 deben volver a ejecutarse localmente para refrescar todos sus outputs y figuras con el `splits.parquet` regenerado. Las features no necesitan reextraerse: el notebook 02 ahora carga los Parquet existentes por defecto.
