## 0. Radiografía rápida de lo que *ya* tiene Roxy

Mirando el zip:

* Estructura de paquete clara: `core/`, `descriptors/`, `features/`, `eda/`, `projection/`, `viz/`, `report/`, `cli/`.
* `RoxyDataset`, motores de descriptores (seq/struct/mol), AAIndex, EDA (summary/target_relations/missing/basic), scaling/selection, reducers (PCA/t-SNE/UMAP), visualización (matplotlib + plotly), reportes (Markdown/HTML).
* `pyproject.toml` mínimo funcional, licencia y README stub.
* Notebooks de demo que ya generamos (en `simple_demos/` supongo que los vas a mover).

O sea: la “motor central” está. Lo que falta es la capa de *ecosistema completo* que Sylphy sí tiene (o al menos la estamos empujando hacia eso).

---

## 1. API pública y “cara” del paquete

### Qué veo ahora

* `roxy/__init__.py` está vacío: `"""Roxy roxy package."""`
* No hay una API “oficial” tipo `from roxy import RoxyDataset, describe_sequences, build_report, project`.

### Qué agregaría

1. **API de alto nivel en `roxy.__init__`**:

   * Re-exportar lo *core*:

     ```python
     from .core.dataset import RoxyDataset
     from .descriptors import GlobalSequenceDescriptors, BasicStructureDescriptors, BasicMoleculeDescriptors
     from .eda.summary import build_report
     from .projection import project
     from .viz import plot_feature_distribution, plot_embedding
     from .report import dataset_report_to_markdown, dataset_report_to_html

     __all__ = [
         "RoxyDataset",
         "GlobalSequenceDescriptors",
         "BasicStructureDescriptors",
         "BasicMoleculeDescriptors",
         "build_report",
         "project",
         "plot_feature_distribution",
         "plot_embedding",
         "dataset_report_to_markdown",
         "dataset_report_to_html",
     ]
     ```

2. **Helpers “opinionados”** (como en Sylphy):

   * Función tipo `describe_sequences(df, seq_col="sequence", y=None, ...)` que:

     * crea `RoxyDataset`,
     * corre `GlobalSequenceDescriptors` + AAIndex,
     * EDA básica + reporte + opcionalmente figuras.

   Eso baja la barrera de entrada brutalmente.

---

## 2. CLI y pipelines reproducibles

### Estado actual

* `roxy/cli/main.py` = placeholder con un `print("Roxy CLI placeholder")`.
* `core/config.py` solo tiene un `DEFAULT_DESCRIPTOR_ENGINES` muy mínimo.

### Qué haría para estar al nivel de Sylphy

1. **CLI con Typer o Click (igual que Sylphy)**:

   * Comandos tipo:

     * `roxy describe-sequences`:

       * lee CSV/Parquet,
       * aplica `seq_global`, `seq_aaindex`,
       * guarda features (`.parquet`) + reporte `.md` y `.html`.
     * `roxy eda`:

       * toma un feature file + labels,
       * corre `build_report`,
       * genera reportes + algunas figuras preconfiguradas.
     * `roxy project`:

       * aplica PCA/UMAP/t-SNE a un feature matrix,
       * guarda embeddings.
   * Opciones estándar: `--config`, `--output-dir`, `--random-state`, etc.

2. **Modelo de configuración**:

   * Un `config` dataclass o Pydantic model (similar a Sylphy):

     * paths de entrada/salida,
     * qué descriptores correr,
     * qué escalado/selección/proyección usar,
     * parámetros básicos (e.g. nº componentes PCA).
   * Parser de YAML/JSON para poder levantar pipelines tipo:

     ```yaml
     dataset:
       path: data/sequences.parquet
       sequence_column: sequence
       label_column: label

     descriptors:
       sequence:
         - seq_global
         - seq_aaindex:
             codes: ["ANDN920101", "ARGP820102"]

     projection:
       pca:
         n_components: 2
         random_state: 42
     ```

---

## 3. Logging, errores y robustez

### Estado actual

* Casi nada de logging global (solo algunos imports sueltos).
* Errores están, pero mayormente como `ValueError`, `KeyError` genéricos.
* AAIndex descarga, datasets y EDA podrían tener fallos silenciosos o poco informativos.

### Qué sumaría

1. **Logger consistente estilo Sylphy**:

   * Un `setup_logger("roxy")` central.
   * Logs en puntos clave:

     * carga/descarga de AAIndex,
     * construcción de `RoxyDataset`,
     * ejecución de descriptores,
     * EDA + generación de reportes.

2. **Clases de excepción específicas**:

   * `RoxyError`.
   * `DescriptorError`, `AAIndexError`, `ProjectionError`.
   * Usarlas en vez de `ValueError`/`RuntimeError` sueltos.

3. **Validación de entrada más fuerte**:

   * `RoxyDataset`:

     * validar índices/longitudes (`samples`, `y`, `features`).
   * `AAIndex`:

     * mensajes claros si falta el CSV, si falla la descarga, si el código no está.
   * `EDA/target_relations`:

     * controlar casos tipo “todos los valores iguales” (ya empezaste con Kruskal, habría que revisar otros test para edge cases).

---

## 4. Empaquetado, dependencias y extras

### Estado actual

* `pyproject.toml` minimal:

  * `name = "roxy"`, `version = "0.0.1"`, deps: `pandas`, `numpy`, `requests`, `scikit-learn`, `plotly`.
  * `plotly` como dependencia dura aunque `viz.dashboard` es opcional.
* `README.md` casi vacío.

### Qué haría

1. **Extras para dependencias pesadas**:

   * Algo así:

     ```toml
     [project.optional-dependencies]
     viz = ["plotly"]
     umap = ["umap-learn"]
     dev = ["pytest", "mypy", "black", "ruff", "ipykernel"]
     ```

   * Y que el código de `viz.dashboard` y `projection.UMAPReducer` dé `ImportError` elegante (ya lo hace, pero documentar que se activa con `[viz]`, `[umap]`).

2. **Metadata de proyecto más completa**:

   * `authors`, `license`, `urls` (`homepage`, `repository`), `classifiers`.
   * Versión tipo `0.1.0` para “primer release usable”.

3. **README decente**:

   * Qué es Roxy, cómo se integra con PRISM y Sylphy.
   * Quickstart con 10–15 líneas de código.
   * Tabla con módulos (`core`, `descriptors`, `eda`, `features`, `projection`, `viz`, `report`).

---

## 5. Documentación y ejemplos

### Estado actual

* README vacío.
* Notebooks de demo los tenemos, pero no están integrados como “oficiales” en el repo (supongo que los vas a mover a `simple_demos/` o `examples/`).

### Próximos pasos

1. **Docs tipo mkdocs o Sphinx**:

   * Secciones:

     * *Overview* (qué problema resuelve Roxy).
     * *Concepts*: `RoxyDataset`, feature blocks, descriptor engines, EDA, projection, viz, report.
     * *How-to guides*:

       * “Describe protein sequences”.
       * “EDA for enzyme ML dataset”.
       * “Project and visualise descriptors”.
     * *API reference* auto-generada.

2. **Carpeta de ejemplos**:

   * `examples/01_basic_sequences_eda.ipynb`
   * `examples/02_feature_selection_and_projection.ipynb`
   * `examples/03_end_to_end_report_generation.ipynb`
   * Esto ya lo tenemos casi todo; basta con limpiarlos, agregar explicaciones y dejarlos listos para doc.

---

## 6. Testing, CI y calidad de código

### Estado actual

* No hay `tests/`.
* No hay config de `pytest`, `mypy`, `ruff`, etc.
* Type hints bastante bien para lo que vi, pero sin verificación automática.

### Para llegar a “nivel Sylphy”

1. **Tests unitarios**:

   * `tests/test_dataset.py`: construcción, añadir features, `to_Xy`, merges.
   * `tests/test_descriptors_sequences.py`: un par de secuencias toy con descriptores conocidos (length, frecuencia, carga).
   * `tests/test_aaindex.py`: cargando un par de índices y validando resultados.
   * `tests/test_eda_summary.py`: `build_report` sobre un dataset pequeño.
   * `tests/test_features_scaling_selection.py`: smoke tests de todas las estrategias.
   * `tests/test_projection.py`: PCA/UMAP/t-SNE corren y devuelven shape correcto.
   * `tests/test_report_markdown_html.py`: que no explote y contenga ciertas palabras clave.

2. **Tests de integración**:

   * Pipeline completo con un dataset chico:

     * `RoxyDataset` + `GlobalSequenceDescriptors` + `AAIndex` + scaling + selección + PCA + EDA + report.

3. **CI (GitHub Actions)**:

   * Workflow con:

     * `pip install .[dev]`
     * `pytest`
     * `mypy roxy`
     * `ruff roxy tests`

4. **Herramientas de estilo**:

   * `pyproject.toml` con config de black, ruff, isort.
   * Opcional: `pre-commit` hooks.

---

## 7. Feature set y cobertura de dominio

Roxy ya cubre:

* secuencias: global + AAIndex,
* estructuras: algo básico,
* compuestos: descriptores químicos básicos,
* EDA/visualización/reporte.

Si quieres que esté al mismo “peso conceptual” que Sylphy, yo pensaría en:

1. **Estrategias adicionales de descriptores** (todas opcionales):

   * Secuencias:

     * ventanas deslizantes (k-mers, composition/transition/distribution),
     * motivos (p. ej. “razón Cys”, “número de sitios X”).
   * Estructuras:

     * simple interface a DSSP / Biopython para conteo de hélices/hojas, accesibilidad solvente.
   * Moléculas:

     * RDKit como opción “rich descriptors” (`[mol]` extra).

2. **Conectores a Sylphy**:

   * Helper tipo:

     ```python
     def add_sylphy_embeddings(ds: RoxyDataset, sylphy_output_path: str, block_name: str = "emb_sylphy") -> RoxyDataset:
         ...
     ```

   * O un adaptador más genérico: “lee un parquet/csv de Sylphy y lo monta como feature block”.

---

## 8. Experiencia end-to-end estilo “pipeline”

Pregunta base: ¿puedo, con un comando, pasar de “CSV/Parquet bruto de PRISM” a “features + EDA + reporte + figuras” de forma reproducible? Sylphy ya está cerca de eso para embeddings. Roxy debería tener:

* 1 CLI “happy path” bien definido (aunque luego lo extiendas).
* 1 notebook demo “full pipeline” que sirva casi como paper-supplement.

---

## 9. Resumen accionable

Si lo queremos ver como backlog/prioridades:

**Corto plazo (para subirlo a GitHub como lib seria)**

* [ ] Completar `roxy.__init__` con API pública.
* [ ] Implementar CLI mínimo (`roxy describe-sequences`, `roxy eda`).
* [ ] Añadir logging y excepciones propias en puntos críticos.
* [ ] Mejorar `pyproject.toml` con extras (`viz`, `umap`, `dev`).
* [ ] Escribir README con quickstart + arquitectura.

**Medio plazo (nivel Sylphy)**

* [ ] Crear `tests/` con unidad + integración.
* [ ] Configurar CI (pytest + mypy + ruff).
* [ ] Mover notebooks de demo a `examples/` y documentarlos.
* [ ] Montar docs (mkdocs o Sphinx) con guía de usuario + API.
* [ ] Implementar CLI orientado a config (YAML) para pipelines reproducibles.

**Largo plazo (ecosistema Mushoku-style)**

* [ ] Ampliar set de descriptores (sec/struct/mol) con extras opcionales.
* [ ] Helpers explícitos de integración con Sylphy y PRISM.
* [ ] Posible “mini-dashboard” Roxy (streamlit/gradio) que use `viz` + `report` sobre datasets PRISM.

