from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer

from roxy.helpers import RoxyHelpers
from roxy.features import ColumnScaler
from roxy.features.selection import FeatureSelector
from roxy.descriptors import DESCRIPTOR_REGISTRY
from roxy.projection import project

app = typer.Typer(help="Roxy command-line interface for feature engineering and EDA.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_table(path: Path) -> pd.DataFrame:
    """Load a tabular file (CSV or Parquet) into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file extension {suffix!r} for {path}. "
        "Supported: .csv, .parquet"
    )


def _save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV or Parquet, chosen by extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=True)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path)
    else:
        raise ValueError(
            f"Unsupported output extension {suffix!r} for {path}. "
            "Use .csv or .parquet."
        )


def _ensure_joblib():
    try:
        import joblib  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "joblib is required for this command. Install it with:\n"
            "    pip install joblib"
        ) from e


# ---------------------------------------------------------------------------
# Command: describe-sequences
# ---------------------------------------------------------------------------


@app.command("describe-sequences")
def describe_sequences(
    input_path: Path = typer.Argument(
        ...,
        help="Input table with at least a sequence column (CSV or Parquet).",
    ),
    output_dir: Path = typer.Option(
        Path("roxy_outputs"),
        "--output-dir",
        "-o",
        help="Directory where features, reports and embeddings will be saved.",
    ),
    sequence_column: str = typer.Option(
        "sequence",
        "--sequence-column",
        "-s",
        help="Name of the column containing amino-acid sequences.",
    ),
    label_column: Optional[str] = typer.Option(
        None,
        "--label-column",
        "-y",
        help="Optional name of the column containing labels (classification/regression).",
    ),
    use_aaindex: bool = typer.Option(
        True,
        "--aaindex/--no-aaindex",
        help="Whether to include AAIndex-based descriptors if available.",
    ),
    project_method: Optional[str] = typer.Option(
        "pca",
        "--project-method",
        "-p",
        help="Projection method to use (e.g. 'pca', 'umap', 'tsne'). "
             "Use 'none' to skip projection.",
    ),
    n_components: int = typer.Option(
        2,
        "--n-components",
        help="Number of components for the projection method.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for projection methods that support it.",
    ),
    output_format: str = typer.Option(
        "parquet",
        "--format",
        "-f",
        help="Output format for features and embeddings (csv or parquet).",
    ),
) -> None:
    """
    Describe protein sequences and generate features + EDA reports.

    This command:

    - loads a table with sequences,
    - runs sequence descriptor engines (global + AAIndex if available),
    - builds a combined feature matrix,
    - generates EDA reports (Markdown and HTML),
    - optionally computes a low-dimensional embedding,
    - saves everything to the specified output directory.
    """
    df = _load_table(input_path)

    # Normalise project_method
    if project_method == "none":
        project_method = None

    result = RoxyHelpers.describe_sequences(
        df,
        seq_col=sequence_column,
        y=label_column,
        dataset_name=input_path.stem,
        use_aaindex=use_aaindex,
        feature_key_prefix="",
        task_type=None,
        project_method=project_method,
        n_components=n_components,
        random_state=random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower()

    # Save combined features
    features_path = output_dir / f"features.{ 'parquet' if fmt == 'parquet' else 'csv' }"
    _save_table(result.X, features_path)
    typer.echo(f"Saved features to: {features_path}")

    # Save embedding, if available
    if result.embedding is not None:
        emb_df = pd.DataFrame(
            result.embedding,
            index=result.X.index,
            columns=[f"comp_{i+1}" for i in range(result.embedding.shape[1])],
        )
        emb_path = output_dir / f"embedding.{ 'parquet' if fmt == 'parquet' else 'csv' }"
        _save_table(emb_df, emb_path)
        typer.echo(f"Saved embedding to: {emb_path}")

    # Save reports
    md_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    md_path.write_text(result.markdown, encoding="utf-8")
    html_path.write_text(result.html, encoding="utf-8")
    typer.echo(f"Saved Markdown report to: {md_path}")
    typer.echo(f"Saved HTML report to: {html_path}")


# ---------------------------------------------------------------------------
# Command: scale-data
# ---------------------------------------------------------------------------


@app.command("scale-data")
def scale_data(
    input_path: Path = typer.Argument(
        ...,
        help="Input feature table (CSV or Parquet).",
    ),
    output_data: Path = typer.Option(
        Path("scaled_features.parquet"),
        "--output-data",
        "-o",
        help="Path where the scaled feature table will be written.",
    ),
    output_scaler: Path = typer.Option(
        Path("scaler.joblib"),
        "--output-scaler",
        "-m",
        help="Path where the fitted scaler object will be saved (joblib).",
    ),
    strategy: str = typer.Option(
        "standard",
        "--strategy",
        "-s",
        help="Scaling strategy: standard, minmax, maxabs, robust.",
    ),
    columns: Optional[List[str]] = typer.Option(
        None,
        "--column",
        "-c",
        help="Optional list of columns to scale. "
             "If omitted, all numeric columns are scaled.",
    ),
) -> None:
    """
    Scale a feature table and save both the scaled data and the scaler.

    Uses Roxy's ColumnScaler under the hood.
    """
    _ensure_joblib()
    import joblib  # type: ignore

    df = _load_table(input_path)

    scaler = ColumnScaler(strategy=strategy, columns=columns)
    df_scaled = scaler.fit_transform(df)

    _save_table(df_scaled, output_data)
    typer.echo(f"Saved scaled features to: {output_data}")

    output_scaler.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_scaler)
    typer.echo(f"Saved scaler to: {output_scaler}")


# ---------------------------------------------------------------------------
# Command: select-features
# ---------------------------------------------------------------------------


@app.command("select-features")
def select_features(
    input_path: Path = typer.Argument(
        ...,
        help="Input feature table (CSV or Parquet).",
    ),
    label_column: Optional[str] = typer.Option(
        None,
        "--label-column",
        "-y",
        help="Name of the label column in the same table. "
             "If provided, it will be removed from X and used as y.",
    ),
    output_data: Path = typer.Option(
        Path("selected_features.parquet"),
        "--output-data",
        "-o",
        help="Path where the reduced feature table will be written.",
    ),
    output_selector: Path = typer.Option(
        Path("selector.joblib"),
        "--output-selector",
        "-m",
        help="Path where the fitted selector object will be saved (joblib).",
    ),
    strategy: str = typer.Option(
        "variance",
        "--strategy",
        "-s",
        help=(
            "Feature selection strategy. Examples: "
            "variance, kbest, mutual_info, lasso, model_tree, model_linear, rfe."
        ),
    ),
    task_type: str = typer.Option(
        "classification",
        "--task-type",
        "-t",
        help="Task type: classification or regression.",
    ),
    columns: Optional[List[str]] = typer.Option(
        None,
        "--column",
        "-c",
        help="Optional subset of feature columns to consider for selection.",
    ),
    k: Optional[int] = typer.Option(
        None,
        "--k",
        help="Optional 'k' parameter for K-best style selectors.",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        help="Optional threshold parameter for model-based selectors.",
    ),
) -> None:
    """
    Run feature selection and save the reduced feature table + selector.

    This command wraps Roxy's FeatureSelector.
    """
    _ensure_joblib()
    import joblib  # type: ignore

    df = _load_table(input_path)

    y = None
    if label_column is not None:
        if label_column not in df.columns:
            raise KeyError(
                f"Label column {label_column!r} not found in input table."
            )
        y = df[label_column]
        X = df.drop(columns=[label_column])
    else:
        X = df

    selector_kwargs = {}
    if k is not None:
        selector_kwargs["k"] = k
    if threshold is not None:
        selector_kwargs["threshold"] = threshold

    selector = FeatureSelector(
        strategy=strategy,
        task_type=task_type,
        columns=columns,
        selector_kwargs=selector_kwargs,
    )
    X_selected = selector.fit_transform(X, y)

    _save_table(X_selected, output_data)
    typer.echo(f"Saved selected features to: {output_data}")

    output_selector.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selector, output_selector)
    typer.echo(f"Saved selector to: {output_selector}")


# ---------------------------------------------------------------------------
# Command: full-pipeline
# ---------------------------------------------------------------------------


@app.command("full-pipeline")
def full_pipeline(
    input_path: Path = typer.Argument(
        ...,
        help="Input table with sequences and optional labels.",
    ),
    output_dir: Path = typer.Option(
        Path("roxy_pipeline_outputs"),
        "--output-dir",
        "-o",
        help="Directory where all pipeline artefacts will be stored.",
    ),
    sequence_column: str = typer.Option(
        "sequence",
        "--sequence-column",
        "-s",
        help="Column with amino-acid sequences.",
    ),
    label_column: Optional[str] = typer.Option(
        None,
        "--label-column",
        "-y",
        help="Optional label column.",
    ),
    use_aaindex: bool = typer.Option(
        True,
        "--aaindex/--no-aaindex",
        help="Include AAIndex descriptors if available.",
    ),
    scaling_strategy: str = typer.Option(
        "standard",
        "--scaling-strategy",
        help="Scaling strategy for numeric features (standard, minmax, maxabs, robust).",
    ),
    selection_strategy: Optional[str] = typer.Option(
        None,
        "--selection-strategy",
        help=(
            "Optional feature selection strategy (variance, kbest, mutual_info, "
            "lasso, model_tree, model_linear, rfe). If None, no selection is applied."
        ),
    ),
    project_method: Optional[str] = typer.Option(
        "pca",
        "--project-method",
        help="Projection method: pca, umap, tsne. Use 'none' to skip projection.",
    ),
    n_components: int = typer.Option(
        2,
        "--n-components",
        help="Number of components for projection.",
    ),
    random_state: int = typer.Option(
        42,
        "--random-state",
        help="Random seed for projection.",
    ),
) -> None:
    """
    Run a full in-memory pipeline: descriptors → scaling → (selection) → (projection) → report.

    Artefacts written:

    - raw feature matrix (descriptors only),
    - scaled feature matrix,
    - optionally selected feature matrix,
    - embedding (if projection is enabled),
    - Markdown + HTML reports.
    """
    df = _load_table(input_path)

    # 1) Describe sequences and build base features + report
    if project_method == "none":
        project_method = None

    result = RoxyHelpers.describe_sequences(
        df,
        seq_col=sequence_column,
        y=label_column,
        dataset_name=input_path.stem,
        use_aaindex=use_aaindex,
        feature_key_prefix="",
        task_type=None,
        project_method=None,  # we reproject after scaling/selection
    )

    X = result.X
    y = result.y

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw features
    raw_path = output_dir / "features_raw.parquet"
    _save_table(X, raw_path)
    typer.echo(f"Saved raw features to: {raw_path}")

    # 2) Scaling
    scaler = ColumnScaler(strategy=scaling_strategy, columns=None)
    X_scaled = scaler.fit_transform(X)
    scaled_path = output_dir / "features_scaled.parquet"
    _save_table(X_scaled, scaled_path)
    typer.echo(f"Saved scaled features to: {scaled_path}")

    # 3) Optional selection
    X_final = X_scaled
    selector = None
    if selection_strategy is not None:
        selector = FeatureSelector(
            strategy=selection_strategy,
            task_type="classification" if y is not None else "regression",
            columns=None,
            selector_kwargs={},
        )
        X_final = selector.fit_transform(X_scaled, y)
        selected_path = output_dir / "features_selected.parquet"
        _save_table(X_final, selected_path)
        typer.echo(f"Saved selected features to: {selected_path}")

    # 4) Projection
    embedding = None
    if project_method is not None:
        embedding = project(
            X_final,
            method=project_method,
            n_components=n_components,
            random_state=random_state,
        )
        emb_df = pd.DataFrame(
            embedding,
            index=X_final.index,
            columns=[f"comp_{i+1}" for i in range(embedding.shape[1])],
        )
        emb_path = output_dir / "embedding.parquet"
        _save_table(emb_df, emb_path)
        typer.echo(f"Saved embedding to: {emb_path}")

    # 5) Reports (reuse report from describe_sequences for now)
    md_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    md_path.write_text(result.markdown, encoding="utf-8")
    html_path.write_text(result.html, encoding="utf-8")
    typer.echo(f"Saved Markdown report to: {md_path}")
    typer.echo(f"Saved HTML report to: {html_path}")


# ---------------------------------------------------------------------------
# Command: info
# ---------------------------------------------------------------------------


@app.command("info")
def info(
    what: str = typer.Option(
        "all",
        "--what",
        "-w",
        help="Which information to show: descriptors, projection, scaling, selection, all.",
    )
) -> None:
    """
    Show information about available descriptor engines and methods.
    """
    what = what.lower()

    if what in ("descriptors", "all"):
        typer.echo("Available descriptor engines (DESCRIPTOR_REGISTRY):")
        for name in sorted(DESCRIPTOR_REGISTRY.keys()):
            typer.echo(f"  - {name}")
        typer.echo("")

    if what in ("projection", "all"):
        typer.echo("Supported projection methods (CLI-level):")
        typer.echo("  - pca")
        typer.echo("  - umap")
        typer.echo("  - tsne")
        typer.echo("")

    if what in ("scaling", "all"):
        typer.echo("Supported scaling strategies:")
        typer.echo("  - standard")
        typer.echo("  - minmax")
        typer.echo("  - maxabs")
        typer.echo("  - robust")
        typer.echo("")

    if what in ("selection", "all"):
        typer.echo("Some supported feature selection strategies:")
        typer.echo("  - variance")
        typer.echo("  - kbest")
        typer.echo("  - mutual_info")
        typer.echo("  - lasso")
        typer.echo("  - model_tree")
        typer.echo("  - model_linear")
        typer.echo("  - rfe")
        typer.echo("")


def main() -> None:
    """Entry point for `python -m roxy.cli`."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
