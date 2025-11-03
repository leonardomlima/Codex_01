"""Utilities for loading and profiling tabular datasets."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Iterable, Optional

import pandas as pd


SUPPORTED_FILE_TYPES = ("csv", "xlsx")


class UnsupportedFileTypeError(ValueError):
    """Raised when the provided file extension is not supported."""


@dataclass
class DatasetMetadata:
    """Basic metadata that describes a pandas DataFrame."""

    filename: str
    rows: int
    columns: int
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    preview: pd.DataFrame
    categorical_summaries: Dict[str, Dict[str, int]]
    numerical_summaries: Dict[str, Dict[str, float]]

    def as_dict(self) -> Dict[str, Any]:
        """Return the metadata as a serialisable dictionary."""
        return {
            "filename": self.filename,
            "rows": self.rows,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "missing_values": self.missing_values,
            "categorical_summaries": self.categorical_summaries,
            "numerical_summaries": self.numerical_summaries,
            "preview": self.preview.to_dict(orient="records"),
        }

    def to_markdown(self) -> str:
        """Render a human readable summary in Markdown format."""
        lines = [f"**Arquivo:** {self.filename}", f"**Linhas:** {self.rows}", f"**Colunas:** {self.columns}"]
        lines.append("\n**Tipos de dados**")
        for col, dtype in self.dtypes.items():
            lines.append(f"- {col}: {dtype}")
        if any(self.missing_values.values()):
            lines.append("\n**Valores ausentes**")
            for col, missing in self.missing_values.items():
                if missing:
                    lines.append(f"- {col}: {missing}")
        return "\n".join(lines)


def _read_csv(file: BytesIO, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(file, **kwargs)


def _read_excel(file: BytesIO, **kwargs: Any) -> pd.DataFrame:
    return pd.read_excel(file, **kwargs)


READERS = {
    "csv": _read_csv,
    "xlsx": _read_excel,
}


def load_dataset(file: BytesIO, filename: str, *, read_kwargs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load a dataset from an uploaded file into a pandas DataFrame.

    Parameters
    ----------
    file:
        File-like object containing the dataset bytes.
    filename:
        The original filename, used for inferring the file type.
    read_kwargs:
        Optional keyword arguments forwarded to the pandas reader.
    """

    read_kwargs = read_kwargs or {}
    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in READERS:
        raise UnsupportedFileTypeError(
            f"Tipo de arquivo '{extension}' não suportado. Utilize: {', '.join(SUPPORTED_FILE_TYPES)}."
        )

    reader = READERS[extension]
    return reader(file, **read_kwargs)


def generate_metadata(df: pd.DataFrame, filename: str, *, preview_rows: int = 5) -> DatasetMetadata:
    """Create a :class:`DatasetMetadata` instance from a DataFrame."""

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_values = df.isna().sum().to_dict()

    categorical_summaries: Dict[str, Dict[str, int]] = {}
    numerical_summaries: Dict[str, Dict[str, float]] = {}

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            numerical_summaries[column] = {
                "mean": float(desc.get("mean", 0.0)),
                "std": float(desc.get("std", 0.0)),
                "min": float(desc.get("min", 0.0)),
                "max": float(desc.get("max", 0.0)),
            }
        else:
            categorical_summaries[column] = series.value_counts(dropna=False).head(5).to_dict()

    preview = df.head(preview_rows)

    return DatasetMetadata(
        filename=filename,
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        dtypes=dtypes,
        missing_values=missing_values,
        preview=preview,
        categorical_summaries=categorical_summaries,
        numerical_summaries=numerical_summaries,
    )


def dataframe_to_markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Convert a DataFrame to a Markdown table limited by ``max_rows``."""
    limited_df = df.head(max_rows)
    return limited_df.to_markdown(index=False)


def dataframe_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute general statistics for a DataFrame."""
    numeric = df.select_dtypes(include=["number"])
    categorical = df.select_dtypes(exclude=["number"])
    return {
        "numeric_summary": numeric.describe().to_dict() if not numeric.empty else {},
        "categorical_top_values": {
            col: categorical[col].value_counts().head(5).to_dict() for col in categorical.columns
        },
    }
