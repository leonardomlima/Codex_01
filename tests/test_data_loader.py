import io

import pandas as pd
import pytest

from app.analysis.data_loader import (
    DatasetMetadata,
    UnsupportedFileTypeError,
    dataframe_statistics,
    dataframe_to_markdown_table,
    generate_metadata,
    load_dataset,
)


def _create_csv_bytes() -> io.BytesIO:
    dataframe = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    buffer = io.BytesIO()
    dataframe.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def _create_excel_bytes() -> io.BytesIO:
    dataframe = pd.DataFrame({"x": [10, 20], "y": [0.5, 0.7]})
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer


def test_load_dataset_csv():
    buffer = _create_csv_bytes()
    df = load_dataset(buffer, "sample.csv")
    assert list(df.columns) == ["col1", "col2"]
    assert df.shape == (3, 2)


def test_load_dataset_excel():
    buffer = _create_excel_bytes()
    df = load_dataset(buffer, "sample.xlsx")
    assert list(df.columns) == ["x", "y"]
    assert df.shape == (2, 2)


def test_load_dataset_unsupported_extension():
    buffer = io.BytesIO(b"dummy")
    with pytest.raises(UnsupportedFileTypeError):
        load_dataset(buffer, "data.json")


def test_generate_metadata_basic_fields():
    df = pd.DataFrame({"num": [1, 2, 3], "cat": ["a", "b", "a"]})
    metadata = generate_metadata(df, "dataset.csv", preview_rows=2)

    assert isinstance(metadata, DatasetMetadata)
    assert metadata.filename == "dataset.csv"
    assert metadata.rows == 3
    assert metadata.columns == 2
    assert metadata.preview.shape[0] == 2
    assert metadata.numerical_summaries["num"]["mean"] == pytest.approx(2.0)
    assert metadata.categorical_summaries["cat"]["a"] == 2


def test_metadata_to_markdown_contains_core_information():
    df = pd.DataFrame({"value": [1, 2, None]})
    metadata = generate_metadata(df, "file.csv")
    markdown = metadata.to_markdown()

    assert "**Arquivo:** file.csv" in markdown
    assert "value: float64" in markdown
    assert "value: 1" in markdown  # missing values count


def test_dataframe_helpers():
    df = pd.DataFrame({"numbers": [1, 2, 3], "letters": ["a", "a", "b"]})

    markdown_table = dataframe_to_markdown_table(df, max_rows=2)
    assert "| numbers | letters |" in markdown_table
    assert markdown_table.count("\n") <= 4  # header + 2 rows

    stats = dataframe_statistics(df)
    assert stats["numeric_summary"]["numbers"]["mean"] == pytest.approx(2.0)
    assert stats["categorical_top_values"]["letters"]["a"] == 2
