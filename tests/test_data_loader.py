"""Unit tests for the idempotent AI4I 2020 data ingestion module."""

import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    DATA_URL,
    EXPECTED_COLS,
    EXPECTED_ROWS,
    MIN_EXPECTED_BYTES,
    RAW_DATA_PATH,
    download_ai4i_dataset,
    load_raw_data,
)
from src.exceptions import DataValidationError

AI4I_COLUMNS: list[str] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def _build_bom_csv_bytes(num_rows: int = EXPECTED_ROWS) -> bytes:
    """Builds a synthetic AI4I 2020 CSV payload prefixed with a UTF-8 BOM."""
    header: str = "\ufeff" + ",".join(AI4I_COLUMNS)
    sample_row: str = "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0"
    lines: list[str] = [header] + [sample_row] * num_rows
    return ("\n".join(lines) + "\n").encode("utf-8")


def _build_zip_bytes(entries: dict[str, bytes]) -> bytes:
    """Creates an in-memory ZIP archive from a mapping of filename to bytes."""
    buffer: BytesIO = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in entries.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


class _FakeHTTPResponse:
    """Minimal context-manager mock for urllib.request.urlopen responses."""

    def __init__(self, payload: bytes) -> None:
        """Initializes the mock response with raw bytes."""
        self._payload: bytes = payload

    def read(self) -> bytes:
        """Returns the full mock payload."""
        return self._payload

    def __enter__(self) -> "_FakeHTTPResponse":
        """Enters the response context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exits the response context manager without suppressing errors."""
        return None


def test_module_constants() -> None:
    """Verifies required module-level ingestion constants."""
    assert DATA_URL.endswith("ai4i+2020+predictive+maintenance+dataset.zip")
    assert RAW_DATA_PATH == Path("data/raw/ai4i2020.csv")
    assert MIN_EXPECTED_BYTES == 500_000


def test_download_ai4i_dataset_extracts_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies download_ai4i_dataset extracts ai4i2020.csv from an in-memory ZIP."""
    csv_bytes: bytes = _build_bom_csv_bytes(num_rows=5)
    zip_bytes: bytes = _build_zip_bytes({"ai4i2020.csv": csv_bytes})
    captured_urls: list[str] = []

    def fake_urlopen(request: urllib.request.Request) -> _FakeHTTPResponse:
        captured_urls.append(request.full_url)
        return _FakeHTTPResponse(zip_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    dest_path: Path = tmp_path / "nested" / "ai4i2020.csv"
    returned_path: Path = download_ai4i_dataset(url=DATA_URL, dest_path=dest_path)

    assert returned_path == dest_path
    assert dest_path.exists()
    assert dest_path.read_bytes() == csv_bytes
    assert captured_urls == [DATA_URL]


def test_download_ai4i_dataset_idempotent_skips_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies network is skipped when local CSV meets MIN_EXPECTED_BYTES."""
    dest_path: Path = tmp_path / "ai4i2020.csv"
    dest_path.write_bytes(b"x" * MIN_EXPECTED_BYTES)

    def fail_urlopen(request: urllib.request.Request) -> _FakeHTTPResponse:
        raise AssertionError("Network call should be skipped for cached dataset.")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    returned_path: Path = download_ai4i_dataset(dest_path=dest_path)
    assert returned_path == dest_path


def test_download_ai4i_dataset_retries_when_incomplete_or_forced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies incomplete local files and force_download=True trigger download."""
    dest_path: Path = tmp_path / "ai4i2020.csv"
    dest_path.write_bytes(b"small")

    fresh_csv: bytes = _build_bom_csv_bytes(num_rows=3)
    zip_bytes: bytes = _build_zip_bytes({"ai4i2020.csv": fresh_csv})
    call_count: int = 0

    def fake_urlopen(request: urllib.request.Request) -> _FakeHTTPResponse:
        nonlocal call_count
        call_count += 1
        return _FakeHTTPResponse(zip_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    download_ai4i_dataset(dest_path=dest_path)
    assert call_count == 1
    assert dest_path.read_bytes() == fresh_csv

    dest_path.write_bytes(b"y" * MIN_EXPECTED_BYTES)
    download_ai4i_dataset(dest_path=dest_path, force_download=True)
    assert call_count == 2
    assert dest_path.read_bytes() == fresh_csv


def test_download_ai4i_dataset_missing_csv_in_zip_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies FileNotFoundError is raised when ai4i2020.csv is absent from ZIP."""
    zip_bytes: bytes = _build_zip_bytes({"unexpected.csv": b"a,b\n1,2\n"})

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request: _FakeHTTPResponse(zip_bytes),
    )

    dest_path: Path = tmp_path / "ai4i2020.csv"
    with pytest.raises(FileNotFoundError, match="ai4i2020.csv"):
        download_ai4i_dataset(dest_path=dest_path)


def test_download_ai4i_dataset_corrupted_zip_raises_data_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies DataValidationError is raised when downloaded bytes are not a ZIP."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request: _FakeHTTPResponse(b"not-a-valid-zip-archive"),
    )

    dest_path: Path = tmp_path / "ai4i2020.csv"
    with pytest.raises(DataValidationError, match="valid ZIP"):
        download_ai4i_dataset(dest_path=dest_path)


def test_load_raw_data_strips_utf8_bom_and_validates_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies load_raw_data downloads if missing, strips UTF-8 BOM, and checks shape."""
    csv_bytes: bytes = _build_bom_csv_bytes(num_rows=EXPECTED_ROWS)
    zip_bytes: bytes = _build_zip_bytes({"ai4i2020.csv": csv_bytes})

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request: _FakeHTTPResponse(zip_bytes),
    )

    data_path: Path = tmp_path / "ai4i2020.csv"
    df: pd.DataFrame = load_raw_data(data_path=data_path)

    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLS)
    assert list(df.columns) == AI4I_COLUMNS
    assert not any(col.startswith("\ufeff") for col in df.columns)


def test_load_raw_data_raises_on_empty_file(tmp_path: Path) -> None:
    """Verifies DataValidationError is raised when the CSV file is empty."""
    empty_path: Path = tmp_path / "empty.csv"
    empty_path.write_bytes(b"")

    with pytest.raises(DataValidationError, match="empty or corrupted"):
        load_raw_data(data_path=empty_path)


def test_load_raw_data_raises_on_unexpected_shape(tmp_path: Path) -> None:
    """Verifies DataValidationError is raised when dimensions do not match (10000, 14)."""
    wrong_shape_path: Path = tmp_path / "short.csv"
    wrong_shape_path.write_bytes(_build_bom_csv_bytes(num_rows=10))

    with pytest.raises(DataValidationError, match="Invalid dataset shape"):
        load_raw_data(data_path=wrong_shape_path)


def test_load_raw_data_raises_on_corrupted_encoding(tmp_path: Path) -> None:
    """Verifies DataValidationError is raised when CSV bytes cannot be decoded."""
    corrupt_path: Path = tmp_path / "corrupt.csv"
    corrupt_path.write_bytes(b"\xff\xfe\x00\x00invalid-utf8-\x80\x81")

    with pytest.raises(DataValidationError, match="empty or corrupted"):
        load_raw_data(data_path=corrupt_path)
