"""Idempotent data ingestion module for AI4I 2020 industrial telemetry.

FleetReliability-AI — Condition-Based Maintenance (CBM) predictive system.
Downloads, extracts, caches, and validates the raw UCI AI4I 2020 dataset.
"""

import logging
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.constants import configure_logging
from src.exceptions import DataValidationError

logger: logging.Logger = configure_logging()

DATA_URL: str = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
RAW_DATA_PATH: Path = Path("data/raw/ai4i2020.csv")
MIN_EXPECTED_BYTES: int = 500_000
TARGET_ZIP_MEMBER: str = "ai4i2020.csv"
EXPECTED_ROWS: int = 10_000
EXPECTED_COLS: int = 14
EXPECTED_SHAPE: tuple[int, int] = (EXPECTED_ROWS, EXPECTED_COLS)


def download_ai4i_dataset(
    url: str = DATA_URL,
    dest_path: Path = RAW_DATA_PATH,
    force_download: bool = False,
) -> Path:
    """Downloads and extracts the AI4I 2020 dataset idempotently.

    Skips network retrieval when ``dest_path`` already exists with a byte size
    at least equal to ``MIN_EXPECTED_BYTES``, unless ``force_download`` is
    ``True``.

    Args:
        url: Official UCI ML Repository ZIP download URL.
        dest_path: Destination CSV file path on local disk.
        force_download: If ``True``, re-downloads even when a valid cached file
            already exists on disk.

    Returns:
        Path: Local path to the extracted raw CSV file.

    Raises:
        FileNotFoundError: If ``ai4i2020.csv`` is missing inside the ZIP archive.
        DataValidationError: If the downloaded payload is not a valid ZIP archive.
        urllib.error.URLError: If the network request fails.

    Example:
        >>> from pathlib import Path
        >>> path = download_ai4i_dataset(dest_path=Path("data/raw/ai4i2020.csv"))
        >>> path.name
        'ai4i2020.csv'
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_download and dest_path.exists():
        file_size: int = dest_path.stat().st_size
        if file_size >= MIN_EXPECTED_BYTES:
            logger.info(
                "Dataset already exists at %s (%d bytes). Skipping download.",
                dest_path,
                file_size,
            )
            return dest_path
        logger.warning(
            "Existing dataset file at %s appears incomplete (%d bytes). Retrying download.",
            dest_path,
            file_size,
        )

    logger.info("Downloading industrial telemetry dataset from %s", url)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FleetReliability-AI/1.0"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            archive_bytes: bytes = response.read()
    except urllib.error.URLError as err:
        logger.error("Failed to download dataset from %s: %s", url, err)
        raise

    try:
        with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
            if TARGET_ZIP_MEMBER not in archive.namelist():
                logger.error(
                    "Expected file '%s' missing from ZIP archive at %s.",
                    TARGET_ZIP_MEMBER,
                    url,
                )
                raise FileNotFoundError(
                    f"Expected file '{TARGET_ZIP_MEMBER}' was not found inside the downloaded ZIP archive."
                )
            with (
                archive.open(TARGET_ZIP_MEMBER) as source,
                open(dest_path, "wb") as target,
            ):
                target.write(source.read())
    except zipfile.BadZipFile as err:
        logger.error("Downloaded payload from %s is not a valid ZIP archive.", url)
        raise DataValidationError(
            f"Downloaded payload from '{url}' is corrupted or not a valid ZIP archive."
        ) from err

    logger.info("Dataset verified and saved successfully to %s", dest_path)
    return dest_path


def load_raw_data(data_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads and validates the raw AI4I 2020 CSV dataset with UTF-8 BOM handling.

    Triggers idempotent download if ``data_path`` does not exist locally, reads
    the CSV using ``utf-8-sig`` encoding to strip the UTF-8 Byte Order Mark,
    and validates that the resulting DataFrame matches the expected shape
    ``(10000, 14)``.

    Args:
        data_path: Local path to the raw CSV file.

    Returns:
        pd.DataFrame: Validated raw telemetry DataFrame with 10,000 rows and 14
            base columns.

    Raises:
        DataValidationError: If the CSV file is empty, unreadable, corrupted, or
            does not match the expected ``(10000, 14)`` dimensions.
        FileNotFoundError: If a triggered download lacks ``ai4i2020.csv`` in
            the ZIP archive.

    Example:
        >>> df = load_raw_data()
        >>> df.shape
        (10000, 14)
    """
    if not data_path.exists():
        download_ai4i_dataset(dest_path=data_path)

    try:
        df: pd.DataFrame = pd.read_csv(data_path, encoding="utf-8-sig")
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError,
    ) as err:
        logger.error("Failed to parse raw CSV dataset at %s: %s", data_path, err)
        raise DataValidationError(
            f"Raw dataset at '{data_path}' is empty or corrupted."
        ) from err

    if df.empty or df.shape != EXPECTED_SHAPE:
        logger.error(
            "Invalid raw dataset shape %s at %s; expected %s.",
            df.shape,
            data_path,
            EXPECTED_SHAPE,
        )
        raise DataValidationError(
            f"Invalid dataset shape {df.shape} at '{data_path}'; expected {EXPECTED_SHAPE}."
        )

    logger.info(
        "Loaded raw telemetry dataset from %s with shape %s.",
        data_path,
        df.shape,
    )
    return df


def main() -> None:
    """CLI entrypoint to download and verify the raw AI4I 2020 dataset."""
    configure_logging()
    csv_path: Path = download_ai4i_dataset()
    df: pd.DataFrame = load_raw_data(data_path=csv_path)
    logger.info(
        "Raw dataset verification completed: path=%s shape=%s",
        csv_path,
        df.shape,
    )


if __name__ == "__main__":
    main()
