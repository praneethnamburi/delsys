"""Pytest configuration for the delsys test suite."""
from pathlib import Path
import pytest

# Path to the directory holding test fixture CSVs; populated as we add
# trimmed sample files (Stage 4 of the extraction plan).
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the test fixture directory."""
    return FIXTURES_DIR
