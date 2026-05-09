"""Pytest configuration for the delsys test suite."""
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# Canonical mapping of fixture file -> (sensor count, signal count) per format.
# These numbers match the originals in C:/dev/immersionToolbox/_data/. They
# were derived empirically and locked in at fixture generation time.
SAMPLE_COUNTS = {
    "emgworks.csv":                  (12, 69),
    "discover142.csv":               (8, 50),
    "discover150.csv":               (10, 64),
    "discover164_link.csv":          (12, 46),
    "discover164_basic.csv":         (10, 37),
    "discover164_mvc.csv":           (9, 9),
    "discover170.csv":               (18, 111),
}


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to the test fixture directory."""
    return FIXTURES_DIR


@pytest.fixture(params=sorted(SAMPLE_COUNTS.keys()))
def sample_csv(request, fixtures_dir):
    """Parametrize a test across every committed CSV fixture.

    Yields ``(path, expected_sensor_count, expected_signal_count)``.
    """
    name = request.param
    path = fixtures_dir / name
    if not path.exists():
        pytest.skip(f"fixture {name} not present")
    n_sensors, n_signals = SAMPLE_COUNTS[name]
    return path, n_sensors, n_signals
