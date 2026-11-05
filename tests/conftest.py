"""Shared fixtures: the built warehouse + initialised MBMS."""

import subprocess
import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def con():
    db = ROOT / "warehouse" / "hotel_rm.duckdb"
    if not db.exists():
        subprocess.run([sys.executable, "etl.py"], cwd=ROOT, check=True)
    c = duckdb.connect(str(db))
    sys.path.insert(0, str(ROOT))
    import mbms
    mbms.init_mbms(c)
    yield c
    c.close()
