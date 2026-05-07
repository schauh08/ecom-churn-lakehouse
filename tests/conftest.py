from __future__ import annotations

import os

import pytest

from src.common.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    spark = get_spark("pytest-suite")
    spark.conf.set("spark.ui.enabled", "false")
    spark.conf.set("spark.sql.shuffle.partitions", "4")
    yield spark
    spark.stop()