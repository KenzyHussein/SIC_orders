import pytest
from pyspark.sql import SparkSession

from pyspark_job import clean_data


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("PySparkTest")
        .getOrCreate()
    )

    yield spark

    spark.stop()


def test_clean_data(spark):
    data = [
        ("Alice", 100.0),
        ("Bob", -20.0),
        ("Charlie", 0.0),
        (None, 50.0),
        ("David", 200.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    rows = result.collect()

    assert len(rows) == 2


def test_valid_records_are_kept(spark):
    data = [
        ("Alice", 100.0),
        ("David", 200.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    names = [row["name"] for row in result.collect()]

    assert "Alice" in names
    assert "David" in names


def test_invalid_amounts_are_removed(spark):
    data = [
        ("Alice", 100.0),
        ("Bob", -20.0),
        ("Charlie", 0.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    names = [row["name"] for row in result.collect()]

    assert "Bob" not in names
    assert "Charlie" not in names
    assert "Alice" in names


def test_null_names_are_removed(spark):
    data = [
        ("Alice", 100.0),
        (None, 50.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    names = [row["name"] for row in result.collect()]

    assert None not in names
    assert "Alice" in names


def test_amount_with_tax(spark):
    data = [
        ("Alice", 100.0),
        ("David", 200.0),
    ]

    df = spark.createDataFrame(
        data,
        ["name", "amount"]
    )

    result = clean_data(df)

    rows = result.collect()

    amounts_with_tax = {
        row["name"]: row["amount_with_tax"]
        for row in rows
    }

    assert amounts_with_tax["Alice"] == pytest.approx(120.0)
    assert amounts_with_tax["David"] == pytest.approx(240.0)