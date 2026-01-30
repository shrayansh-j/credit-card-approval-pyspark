from pyspark.sql import DataFrame
from pyspark.sql.functions import isnull, when, col, count


def check_null_counts(df: DataFrame):
    """
    Returns a dataframe showing null counts for each column
    """
    return df.select(
        [count(when(isnull(c), c)).alias(c) for c in df.columns]
    )


def remove_nulls_and_duplicates(df: DataFrame):
    """
    Remove rows with null values and duplicate rows
    """
    df = df.dropna()
    df = df.dropDuplicates()
    return df


def create_age_and_employment_features(df: DataFrame):
    """
    Create AGE and YEARS_EMPLOYED from raw day columns
    """
    df = df.withColumn("AGE", (-col("DAYS_BIRTH") / 365).cast("int"))
    df = df.withColumn("YEARS_EMPLOYED", (-col("DAYS_EMPLOYED") / 365).cast("int"))
    df = df.drop("DAYS_BIRTH", "DAYS_EMPLOYED")
    return df


def preprocess_data(df: DataFrame):
    """
    Full preprocessing pipeline
    """
    df = remove_nulls_and_duplicates(df)
    df = create_age_and_employment_features(df)
    return df
