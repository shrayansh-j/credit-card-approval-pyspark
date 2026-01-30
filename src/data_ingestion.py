from pyspark.sql import SparkSession
from config import STORAGE_ACCOUNT, CONTAINER_NAME, SAS_TOKEN, get_wasbs_path


def configure_spark_for_adls(spark: SparkSession):
    """
    Configure Spark to access Azure Data Lake Storage Gen2 using SAS token
    """
    spark.conf.set(
        f"fs.azure.sas.{CONTAINER_NAME}.{STORAGE_ACCOUNT}.blob.core.windows.net",
        SAS_TOKEN.lstrip("?")
    )


def load_raw_data(spark: SparkSession):
    """
    Load application and credit datasets from ADLS Gen2
    Returns:
        df1: Application dataframe
        df2: Credit dataframe
    """
    csv_file_1 = get_wasbs_path("application_record1.csv")
    csv_file_2 = get_wasbs_path("credit_record1.csv")

    df1 = spark.read.option("header", True).option("inferSchema", True).csv(csv_file_1)
    df2 = spark.read.option("header", True).option("inferSchema", True).csv(csv_file_2)

    return df1, df2


def prepare_and_merge(df1, df2):
    """
    Rename columns and join both datasets on ID
    Returns:
        combined_df
    """
    df1 = df1.withColumnRenamed("APP_ID", "ID")
    df2 = df2.withColumnRenamed("CR_ID", "ID")

    combined_df = df1.join(df2, "ID", "inner")
    return combined_df


def load_data(spark: SparkSession):
    """
    Complete ingestion pipeline:
    - Configure Spark
    - Load raw datasets
    - Merge datasets

    Returns:
        combined_df
    """
    configure_spark_for_adls(spark)
    df1, df2 = load_raw_data(spark)
    combined_df = prepare_and_merge(df1, df2)
    return combined_df
