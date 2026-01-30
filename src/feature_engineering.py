from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, QuantileDiscretizer, VectorAssembler


# -------------------------------------------------
# Outlier Capping
# -------------------------------------------------

def cap_outliers(df: DataFrame, income_cap=1575000.0, years_emp_cap=43.0):
    """
    Cap extreme values for income and years employed
    """
    df = df.withColumn(
        "AMT_INCOME_TOTAL",
        when(col("AMT_INCOME_TOTAL") > income_cap, income_cap)
        .otherwise(col("AMT_INCOME_TOTAL"))
    )

    df = df.withColumn(
        "YEARS_EMPLOYED",
        when(col("YEARS_EMPLOYED") > years_emp_cap, years_emp_cap)
        .otherwise(col("YEARS_EMPLOYED"))
    )

    return df


# -------------------------------------------------
# Income Binning
# -------------------------------------------------

def create_income_bins(df: DataFrame, num_bins=5):
    """
    Create quantile-based income bins
    """
    discretizer = QuantileDiscretizer(
        inputCol="AMT_INCOME_TOTAL",
        outputCol="INCOME_BIN",
        numBuckets=num_bins,
        relativeError=0.01
    )

    model = discretizer.fit(df)
    df = model.transform(df)
    return df


# -------------------------------------------------
# Create Label
# -------------------------------------------------

def create_label(df: DataFrame):
    """
    Convert STATUS column into binary label
    """
    df = df.withColumn(
        "Label",
        F.when(F.col("STATUS").isin('1','2','3','4','5','6','7','8','9'), 1)
         .otherwise(0)
    )

    df = df.drop("STATUS")
    return df


# -------------------------------------------------
# Encode Categorical Columns
# -------------------------------------------------

def encode_categorical(df: DataFrame):
    """
    Convert categorical columns into numerical using StringIndexer
    """
    categorical_cols = [
        "CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
        "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE"
    ]

    index_output_cols = [c + "_Index" for c in categorical_cols]

    indexer = StringIndexer(
        inputCols=categorical_cols,
        outputCols=index_output_cols,
        handleInvalid="keep"
    )

    df = indexer.fit(df).transform(df)
    df = df.drop(*categorical_cols)

    return df


# -------------------------------------------------
# Credit History Aggregations
# -------------------------------------------------

def aggregate_credit_history(df: DataFrame):
    """
    Create credit history length and min/max month balance
    """
    agg_df = df.groupBy("ID").agg(
        F.count("MONTHS_BALANCE").alias("MONTH_HISTORY_LENGTH"),
        F.min("MONTHS_BALANCE").alias("MONTHS_BALANCE_MIN"),
        F.max("MONTHS_BALANCE").alias("MONTHS_BALANCE_MAX")
    )

    df = df.drop("MONTHS_BALANCE")
    df = df.join(agg_df, "ID")

    return df


# -------------------------------------------------
# Assemble Feature Vector
# -------------------------------------------------

def assemble_features(df: DataFrame):
    """
    Assemble all features into a single vector
    """
    feature_cols = [
        "CNT_CHILDREN","AMT_INCOME_TOTAL","FLAG_MOBIL",
        "FLAG_WORK_PHONE","FLAG_PHONE","FLAG_EMAIL",
        "CNT_FAM_MEMBERS","AGE","YEARS_EMPLOYED",
        "FLAG_OWN_CAR_Index","NAME_HOUSING_TYPE_Index",
        "NAME_EDUCATION_TYPE_Index","NAME_INCOME_TYPE_Index",
        "NAME_FAMILY_STATUS_Index","FLAG_OWN_REALTY_Index",
        "CODE_GENDER_Index","OCCUPATION_TYPE_Index",
        "MONTH_HISTORY_LENGTH","MONTHS_BALANCE_MIN",
        "MONTHS_BALANCE_MAX","INCOME_BIN"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    df = assembler.transform(df)
    return df.select("features", "Label")


# -------------------------------------------------
# Train / Valid / Test Split
# -------------------------------------------------

def split_data(df: DataFrame, seed=42):
    """
    Split dataset into train, validation and test sets
    """
    train_df, temp_df = df.randomSplit([0.7, 0.3], seed=seed)
    valid_df, test_df = temp_df.randomSplit([0.5, 0.5], seed=seed)
    return train_df, valid_df, test_df


# -------------------------------------------------
# Complete Feature Engineering Pipeline
# -------------------------------------------------

def build_features(df: DataFrame):
    """
    Full feature engineering pipeline
    """
    df = cap_outliers(df)
    df = create_income_bins(df)
    df = create_label(df)
    df = encode_categorical(df)
    df = aggregate_credit_history(df)
    df = df.dropDuplicates(["ID"])
    final_df = assemble_features(df)

    train_df, valid_df, test_df = split_data(final_df)
    return train_df, valid_df, test_df
