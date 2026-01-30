from pyspark.sql import SparkSession

# Ingestion
from data_ingestion import load_data

# Preprocessing
from preprocessing import preprocess_data

# Feature Engineering
from feature_engineering import build_features

# Modeling
from modeling import train_xgboost

# Evaluation
from evaluation import (
    threshold_metrics,
    compute_auc,
    plot_roc_curve,
    plot_pr_curve
)

# Config
from config import FINAL_THRESHOLD, MODEL_SAVE_PATH


# -------------------------------------------------
# Create Spark Session
# -------------------------------------------------

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Credit Card Approval ML Pipeline") \
        .getOrCreate()
    return spark


# -------------------------------------------------
# Main Pipeline
# -------------------------------------------------

def main():

    spark = create_spark_session()

    print("Starting ML Pipeline...")

    # 1. Data Ingestion
    print("Loading data...")
    combined_df = load_data(spark)

    # 2. Preprocessing
    print("Preprocessing data...")
    clean_df = preprocess_data(combined_df)

    # 3. Feature Engineering
    print("Building features...")
    train_df, valid_df, test_df = build_features(clean_df)

    # 4. Modeling
    print("Training XGBoost model...")
    best_model, best_params, best_auc = train_xgboost(train_df, valid_df)

    print("Best Params:", best_params)
    print("Best Validation AUC:", best_auc)

    # 5. Evaluation
    print("Evaluating model...")

    thresholds = [0.25, 0.3, 0.4, 0.5, 0.6, FINAL_THRESHOLD]
    metrics = threshold_metrics(best_model, test_df, thresholds)

    for m in metrics:
        print(m)

    test_auc = compute_auc(best_model, test_df)
    print("Test AUC:", test_auc)

    plot_roc_curve(best_model, test_df)
    plot_pr_curve(best_model, test_df)

    # 6. Save Model
    print("Saving final model...")
    best_model.write().overwrite().save(MODEL_SAVE_PATH)

    print("Pipeline completed successfully!")


# -------------------------------------------------
# Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    main()
