from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBClassifier


# -------------------------------------------------
# Baseline Model - Logistic Regression
# -------------------------------------------------

def train_logistic_regression(train_df):
    lr = LogisticRegression(featuresCol="features", labelCol="Label")
    model = lr.fit(train_df)
    return model


# -------------------------------------------------
# XGBoost Training with Grid Search
# -------------------------------------------------

def train_xgboost(train_df, valid_df):
    evaluator = BinaryClassificationEvaluator(
        labelCol="Label",
        rawPredictionCol="rawPrediction"
    )

    scale_pos_weight = (
        train_df.filter("Label = 0").count() /
        train_df.filter("Label = 1").count()
    )

    param_grid = [
        {"max_depth":4, "eta":0.05, "subsample":0.8, "colsample_bytree":0.8, "min_child_weight":1},
        {"max_depth":6, "eta":0.05, "subsample":0.8, "colsample_bytree":0.8, "min_child_weight":1},
        {"max_depth":6, "eta":0.1, "subsample":0.9, "colsample_bytree":0.9, "min_child_weight":2},
        {"max_depth":8, "eta":0.1, "subsample":0.9, "colsample_bytree":0.9, "min_child_weight":3},
        {"max_depth":8, "eta":0.15, "subsample":0.7, "colsample_bytree":0.7, "min_child_weight":5}
    ]

    best_auc = -1
    best_model = None
    best_params = None

    for params in param_grid:

        xgb = SparkXGBClassifier(
            features_col="features",
            label_col="Label",
            max_depth=params["max_depth"],
            eta=params["eta"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            num_workers=4
        )

        model = xgb.fit(train_df)
        preds = model.transform(valid_df)
        auc = evaluator.evaluate(preds)

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params

    return best_model, best_params, best_auc
