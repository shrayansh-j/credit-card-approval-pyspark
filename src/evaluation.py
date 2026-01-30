import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score
)


# -------------------------------------------------
# Threshold Evaluation
# -------------------------------------------------

def threshold_metrics(model, test_df, thresholds):
    pred = model.transform(test_df)
    pred = pred.withColumn("prob1", vector_to_array(col("probability"))[1])

    results = []

    for t in thresholds:
        pred_t = pred.withColumn("pred_t", (col("prob1") > t).cast("int"))

        TP = pred_t.filter("Label = 1 AND pred_t = 1").count()
        FP = pred_t.filter("Label = 0 AND pred_t = 1").count()
        TN = pred_t.filter("Label = 0 AND pred_t = 0").count()
        FN = pred_t.filter("Label = 1 AND pred_t = 0").count()

        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        results.append((t, precision, recall, f1, TP, FP, TN, FN))

    return results


# -------------------------------------------------
# AUC Evaluation
# -------------------------------------------------

def compute_auc(model, test_df):
    evaluator = BinaryClassificationEvaluator(
        labelCol="Label",
        rawPredictionCol="rawPrediction"
    )
    return evaluator.evaluate(model.transform(test_df))


# -------------------------------------------------
# Plot ROC Curve
# -------------------------------------------------

def plot_roc_curve(model, test_df):
    pred = model.transform(test_df)
    pred = pred.withColumn("prob1", vector_to_array(col("probability"))[1])

    pdf = pred.select("prob1","Label").toPandas()

    fpr, tpr, _ = roc_curve(pdf["Label"], pdf["prob1"])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return roc_auc


# -------------------------------------------------
# Plot Precision-Recall Curve
# -------------------------------------------------

def plot_pr_curve(model, test_df):
    pred = model.transform(test_df)
    pred = pred.withColumn("prob1", vector_to_array(col("probability"))[1])

    pdf = pred.select("prob1","Label").toPandas()

    precision, recall, _ = precision_recall_curve(
        pdf["Label"], pdf["prob1"]
    )

    ap = average_precision_score(pdf["Label"], pdf["prob1"])

    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()

    return ap


# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

def plot_confusion_matrix(TP, FP, TN, FN):
    data = {
        "Predicted Default":[TP, FP],
        "Predicted Safe":[FN, TN]
    }

    df = pd.DataFrame(
        data,
        index=["Actual Default","Actual Safe"]
    )

    plt.figure(figsize=(6,4))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
