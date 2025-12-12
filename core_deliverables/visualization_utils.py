import matplotlib.pyplot as plt
import os
import csv
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from fairlearn.metrics import MetricFrame, selection_rate


def save_results_visualization(model_name, y_test, pred, proba, sensitive_features_dict, output_dir="plots"):
    """
    Generates and saves visualizations for model performance and fairness.
    
    Args:
        model_name (str): Name of the model (used for titles and filenames).
        y_test (array-like): True labels.
        pred (array-like): Predicted labels.
        proba (array-like): Predicted probabilities for the positive class.
        sensitive_features_dict (dict): Dictionary of sensitive features {name: values}.
        output_dir (str): Directory to save plots.
    """
    # Ensure output directory is relative to this file for consistency
    if not os.path.isabs(output_dir):
        base_dir = os.path.dirname(__file__)
        output_dir = os.path.join(base_dir, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Simple, widely available Matplotlib style
    try:
        plt.style.use("ggplot")
    except OSError:
        # Fallback if style is unavailable
        plt.style.use("default")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, f"{model_name}_roc_curve.png"))
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.grid(False)
    plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    
    # 3. Fairness Metrics per Sensitive Feature
    for sens_name, sens_values in sensitive_features_dict.items():
        # Create a MetricFrame
        mf = MetricFrame(
            metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
            y_true=y_test,
            y_pred=pred,
            sensitive_features=sens_values
        )
        
        metrics_df = mf.by_group
        
        # Plot Selection Rate
        plt.figure(figsize=(10, 6))
        metrics_df["selection_rate"].plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title(f'Selection Rate by {sens_name.capitalize()} - {model_name}')
        plt.ylabel('Selection Rate')
        plt.xlabel(sens_name.capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_selection_rate_{sens_name}.png"))
        plt.close()
        
        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        metrics_df["accuracy"].plot(kind="bar", color="lightgreen", edgecolor="black")
        plt.title(f'Accuracy by {sens_name.capitalize()} - {model_name}')
        plt.ylabel('Accuracy')
        plt.xlabel(sens_name.capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_accuracy_{sens_name}.png"))
        plt.close()


def log_summary_row(
    model_name,
    overall_accuracy,
    overall_auc,
    dp_gender,
    eo_gender,
    dp_race,
    eo_race,
    csv_name="model_results.csv",
):
    """
    Append a one-row summary of metrics for a given model to a CSV table.
    """
    # Ensure CSV lives next to this file by default
    if not os.path.isabs(csv_name):
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, csv_name)
    else:
        csv_path = csv_name

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "model_name",
                    "overall_accuracy",
                    "overall_auc",
                    "dp_gender",
                    "eo_gender",
                    "dp_race",
                    "eo_race",
                ]
            )
        writer.writerow(
            [
                model_name,
                overall_accuracy,
                overall_auc,
                dp_gender,
                eo_gender,
                dp_race,
                eo_race,
            ]
        )
