from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model performance
    """

    # Predictions
    y_pred = model.predict(X_test)


    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAccuracy:")
    print(accuracy)


    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(model, X_test, y_test):

    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Random Forest IDS")

    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        "results/confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("Confusion matrix saved!")