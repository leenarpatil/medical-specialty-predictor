import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB


def train_best_model(X_train, y_train):
    """
    Train the best-performing model.
    Based on your current results, this is Multinomial Naive Bayes.
    """
    best_model = MultinomialNB()
    best_model.fit(X_train, y_train)
    return best_model


def save_artifacts(model, tfidf, label_encoder, output_dir="outputs"):
    """
    Save trained model, vectorizer, and label encoder.
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(model, os.path.join(output_dir, "best_model_nb.pkl"))
    joblib.dump(tfidf, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    print("\nSaved model artifacts:")
    print("- outputs/best_model_nb.pkl")
    print("- outputs/tfidf_vectorizer.pkl")
    print("- outputs/label_encoder.pkl")


def plot_confusion_matrix_for_best_model(model, X_test, y_test, label_encoder, output_dir="outputs"):
    """
    Plot and save confusion matrix for the best model.
    """
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.title("Confusion Matrix - Best Model (Naive Bayes)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "confusion_matrix_nb.png"))
    plt.show()

    print("\nConfusion matrix saved to outputs/confusion_matrix_nb.png")


def save_predictions_sample(model, X_test, y_test, label_encoder, output_dir="outputs", sample_size=30):
    """
    Save a sample of actual vs predicted labels.
    """
    y_pred = model.predict(X_test)

    actual_labels = label_encoder.inverse_transform(y_test)
    predicted_labels = label_encoder.inverse_transform(y_pred)

    results_df = pd.DataFrame({
        "Actual": actual_labels,
        "Predicted": predicted_labels
    })

    sample_df = results_df.head(sample_size)
    sample_df.to_csv(os.path.join(output_dir, "sample_predictions.csv"), index=False)

    print("\nSample predictions saved to outputs/sample_predictions.csv")