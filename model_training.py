import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


def prepare_top_classes(data, top_n=10):
    """
    Keep only the top N most frequent specialties.
    """

    data = data.copy()

    top_classes = data["medical_specialty"].value_counts().head(top_n).index

    filtered_data = data[data["medical_specialty"].isin(top_classes)].copy()

    filtered_data = filtered_data.reset_index(drop=True)

    print(f"\nUsing top {top_n} specialties only")
    print("Filtered dataset shape:", filtered_data.shape)

    print("\nClass distribution:")
    print(filtered_data["medical_specialty"].value_counts())

    return filtered_data


def encode_target(data):
    """
    Encode specialty labels into numbers.
    """

    label_encoder = LabelEncoder()
    data = data.copy()

    data["label"] = label_encoder.fit_transform(data["medical_specialty"])

    print("\nEncoded class mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{i} -> {class_name}")

    return data, label_encoder


def split_and_vectorize(data):
    """
    Split data and apply TF-IDF vectorization.
    """

    X = data["cleaned_transcription"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("\nTF-IDF shapes:")
    print("X_train_tfidf:", X_train_tfidf.shape)
    print("X_test_tfidf:", X_test_tfidf.shape)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf


def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    """
    Train multiple models and compare performance.
    """

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = []

    for model_name, model in models.items():
        print(f"\n========== {model_name} ==========")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")

        print("Accuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        ))

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1
        })

    results_df = pd.DataFrame(results)

    return results_df