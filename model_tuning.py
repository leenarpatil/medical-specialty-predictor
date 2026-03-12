import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def run_cross_validation(data):
    """
    Run cross-validation using a balanced Logistic Regression pipeline.
    """

    X = data["cleaned_transcription"]
    y = data["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accuracy_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    f1_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="f1_macro"
    )

    print("\n========== CROSS-VALIDATION RESULTS ==========")
    print("Accuracy scores for 5 folds:", accuracy_scores)
    print("Mean CV Accuracy:", round(accuracy_scores.mean(), 4))

    print("\nMacro F1 scores for 5 folds:", f1_scores)
    print("Mean CV Macro F1:", round(f1_scores.mean(), 4))

    cv_results = pd.DataFrame({
        "Fold": [1, 2, 3, 4, 5],
        "Accuracy": accuracy_scores,
        "Macro_F1": f1_scores
    })

    return cv_results


def run_grid_search(X_train, y_train):
    """
    Run GridSearchCV for balanced Logistic Regression with TF-IDF.
    """

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ))
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000, 7000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "clf__C": [0.1, 1, 5]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\n========== GRID SEARCH RESULTS ==========")
    print("Best Parameters:")
    print(grid_search.best_params_)

    print("\nBest Cross-Validation Score (Macro F1):")
    print(round(grid_search.best_score_, 4))

    return grid_search


def evaluate_tuned_model(best_model, X_test, y_test, label_encoder):
    """
    Evaluate the tuned best model on the held-out test set.
    """

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n========== TUNED MODEL TEST RESULTS ==========")
    print("Test Accuracy:", round(accuracy, 4))
    print("Test Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    results = pd.DataFrame({
        "Metric": ["Test Accuracy", "Test Macro F1"],
        "Value": [accuracy, macro_f1]
    })

    return results