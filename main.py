from src.load_data import load_dataset, inspect_dataset
from src.clean_data import clean_dataset, inspect_cleaned_data
from src.text_preprocessing import apply_preprocessing, inspect_preprocessed_data
from src.eda_analysis import (
    plot_specialty_distribution,
    plot_text_length_distribution,
    plot_common_words,
    generate_wordcloud
)
from src.model_training import (
    prepare_top_classes,
    encode_target,
    split_and_vectorize,
    train_and_evaluate_models
)
from src.model_interpretation import (
    train_best_model,
    save_artifacts,
    plot_confusion_matrix_for_best_model,
    save_predictions_sample
)
from src.model_tuning import (
    run_cross_validation,
    run_grid_search,
    evaluate_tuned_model
)
from src.save_tuned_model import save_tuned_pipeline

def main():
    # File path to dataset
    file_path = "data/mtsamples.csv"

    # Load raw dataset
    df = load_dataset(file_path)

    # Inspect raw dataset
    inspect_dataset(df)

    # Clean dataset
    cleaned_data = clean_dataset(df)

    # Inspect cleaned dataset
    inspect_cleaned_data(cleaned_data)

    # Save cleaned dataset
    cleaned_data.to_csv("outputs/cleaned_mtsamples.csv", index=False)
    print("\nCleaned dataset saved successfully to outputs/cleaned_mtsamples.csv")

    # Preprocess text
    preprocessed_data = apply_preprocessing(cleaned_data)

    # Inspect preprocessed data
    inspect_preprocessed_data(preprocessed_data)

    # Save preprocessed dataset
    preprocessed_data.to_csv("outputs/preprocessed_mtsamples.csv", index=False)
    print("\nPreprocessed dataset saved successfully to outputs/preprocessed_mtsamples.csv")

    # ===============================
    # EDA ANALYSIS
    # ===============================

    print("\nRunning Exploratory Data Analysis...")

    plot_specialty_distribution(preprocessed_data)

    plot_text_length_distribution(preprocessed_data)

    plot_common_words(preprocessed_data)

    generate_wordcloud(preprocessed_data)

        # ===============================
    # MODEL TRAINING
    # ===============================

    print("\nStarting Machine Learning Pipeline...")

    modeling_data = prepare_top_classes(preprocessed_data, top_n=10)

    modeling_data, label_encoder = encode_target(modeling_data)

    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf = split_and_vectorize(modeling_data)

    results_df = train_and_evaluate_models(
        X_train_tfidf,
        X_test_tfidf,
        y_train,
        y_test,
        label_encoder
    )

    print("\n========== MODEL COMPARISON ==========")
    print(results_df)

    results_df.to_csv("outputs/model_comparison_results.csv", index=False)
    print("\nModel comparison results saved to outputs/model_comparison_results.csv")

        # ===============================
    # BEST MODEL INTERPRETATION + SAVING
    # ===============================

    print("\nTraining and saving the best model...")

    best_model = train_best_model(X_train_tfidf, y_train)

    save_artifacts(best_model, tfidf, label_encoder)

    plot_confusion_matrix_for_best_model(
        best_model,
        X_test_tfidf,
        y_test,
        label_encoder
    )

    save_predictions_sample(
        best_model,
        X_test_tfidf,
        y_test,
        label_encoder
    )
    # ===============================
    # CROSS-VALIDATION + HYPERPARAMETER TUNING
    # ===============================

    print("\nStarting Cross-Validation and Hyperparameter Tuning...")

    # Cross-validation on the filtered modeling dataset
    cv_results_df = run_cross_validation(modeling_data)
    cv_results_df.to_csv("outputs/cross_validation_results.csv", index=False)
    print("\nCross-validation results saved to outputs/cross_validation_results.csv")

    # For GridSearch, use raw text split (not TF-IDF matrices)
    X = modeling_data["cleaned_transcription"]
    y = modeling_data["label"]

    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tuned_grid = run_grid_search(X_train_raw, y_train_raw)

    tuned_results_df = evaluate_tuned_model(
        tuned_grid.best_estimator_,
        X_test_raw,
        y_test_raw,
        label_encoder
    )
    save_tuned_pipeline(tuned_grid.best_estimator_, label_encoder)

    tuned_results_df.to_csv("outputs/tuned_model_results.csv", index=False)
    print("\nTuned model test results saved to outputs/tuned_model_results.csv")

if __name__ == "__main__":
    main()