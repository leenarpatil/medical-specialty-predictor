import os
import joblib


def save_tuned_pipeline(best_estimator, label_encoder, output_dir="outputs"):
    """
    Save the tuned GridSearch best estimator and label encoder.
    """

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(best_estimator, os.path.join(output_dir, "best_tuned_pipeline.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    print("\nSaved tuned model artifacts:")
    print("- outputs/best_tuned_pipeline.pkl")
    print("- outputs/label_encoder.pkl")