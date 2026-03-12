import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Load tuned pipeline and label encoder
pipeline = joblib.load("best_tuned_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Access fitted TF-IDF and classifier inside pipeline
tfidf = pipeline.named_steps["tfidf"]
clf = pipeline.named_steps["clf"]

# Preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)

    return " ".join(cleaned_words)


def get_top_word_contributions(cleaned_text, predicted_class_index, top_n=10):
    """
    Show which words contributed most to the predicted class.
    Works for linear models like Logistic Regression.
    """
    X_vec = tfidf.transform([cleaned_text])

    feature_names = np.array(tfidf.get_feature_names_out())
    coef = clf.coef_[predicted_class_index]

    # contribution = tfidf_value * class_weight
    contributions = X_vec.toarray()[0] * coef

    nonzero_idx = np.where(X_vec.toarray()[0] > 0)[0]

    contrib_df = pd.DataFrame({
        "feature": feature_names[nonzero_idx],
        "contribution": contributions[nonzero_idx]
    })

    contrib_df = contrib_df.sort_values(by="contribution", ascending=False).head(top_n)

    return contrib_df


# Page config
st.set_page_config(page_title="Medical Specialty Predictor", layout="wide")

st.title("Medical Specialty Predictor")
st.caption("Educational research prototype. Do not enter real patient data.")

st.write("Paste a medical transcription note below and predict its specialty.")

user_input = st.text_area("Enter medical report text:", height=250)

if st.button("Predict Specialty"):
    if user_input.strip() == "":
        st.warning("Please enter some medical text first.")
    else:
        cleaned_text = preprocess_text(user_input)

        # Predict using full pipeline
        prediction = pipeline.predict([cleaned_text])[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]

        st.success(f"Predicted Specialty: {predicted_label}")

        # Probabilities
        if hasattr(pipeline.named_steps["clf"], "predict_proba"):
            probabilities = pipeline.predict_proba([cleaned_text])[0]

            prob_df = pd.DataFrame({
                "Specialty": label_encoder.classes_,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)

            st.subheader("Prediction Confidence")
            st.dataframe(prob_df, use_container_width=True)

            # Probability bar chart
            st.subheader("Probability Chart")
            top_probs = prob_df.head(10).sort_values(by="Probability", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_probs["Specialty"], top_probs["Probability"])
            ax.set_xlabel("Probability")
            ax.set_ylabel("Specialty")
            ax.set_title("Top Predicted Specialties")
            plt.tight_layout()

            st.pyplot(fig)

        # Top word contributions
        st.subheader("Top Keywords Behind the Prediction")

        contrib_df = get_top_word_contributions(cleaned_text, prediction, top_n=10)

        if contrib_df.empty:
            st.info("No meaningful keyword contributions found.")
        else:
            st.dataframe(contrib_df, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            plot_df = contrib_df.sort_values(by="contribution", ascending=True)

            ax2.barh(plot_df["feature"], plot_df["contribution"])
            ax2.set_xlabel("Contribution Score")
            ax2.set_ylabel("Keyword")
            ax2.set_title("Top Positive Keyword Contributions")
            plt.tight_layout()


            st.pyplot(fig2)
