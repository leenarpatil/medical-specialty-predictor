import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Clean and normalize a single text document.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Remove numbers
    4. Remove extra spaces
    5. Remove stopwords
    6. Lemmatize words
    """

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = []
    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            cleaned_words.append(lemma)

    # Join back into a single string
    cleaned_text = " ".join(cleaned_words)

    return cleaned_text


def apply_preprocessing(data):
    """
    Apply preprocessing to the transcription column.
    Creates a new column: cleaned_transcription
    """

    data = data.copy()

    data["cleaned_transcription"] = data["transcription"].apply(preprocess_text)

    return data


def inspect_preprocessed_data(data):
    """
    Show sample original and cleaned text.
    """
    print("\n========== PREPROCESSED DATA INSPECTION ==========")

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nSample original transcription:")
    print(data.loc[0, "transcription"][:500])

    print("\nSample cleaned transcription:")
    print(data.loc[0, "cleaned_transcription"][:500])

    print("\nNumber of empty cleaned texts:")
    print((data["cleaned_transcription"].str.strip() == "").sum())