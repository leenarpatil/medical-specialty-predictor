import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud


def plot_specialty_distribution(data):
    plt.figure(figsize=(12, 6))

    top_specialties = data["medical_specialty"].value_counts().head(15)

    sns.barplot(
        x=top_specialties.values,
        y=top_specialties.index
    )

    plt.title("Top 15 Medical Specialties in Dataset")
    plt.xlabel("Number of Reports")
    plt.ylabel("Specialty")
    plt.tight_layout()

    plt.savefig("outputs/specialty_distribution.png")
    plt.show()


def plot_text_length_distribution(data):
    text_lengths = data["cleaned_transcription"].apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=50)

    plt.title("Distribution of Report Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig("outputs/text_length_distribution.png")
    plt.show()


def plot_common_words(data):
    all_words = " ".join(data["cleaned_transcription"]).split()
    word_counts = Counter(all_words)

    common_words = word_counts.most_common(20)

    words = [w[0] for w in common_words]
    counts = [w[1] for w in common_words]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts, y=words)

    plt.title("Top 20 Most Frequent Words in Medical Reports")
    plt.xlabel("Count")
    plt.ylabel("Word")
    plt.tight_layout()

    plt.savefig("outputs/top_words.png")
    plt.show()


def generate_wordcloud(data):
    text = " ".join(data["cleaned_transcription"])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("outputs/wordcloud.png")
    plt.show()