# single_sentiment_analysis.py
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data (if not already downloaded)
nltk.download("vader_lexicon")

def analyze_sentiment(tweet):
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(tweet)["compound"]

    # Determine sentiment label based on polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_dataset(csv):
    df = pd.read_csv(csv, header=None)
    df["sentiment"] = df[5].apply(analyze_sentiment)  # Assuming tweet text is in the 6th column (index 5)
    return df

if __name__ == "__main__":
    csv_file = "C:/Users/Abhi/Downloads/testdata.manual.2009.06.14.csv"  # Replace with the correct CSV file path
    df = analyze_dataset(csv_file)

    # Calculate the count of each sentiment
    sentiment_counts = df["sentiment"].value_counts()

    # Plot the sentiment distribution using a bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("Sentiment Analysis Results")
    plt.show()
