from flask import Flask, render_template, jsonify, request
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

app = Flask(__name__)

# Download NLTK data (if not already downloaded)
nltk.download("vader_lexicon")

def analyze_sentiment(tweet):
    sia = SentimentIntensityAnalyzer()
    polarity = sia.polarity_scores(tweet)["compound"]

    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_dataset(csv):
    df = pd.read_csv(csv, header=None)
    df["sentiment"] = df[5].apply(analyze_sentiment)
    return df

# Replace this with the correct path to your CSV file
csv_file = "C:/Users/Abhi/Downloads/testdata.manual.2009.06.14.csv"
df = analyze_dataset(csv_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment_data')
def get_sentiment_data():
    # Calculate the count of each sentiment
    sentiment_counts = df["sentiment"].value_counts()

    # Convert the sentiment_counts to a JSON format
    sentiment_data = [{"sentiment": sentiment, "count": count} for sentiment, count in sentiment_counts.items()]
    return jsonify(sentiment_data)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = analyze_sentiment(text)
        return render_template('prediction.html', text=text, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
