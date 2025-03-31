import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template
import sqlite3

# Load dataset (Ensure you have this dataset in the correct location)
data = pd.read_csv('review_dataset.csv')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

data['processed_review'] = data['review'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_review'])
y = data['label']  # 0 for fake, 1 for genuine

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save model and vectorizer
joblib.dump(model, 'fake_review_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Flask Web App
app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('reviews.db')
    conn.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, review TEXT, label TEXT)")
    return conn

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_review():
    review = request.form['review']
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('fake_review_model.pkl')
    processed_review = preprocess_text(review)
    review_vector = vectorizer.transform([processed_review])
    prediction = model.predict(review_vector)[0]
    label = 'Genuine' if prediction == 1 else 'Fake'

    # Store in database
    conn = get_db_connection()
    conn.execute("INSERT INTO reviews (review, label) VALUES (?, ?)", (review, label))
    conn.commit()
    conn.close()

    return jsonify({'review': review, 'prediction': label})

@app.route('/summary')
def summary():
    conn = get_db_connection()
    reviews = conn.execute("SELECT * FROM reviews").fetchall()
    conn.close()

    total_reviews = len(reviews)
    fake_reviews = sum(1 for r in reviews if r[2] == 'Fake')
    genuine_reviews = total_reviews - fake_reviews
    fake_percentage = (fake_reviews / total_reviews) * 100 if total_reviews > 0 else 0
    genuine_percentage = (genuine_reviews / total_reviews) * 100 if total_reviews > 0 else 0
    summary_text = "Mainly promotional" if fake_percentage > 50 else "Likely genuine"

    return jsonify({
        'total_reviews': total_reviews,
        'fake_reviews': fake_reviews,
        'fake_percentage': fake_percentage,
        'genuine_reviews': genuine_reviews,
        'genuine_percentage': genuine_percentage,
        'accuracy': accuracy * 100,
        'summary': summary_text
    })

if __name__ == '__main__':
    app.run(debug=True)
