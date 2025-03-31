# import pandas as pd
# import re
# import joblib
# import requests
# from flask import Flask, request, jsonify, render_template
# from google_play_scraper import reviews
#
# # Load trained model and vectorizer
# model = joblib.load('fake_review_model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')
#
# app = Flask(__name__)
#
#
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
#     return text
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/analyze', methods=['POST'])
# def analyze():
#     app_id = request.form['app_id']
#     result, _ = reviews(app_id, lang='en', country='us', count=100)
#
#     total_reviews = len(result)
#     fake_reviews = 0
#     genuine_reviews = 0
#
#     for review in result:
#         processed_review = preprocess_text(review['content'])
#         review_vector = vectorizer.transform([processed_review])
#         prediction = model.predict(review_vector)[0]
#         if prediction == 0:
#             fake_reviews += 1
#         else:
#             genuine_reviews += 1
#
#     fake_percentage = (fake_reviews / total_reviews) * 100 if total_reviews > 0 else 0
#     genuine_percentage = (genuine_reviews / total_reviews) * 100 if total_reviews > 0 else 0
#     summary_text = "Mainly promotional" if fake_percentage > 50 else "Likely genuine"
#
#     return render_template('result.html', total_reviews=total_reviews, fake_reviews=fake_reviews,
#                            genuine_reviews=genuine_reviews, fake_percentage=fake_percentage,
#                            genuine_percentage=genuine_percentage, summary_text=summary_text)
#
#
# if __name__ == '__main__':
#     app.run(debug=True)


import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset (Ensure you have review_dataset.csv in the same folder)
data = pd.read_csv('review_dataset.csv')

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text

data['processed_review'] = data['review'].apply(preprocess_text)

# Convert text to features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['processed_review'])
y = data['label']  # 0 for fake, 1 for genuine

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate and save the model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and vectorizer
joblib.dump(model, 'fake_review_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!")

