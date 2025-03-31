# # import requests
# # import re
# # import joblib
# # import pandas as pd
# # from flask import Flask, request, jsonify, render_template
# # from google_play_scraper import reviews
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.naive_bayes import MultinomialNB
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score
# #
# # # Load dataset
# # try:
# #     data = pd.read_csv('review_dataset.csv')
# # except FileNotFoundError:
# #     print("Dataset not found! Please provide 'review_dataset.csv'.")
# #     exit()
# #
# # # Preprocess text
# # def preprocess_text(text):
# #     text = text.lower()
# #     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
# #     return text
# #
# # data['processed_review'] = data['review'].apply(preprocess_text)
# #
# # # Feature extraction using TF-IDF
# # vectorizer = TfidfVectorizer(max_features=5000)
# # X = vectorizer.fit_transform(data['processed_review'])
# # y = data['label']  # 0 for fake, 1 for genuine
# #
# # # Train/Test split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # # Train the model
# # model = MultinomialNB()
# # model.fit(X_train, y_train)
# #
# # # Evaluate model
# # accuracy = accuracy_score(y_test, model.predict(X_test))
# # print(f'Model Accuracy: {accuracy * 100:.2f}%')
# #
# # # Save model and vectorizer
# # joblib.dump(model, 'fake_review_model.pkl')
# # joblib.dump(vectorizer, 'vectorizer.pkl')
# #
# # # Flask App
# # app = Flask(__name__)
# #
# # @app.route('/')
# # def home():
# #     return render_template('index.html')
# #
# # @app.route('/analyze', methods=['POST'])
# # def analyze_app():
# #     app_id = request.form['app_id']
# #     fetched_reviews, _ = reviews(app_id, count=100)
# #     model = joblib.load('fake_review_model.pkl')
# #     vectorizer = joblib.load('vectorizer.pkl')
# #
# #     fake_count = 0
# #     total_reviews = len(fetched_reviews)
# #
# #     for review in fetched_reviews:
# #         processed_review = preprocess_text(review['content'])
# #         review_vector = vectorizer.transform([processed_review])
# #         prediction = model.predict(review_vector)[0]
# #         if prediction == 0:
# #             fake_count += 1
# #
# #     fake_percentage = (fake_count / total_reviews) * 100 if total_reviews > 0 else 0
# #     genuine_percentage = 100 - fake_percentage
# #     summary = "This app has {:.2f}% fake and {:.2f}% genuine reviews.".format(fake_percentage, genuine_percentage)
# #     recommendation = "This app is not recommended." if fake_percentage > 50 else "People can try this app."
# #
# #     return jsonify({
# #         'total_reviews': total_reviews,
# #         'fake_reviews': fake_count,
# #         'fake_percentage': fake_percentage,
# #         'genuine_percentage': genuine_percentage,
# #         'accuracy': accuracy * 100,
# #         'summary': summary,
# #         'recommendation': recommendation
# #     })
# #
# # if __name__ == '__main__':
# #     app.run(debug=True)
#
#
# import pandas as pd
# import re
# import joblib
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
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text
#
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
#
# @app.route('/analyze', methods=['POST'])
# def analyze_app():
#     app_id = request.form['app_id']
#     try:
#         result, _ = reviews(
#             app_id,
#             lang='en',
#             country='in',
#             count=100  # Fetch latest 100 reviews
#         )
#
#         review_texts = [r['content'] for r in result]
#         processed_reviews = [preprocess_text(r) for r in review_texts]
#         review_vectors = vectorizer.transform(processed_reviews)
#         predictions = model.predict(review_vectors)
#
#         total_reviews = len(predictions)
#         fake_reviews = sum(1 for p in predictions if p == 0)
#         genuine_reviews = total_reviews - fake_reviews
#         fake_percentage = (fake_reviews / total_reviews) * 100 if total_reviews > 0 else 0
#         genuine_percentage = (genuine_reviews / total_reviews) * 100 if total_reviews > 0 else 0
#         summary = "This app is not recommended." if fake_percentage > 50 else "People can try this app."
#
#         return jsonify({
#             'total_reviews': total_reviews,
#             'fake_reviews': fake_reviews,
#             'fake_percentage': fake_percentage,
#             'genuine_reviews': genuine_reviews,
#             'genuine_percentage': genuine_percentage,
#             'accuracy': 95.0,  # Placeholder accuracy
#             'summary': summary
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#


from flask import Flask, request, jsonify, render_template
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import joblib
import re

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# Function to fetch reviews from Google Play Store using Selenium
def fetch_reviews(app_id):
    url = f"https://play.google.com/store/apps/details?id={app_id}&hl=en&gl=IN"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (no UI)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    sleep(5)  # Allow time for the page to load

    try:
        reviews = driver.find_elements(By.CLASS_NAME, "h3YV2d")  # Extract reviews
        review_texts = [review.text for review in reviews[:20]]  # Get first 20 reviews
        driver.quit()
        return review_texts
    except Exception as e:
        driver.quit()
        return []


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/analyze', methods=['POST'])
def analyze():
    app_id = request.form['app_id']
    reviews = fetch_reviews(app_id)

    if not reviews:
        return jsonify({"error": "No reviews found or unable to fetch."})

    processed_reviews = [preprocess_text(review) for review in reviews]
    review_vectors = vectorizer.transform(processed_reviews)
    predictions = model.predict(review_vectors)

    fake_count = sum(1 for p in predictions if p == 0)
    genuine_count = len(predictions) - fake_count
    fake_percentage = (fake_count / len(predictions)) * 100
    genuine_percentage = 100 - fake_percentage
    accuracy = 95  # Assuming model accuracy is 95% (you can change this)

    # Generate a summary statement
    summary = "This app is not recommended." if fake_percentage > 50 else "People can try this app."

    return jsonify({
        "total_reviews": len(predictions),
        "fake_reviews": fake_count,
        "fake_percentage": round(fake_percentage, 2),
        "genuine_reviews": genuine_count,
        "genuine_percentage": round(genuine_percentage, 2),
        "accuracy": accuracy,
        "summary": summary
    })


if __name__ == '__main__':
    app.run(debug=True)
