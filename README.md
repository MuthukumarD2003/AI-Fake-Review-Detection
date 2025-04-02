# Fake Review Detector

A machine learning web application that analyzes Google Play Store app reviews to detect potentially fake or suspicious reviews.

## Overview

This Flask-based web application helps users determine the authenticity of reviews for mobile applications on the Google Play Store. It uses natural language processing and machine learning techniques to classify reviews as either genuine or fake, providing an overall assessment of review authenticity for any app.

## Features

- **Review Analysis**: Fetches and analyzes reviews from Google Play Store apps
- **Fake Review Detection**: Uses a Random Forest Classifier with TF-IDF features to identify potentially fake reviews
- **Sentiment Analysis**: Incorporates sentiment scoring to enhance detection capabilities
- **Detailed Reporting**: Provides comprehensive statistics including:
  - Percentage of fake vs. genuine reviews
  - Confidence scores for each prediction
  - Review length and sentiment metrics
  - Overall trustworthiness summary
- **Custom Model Training**: Allows users to upload their own labeled dataset to retrain the model

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Web Scraping**: BeautifulSoup, requests
- **Model Persistence**: joblib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fake-review-detector.git
   cd fake-review-detector
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `templates` directory and add an `index.html` file (sample below)

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`

3. Enter the Google Play Store app ID (found in the URL of the app's page) and click "Analyze"

4. Review the results to determine the authenticity of the app's reviews

## Sample index.html Template

Create a file named `index.html` in a `templates` directory with the following content:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Fake Review Detector</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; }
        .result-section { margin-top: 30px; }
        .review-item { 
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
        }
        .fake { background-color: #ffebee; }
        .genuine { background-color: #e8f5e9; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Fake Review Detector</h1>
        
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Analyze App Reviews</h5>
                        <form id="analyzeForm">
                            <div class="mb-3">
                                <label for="appId" class="form-label">Google Play Store App ID</label>
                                <input type="text" class="form-control" id="appId" name="app_id" 
                                    placeholder="e.g., com.whatsapp" required>
                                <div class="form-text">Find this in the URL of the app's Google Play Store page</div>
                            </div>
                            <button type="submit" class="btn btn-primary">Analyze Reviews</button>
                        </form>
                    </div>
                </div>

                <div id="loadingIndicator" class="text-center mt-4 d-none">
                    <div class="spinner-border text-primary" role="status"></div>
                    <p>Analyzing reviews... This may take a moment.</p>
                </div>

                <div id="results" class="result-section d-none">
                    <!-- Results will be displayed here -->
                </div>
            </div>

            <div class="col-md-8 mt-5">
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">Train Custom Model (Optional)</h5>
                        <form id="trainForm" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="trainingFile" class="form-label">Upload Training Data</label>
                                <input type="file" class="form-control" id="trainingFile" name="file" 
                                    accept=".csv,.xls,.xlsx" required>
                                <div class="form-text">CSV or Excel file with 'review' and 'label' columns (0=fake, 1=genuine)</div>
                            </div>
                            <button type="submit" class="btn btn-secondary">Train Model</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('analyzeForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const appId = document.getElementById('appId').value;
            const loadingIndicator = document.getElementById('loadingIndicator');
            const results = document.getElementById('results');
            
            // Show loading indicator
            loadingIndicator.classList.remove('d-none');
            results.classList.add('d-none');
            
            // Send request to backend
            fetch('/analyze', {
                method: 'POST',
                body: new FormData(this)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading indicator
                loadingIndicator.classList.add('d-none');
                
                if (data.error) {
                    results.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    results.classList.remove('d-none');
                    return;
                }
                
                // Format and display results
                let resultsHTML = `
                    <h3>Analysis Results</h3>
                    <div class="card mb-4">
                        <div class="card-body">
                            <h5 class="card-title">Summary</h5>
                            <p class="card-text">${data.summary}</p>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <p><strong>Total Reviews:</strong> ${data.total_reviews}</p>
                                    <p><strong>Fake Reviews:</strong> ${data.fake_reviews} (${data.fake_percentage}%)</p>
                                    <p><strong>Genuine Reviews:</strong> ${data.genuine_reviews} (${data.genuine_percentage}%)</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Model Accuracy:</strong> ${data.model_accuracy}%</p>
                                    <p><strong>Avg. Review Length:</strong> ${data.avg_review_length} characters</p>
                                    <p><strong>Avg. Sentiment Score:</strong> ${data.avg_sentiment}</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <h4>Review Details</h4>`;
                
                data.review_details.forEach(review => {
                    const reviewClass = review.prediction === 'Fake' ? 'fake' : 'genuine';
                    resultsHTML += `
                        <div class="review-item ${reviewClass}">
                            <div class="row">
                                <div class="col-md-9">
                                    <p>${review.review}</p>
                                </div>
                                <div class="col-md-3">
                                    <p><strong>Prediction:</strong> ${review.prediction}</p>
                                    <p><strong>Confidence:</strong> ${review.confidence}%</p>
                                    <p><strong>Sentiment:</strong> ${review.sentiment}</p>
                                </div>
                            </div>
                        </div>`;
                });
                
                results.innerHTML = resultsHTML;
                results.classList.remove('d-none');
            })
            .catch(error => {
                loadingIndicator.classList.add('d-none');
                results.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                results.classList.remove('d-none');
            });
        });
        
        document.getElementById('trainForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const loadingIndicator = document.getElementById('loadingIndicator');
            const results = document.getElementById('results');
            
            // Show loading indicator
            loadingIndicator.classList.remove('d-none');
            results.classList.add('d-none');
            
            fetch('/train', {
                method: 'POST',
                body: new FormData(this)
            })
            .then(response => response.json())
            .then(data => {
                loadingIndicator.classList.add('d-none');
                
                if (data.error) {
                    results.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                } else {
                    results.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
                }
                
                results.classList.remove('d-none');
            })
            .catch(error => {
                loadingIndicator.classList.add('d-none');
                results.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                results.classList.remove('d-none');
            });
        });
    </script>
</body>
</html>
```

## Creating a Training Dataset

To train a custom model, prepare a CSV or Excel file with these columns:
- `review`: The text content of the review
- `label`: 0 for fake reviews, 1 for genuine reviews

## How It Works

1. **Data Collection**: The app fetches reviews from the Google Play Store using web scraping
2. **Text Preprocessing**: Reviews are normalized by converting to lowercase and removing special characters
3. **Feature Extraction**: Text features are extracted using TF-IDF vectorization
4. **Classification**: A Random Forest model predicts whether each review is fake or genuine
5. **Feature Enhancement**: Additional metrics like review length and sentiment scores supplement the ML model
6. **Result Compilation**: The app aggregates all metrics and provides a comprehensive analysis

## Limitations

- The app relies on web scraping, which may break if Google changes their page structure
- Initial model training uses a small sample dataset; performance improves with custom training
- Limited to Google Play Store apps; doesn't currently support Apple App Store or other platforms

## Contributing

Contributions are welcome! Here are some ways you can help:

- Improve the ML model's accuracy
- Add support for other app stores
- Enhance the web interface
- Expand the feature set for review analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
