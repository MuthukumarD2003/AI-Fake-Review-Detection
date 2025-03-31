# Fake Product Review Detection System

## 📌 Project Overview
This project is a **Fake Product Review Detection System** that automatically fetches reviews from the **Google Play Store**, analyzes them using **AI & Machine Learning**, and provides a verdict on whether an app is trustworthy.

### ✨ Features
- Scrapes app reviews from **Google Play Store**
- Uses **Machine Learning (Naïve Bayes)** for fake review detection
- Shows the following statistics:
  - ✅ **Total number of reviews**
  - 🚫 **Number and percentage of fake reviews**
  - 🏆 **Number and percentage of real reviews**
  - 🎯 **Model Accuracy**
  - 📊 **Final Summary & Recommendation**
- Web-based interface using **Flask & HTML/CSS**

## 🏗️ Tech Stack
- **Frontend**: HTML, CSS
- **Backend**: Python (Flask)
- **Machine Learning**: Scikit-learn (Naïve Bayes, TF-IDF Vectorizer)
- **Database**: SQLite (for storing review data)
- **Scraping**: google-play-scraper

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MuthukumarD2003/AI-Fake-Review-Detection.git
cd AI-Fake-Review-Detection
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model
```bash
python train_model.py
```
This will:
- Train the model on the dataset
- Save the trained model (`fake_review_model.pkl`)
- Save the vectorizer (`vectorizer.pkl`)

### 5️⃣ Run the Flask App
```bash
python app.py
```
Access the web interface at: **http://127.0.0.1:5000**

## 📌 Usage
1. Enter the **Google Play Store App ID** (e.g., `com.whatsapp`)
2. Click **Analyze**
3. View results:
   - Percentage of fake vs real reviews
   - Model accuracy
   - Final recommendation: ✅ "People can try this app" OR 🚫 "This app is not recommended"

## 📝 File Structure
```
📂 AI-Fake-Review-Detection
├── 📂 static             # CSS, JS, Images
├── 📂 templates          # HTML Files
│   ├── index.html       # Home Page
│   ├── result.html      # Result Page
├── app.py               # Flask Backend
├── train_model.py       # ML Model Training Script
├── requirements.txt     # Dependencies
├── fake_review_model.pkl  # Trained Model
├── vectorizer.pkl       # TF-IDF Vectorizer
├── reviews.db           # SQLite Database
└── README.md            # Documentation (This File)
```

## 🛠️ Troubleshooting
- If `ModuleNotFoundError`, run:
  ```bash
  pip install -r requirements.txt
  ```
- If **Flask App Doesn't Run**:
  ```bash
  export FLASK_APP=app.py   # On macOS/Linux
  set FLASK_APP=app.py      # On Windows
  python app.py
  ```

## 👨‍💻 Author
- **Muthukumar D**  
- 📧 Email: dmuthukumar13@gmail.com  
- 🌐 GitHub: [MuthukumarD2003](https://github.com/MuthukumarD2003)

