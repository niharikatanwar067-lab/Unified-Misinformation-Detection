# Unified Misinformation Detection System

## 📌 Project Overview

The **Unified Misinformation Detection System** is an AI/ML-based application designed to classify textual content as **Real**, **Misinformation**, or **Uncertain**.  
The system addresses modern challenges such as fake news, misleading online content, and ambiguity in AI-generated or partially verified text.

It combines traditional Natural Language Processing techniques with supervised machine learning and provides an interactive web interface using **Streamlit**.

---

## 🎯 Objectives

- Detect misinformation in textual content
- Handle ambiguous cases using confidence-based uncertainty thresholds
- Demonstrate real-world ML deployment with an interactive UI
- Highlight limitations of binary classification in real information ecosystems

---

## 🧠 Machine Learning Approach

### Data

- Labeled news dataset containing **real (label = 1)** and **fake (label = 0)** samples
- Text preprocessing includes:
  - Lowercasing
  - Removal of punctuation and stopwords
  - Token normalization

### Feature Engineering

- **TF-IDF Vectorization**
  - Converts text into numerical features based on term importance

### Model

- **Logistic Regression**
  - Selected for interpretability and probabilistic output
  - Enables confidence-based prediction logic

---

## ⚙️ Prediction Logic

Instead of strict binary classification, the system introduces an **UNCERTAIN** category based on prediction confidence:

| Probability Score | Output         |
| ----------------- | -------------- |
| > 0.65            | REAL           |
| < 0.35            | MISINFORMATION |
| 0.35 – 0.65       | UNCERTAIN      |

This approach reflects real-world ambiguity and avoids overconfident predictions.

---

## 🖥️ Application Interface

The application is built using **Streamlit**, allowing users to:

- Enter or paste text input
- Receive instant classification results
- View prediction confidence scores
- Understand uncertain cases clearly

---

## 📂 Project Structure

Unified-Misinformation-Detection/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── screenshots/
│ ├── real_prediction.png
│ ├── fake_prediction.png
│ └── uncertain_prediction.png
└── dataset/
└── news.csv

---

## 🚀 How to Run Locally

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Git & GitHub

---

## 📌 Key Learnings

- Confidence calibration in machine learning models
- Handling ambiguity in real-world text classification
- Limitations of supervised learning for misinformation detection
- End-to-end ML project deployment

---

## 🔮 Future Enhancements

- Deep learning models (LSTM, Transformers)
- AI-generated text detection
- Multi-source credibility analysis
- Live news API integration
- Model explainability using SHAP or LIME

---

## 👤 Author

**Niharika**  
AI / Machine Learning Project
