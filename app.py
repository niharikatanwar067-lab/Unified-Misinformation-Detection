import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
import pickle

nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)


# Load vectorizer and model
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Misinformation Detector", layout="centered")

st.title("📰 Misinformation Detection System")
st.write("Enter a news/article text and check whether it is **Real** or **Misinformation**.")

user_input = st.text_area("Paste the text here:", height=200)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_text = tfidf.transform([cleaned_input])
        
        # prediction = model.predict(vectorized_text)[0]
        # prob = model.predict_proba(vectorized_text)[0]
        prediction = model.predict(vectorized_text)[0]
        prob = model.predict_proba(vectorized_text)[0]
        confidence = max(prob) * 100

        # Label mapping: 0 = REAL, 1 = MISINFORMATION
        # if prediction == 0:
        #     st.success(f"✅ This text is likely REAL ({confidence:.2f}% confidence)")
        # else:
        #     st.error(f"❌ This text is likely MISINFORMATION ({confidence:.2f}% confidence)")

        
        fake_prob = prob[0]   # label 0 = MISINFORMATION
        real_prob = prob[1]   # label 1 = REAL

        if fake_prob > 0.65:
            st.error(f"❌ This text is likely MISINFORMATION ({fake_prob*100:.2f}% confidence)")
        elif real_prob > 0.65:
            st.success(f"✅ This text is likely REAL ({real_prob*100:.2f}% confidence)")
        else:
            st.warning("⚠️ The model is uncertain about this text.")

    # st.write("Raw prediction:", prediction)
    # st.write("Probabilities:", prob)



# To run the app, use the command:
# streamlit run app.py