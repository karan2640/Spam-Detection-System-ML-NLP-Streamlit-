import streamlit as st
import pickle
import string
import nltk
import os
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

if "history" not in st.session_state:
    st.session_state.history = []

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]

    return " ".join(words)


if not os.path.exists('vectorizer.pkl') or not os.path.exists('model.pkl'):
    st.error("❌ Model files not found. Run train_model.py first.")
    st.stop()

with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def clear_text():
    st.session_state.input_sms = ""

def set_spam_example():
    st.session_state.input_sms = "Congratulations! You've won a free lottery. Click now!"

def set_normal_example():
    st.session_state.input_sms = "Hey, are we meeting tomorrow?"


with st.sidebar:
    st.title("📊 Spam Classifier")
    st.write("ML + NLP Project")
    st.info("Detect Spam Messages using ML")


st.markdown("<h1 style='text-align:center;color:#4CAF50;'>📩 Spam Detection System</h1>", unsafe_allow_html=True)

st.write("---")


input_sms = st.text_area(
    "✉️ Enter your message here:",
    height=150,
    key="input_sms"
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", disabled=not input_sms.strip())

with col2:
    st.button("🧹 Clear", on_click=clear_text)

st.write("### 🔥 Try Sample Messages")
col3, col4 = st.columns(2)

with col3:
    st.button("Spam Example", on_click=set_spam_example)

with col4:
    st.button("Normal Example", on_click=set_normal_example)


if predict_btn:
    try:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]


        try:
            prob = model.predict_proba(vector_input)[0]
            spam_prob = prob[1] * 100
            ham_prob = prob[0] * 100
        except:
            spam_prob, ham_prob = None, None

        st.write("---")

        if result == 1:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ NOT SPAM")


        st.session_state.history.append((input_sms, result))

        # Confidence
        if spam_prob is not None:
            st.write("### 📊 Confidence Score")
            st.progress(int(spam_prob))
            st.write(f"Spam: {spam_prob:.2f}%")
            st.write(f"Not Spam: {ham_prob:.2f}%")

            df = pd.DataFrame({
                "Label": ["Spam", "Not Spam"],
                "Probability": [spam_prob, ham_prob]
            })
            st.bar_chart(df.set_index("Label"))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


st.write("---")
st.write("### 🕒 Prediction History")

for msg, res in st.session_state.history[-5:][::-1]:
    label = "🚨 Spam" if res == 1 else "✅ Not Spam"
    st.write(f"{msg[:40]}... → {label}")


st.write("---")
st.caption("Made with ❤️ using Streamlit")
