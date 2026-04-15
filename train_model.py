import streamlit as st
import pickle
import string
import nltk
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# -------------------- Setup --------------------
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Download NLTK resources safely
for resource in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------- Preprocessing --------------------
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]

    return " ".join(words)

# -------------------- Load Model --------------------
if not os.path.exists('vectorizer.pkl') or not os.path.exists('model.pkl'):
    st.error("❌ Model files not found. Please run train_model.py first.")
    st.stop()

with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# -------------------- UI --------------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📩 Spam Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether your message is Spam or Not using ML</div>', unsafe_allow_html=True)

st.write("---")

# Input box
input_sms = st.text_area("✉️ Enter your message here:", height=150)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict")

with col2:
    clear_btn = st.button("🧹 Clear")

# Sample messages
st.write("### 🔥 Try Sample Messages")
col3, col4 = st.columns(2)

with col3:
    if st.button("Spam Example"):
        input_sms = "Congratulations! You've won a free lottery. Click now!"

with col4:
    if st.button("Normal Example"):
        input_sms = "Hey, are we meeting tomorrow for lunch?"

# -------------------- Prediction --------------------
if clear_btn:
    st.experimental_rerun()

if predict_btn:
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message")
    else:
        try:
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            # Probability (if available)
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

            # Show probabilities
            if spam_prob is not None:
                st.write("### 📊 Confidence Score")
                st.progress(int(spam_prob))
                st.write(f"Spam: {spam_prob:.2f}%")
                st.write(f"Not Spam: {ham_prob:.2f}%")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.write("---")
st.caption("Made with ❤️ using Streamlit | ML Spam Classifier Project")
