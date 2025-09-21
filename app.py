import os
import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter

# Use local nltk_data folder
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_dir)

stopWords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
ps = PorterStemmer()

# Load models
tfv = pickle.load(open('tfv.pkl', 'rb'))
gnb = pickle.load(open('gnb.pkl', 'rb'))
mnb = pickle.load(open('mnb.pkl', 'rb'))
bnb = pickle.load(open('bnb.pkl', 'rb'))
tfvgnb = pickle.load(open('tfvgnb.pkl', 'rb'))
tfvmnb = pickle.load(open('tfvmnb.pkl', 'rb'))
tfvbnb = pickle.load(open('tfvbnb.pkl', 'rb'))

def transformText(text):
    text = text.lower()
    text = word_tokenize(text)
    tmp = []
    for i in text:
        if i.isalnum() and i not in stopWords and i not in punctuation:
            tmp.append(ps.stem(i))
    return " ".join(tmp)

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="centered")
st.title("📧 Email Spam Detector")
st.markdown("Enter the text of an email below to check whether it is **Spam** or **Not Spam**.")

email_input = st.text_area("Enter Email:", height=150)

if st.button("Predict"):
    if not email_input.strip():
        st.warning("Please enter an email message to check!")
    else:
        cleaned_text = transformText(email_input)
        vectorized_text = tfv.transform([cleaned_text])

        # Convert dense for GaussianNB
        vectorized_dense = vectorized_text.toarray()

        predictions = [
            gnb.predict(vectorized_dense)[0],
            mnb.predict(vectorized_dense)[0],
            bnb.predict(vectorized_dense)[0],
            tfvgnb.predict(vectorized_text)[0],
            tfvmnb.predict(vectorized_text)[0],
            tfvbnb.predict(vectorized_text)[0]
        ]

        final_prediction = Counter(predictions).most_common(1)[0][0]

        if final_prediction == 1:
            st.markdown("<h2 style='color:red;'>Spam ⚠️</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Not Spam ✅</h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
