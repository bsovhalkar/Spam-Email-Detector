# import streamlit as st
# import pickle 
# import nltk
# tfv = pickle.load(open('tfv.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email Spam Detector")





# def tranformText(text):
#     import string
#     from nltk.stem.porter import PorterStemmer
#     punctuation = string.punctuation
#     from nltk.corpus import stopwords
#     nltk.download('stopwords')
#     stopWords = stopwords.words('english')
#     ps = PorterStemmer()
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     tmp = []
#     for i in text:
#         if i.isalnum() and i not in stopWords and i not in punctuation:
#             i = ps.stem(i)
#             tmp.append(i)
#     return " ".join(tmp)



# input = st.text_input("Enter the Email to check ")
# newTxt = tranformText(input)

# if st.button('Predict'):

#     transTxt = tfv.transform([newTxt])

#     ans = model.predict(transTxt)[0]

#     if ans==1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")







import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

stopWords = set(stopwords.words('english'))
punctuation = set(string.punctuation)
ps = PorterStemmer()

tfv = pickle.load(open('tfv.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

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
    if email_input.strip() == "":
        st.warning("Please enter an email message to check!")
    else:
        cleaned_text = transformText(email_input)
        vectorized_text = tfv.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:
            st.markdown("<h2 style='color:red;'>Spam ⚠️</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Not Spam ✅</h2>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
