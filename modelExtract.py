import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
import pickle

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('spam uci.csv', encoding="latin1")
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.columns = ['target', 'text']
df['target'] = LabelEncoder().fit_transform(df['target'])
df = df.drop_duplicates(keep='first')

ps = PorterStemmer()
stopWords = set(stopwords.words('english'))
punctuation = set(string.punctuation)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    tmp = []
    for i in text:
        if i.isalnum() and i not in stopWords and i not in punctuation:
            tmp.append(ps.stem(i))
    return " ".join(tmp)

df['newText'] = df['text'].apply(transform_text)
tfv = TfidfVectorizer(max_features=2500)
X = tfv.fit_transform(df['newText']).toarray()
y = df['target'].values
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=2)

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
tfvgnb = GaussianNB()
tfvmnb = MultinomialNB()
tfvbnb = BernoulliNB()

gnb.fit(X, y)
mnb.fit(X, y)
bnb.fit(X, y)
tfvgnb.fit(X, y)
tfvmnb.fit(X, y)
tfvbnb.fit(X, y)

pickle.dump(gnb, open('gnb.pkl', 'wb'))
pickle.dump(mnb, open('mnb.pkl', 'wb'))
pickle.dump(bnb, open('bnb.pkl', 'wb'))
pickle.dump(tfvgnb, open('tfvgnb.pkl', 'wb'))
pickle.dump(tfvmnb, open('tfvmnb.pkl', 'wb'))
pickle.dump(tfvbnb, open('tfvbnb.pkl', 'wb'))
pickle.dump(tfv, open('tfv.pkl', 'wb'))
