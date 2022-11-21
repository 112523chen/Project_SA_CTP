import streamlit as st
import pickle
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('wordnet')


#store saved models into variables - Best model cleaned BOW and MNB
model = pickle.load(open('references/models/cleaned_BoW_MNB_88.pkl','rb'))
vectorizer = pickle.load(open('references/vectorizers/BoW_vectorizer_MNB_88.pkl','rb'))


#app variables
inputTextLimit = 75


#helper functions
def clean_tweets_with_lem(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    words = word_tokenize(tweet)
    tweet = " ".join([word for word in words if word not in stopwords.words('english') ])
    words = word_tokenize(tweet)
    lem = WordNetLemmatizer()
    tweet = " ".join([ lem.lemmatize(word) for word in words])
    return tweet

def clean_tweets_with_stem(tweet): # clean up text
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    words = word_tokenize(tweet)
    tweet = " ".join([word for word in words if word not in stopwords.words('english') ])
    words = word_tokenize(tweet)
    porter = PorterStemmer()
    tweet = " ".join([ porter.stem(word) for word in words])
    return tweet

def clean_tweets_without_nlp(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    words = word_tokenize(tweet)
    tweet = " ".join([ word for word in words])
    return tweet

def findEmotion(text): # Find emotion behind text
    emotions = ['anger','fear','joy','love','sadness','surprise']
    text = vectorizer.transform([clean_tweets_with_lem(text)])
    prediction= model.predict(text)[0]
    d = { "emotion": emotions, "probability": model.predict_proba(text)[0] }
    df = pd.DataFrame(d)
    df = df.set_index('emotion')
    return prediction, df


#app below
st.set_page_config( # head tag
    page_title="Emotion Prediction Model Demo", 
    page_icon="📝",
    layout="centered",
    menu_items={
        'Get Help': 'https://github.com/112523chen/CTP-Fall-2022-Project',
        'Report a bug': "https://github.com/112523chen/CTP-Fall-2022-Project/issues/new",
        'About': """
                ***Streamlit app*** that predicts the emotion behind text
                """ 
    })

agree = st.sidebar.checkbox('See Probability Estimates')

c1, c2, c3 = st.columns([1, 6, 1]) # establish margin
with c2:
    st.title('Emotion Analysis Model') #header tag
    text = st.text_input('Text Sample', '') # message, default
    if text is not '': #check if text var is an actually input 
        prediction = findEmotion(text)[0] # finds the emotion behind the user input
        prediction_df = findEmotion(text)[1]
        if len(text) > inputTextLimit:
            st.header(f"The text has the {prediction} emotion behind it") #presents prediction
            st.write(text)
        else:
            st.header(f"The text above has the {prediction} emotion behind it") #presents prediction
        if agree:
            st.subheader("Emotion Probabilities")
            st.table(prediction_df)
    else:
        st.info(
                f"""
                    ⬆️ Enter text first.
                    """
            )
        st.stop()