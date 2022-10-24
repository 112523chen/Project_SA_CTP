import streamlit as st
import pickle
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer

#store saved models into variables
model = pickle.load(open('BOW-MNB.pkl','rb'))
vectorizer = pickle.load(open('vectorizer-BOW-MNB.pkl','rb'))

#helper functions
def clean_tweets(tweet): # clean up text
    tweet = tweet.lower()
    tweet = re.sub(r'[^\w\s]', '', tweet)
    words = word_tokenize(tweet)
    tweet = " ".join([word for word in words if word not in stopwords.words('english') ])
    words = word_tokenize(tweet)
    porter = PorterStemmer()
    tweet = " ".join([ porter.stem(word) for word in words])
    return tweet

def findEmotion(text): # Find emotion behind text
    emotions = ['sadness','joy','love','anger','fear','surprise']
    text = clean_tweets(text)
    text = vectorizer.transform([text])
    idx = model.predict(text)[0]
    return emotions[idx]
    

#app below

st.set_page_config( # head tag
    page_title="Image Classification Demo", 
    page_icon="random",
    menu_items={
        'Get Help': 'https://github.com/112523chen/CTP-Fall-2022-Project',
        'Report a bug': "https://github.com/112523chen/CTP-Fall-2022-Project/issues/new",
        'About': """
                ***Streamlit app*** that predicts the emotion behind text
                """ 
    })


c1, c2, c3 = st.columns([1, 6, 1]) # establish margin
with c2:
    st.title('Emotion Analysis Model') #header tag
    text = st.text_input('Text Sample', 'Enter text here...') # message, default
    if text != 'Enter text here...' or text != None: #check if text var is an actually input 
        prediction = findEmotion(text) # finds the emotion behind the user input
        st.header(f"The text above has the {prediction} emotion behind it") #presents prediction
