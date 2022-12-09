#import libraries and helper functions
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from references.helper_functions import clean_tweets_with_lem, findEmotion, isNeutral, getColor


#store saved models into variables
model = pickle.load(open('references/models/cleaned_BoW_MNB_88.pkl','rb'))
vectorizer = pickle.load(open('references/vectorizers/BoW_vectorizer_MNB_88.pkl','rb'))


#app variables
inputTextLimit = 75


#application
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

        neturalFlag, prob_df = isNeutral(text, vectorizer, model) # finds if emotions aren't high
        prediction, prob_df = findEmotion(text, vectorizer, model) # finds the emotion behind the user input

        if neturalFlag == True:
            prediction = "neutral"
        
        if len(text) > inputTextLimit:
            st.header(f"The text has the {prediction} emotion behind it") #presents prediction
            st.write(text)
        else:
            st.header(f"The text above has the {prediction} emotion behind it") #presents prediction
        if agree:
            st.subheader("Emotion Probabilities")
            fig, ax = plt.subplots()
            ax.bar(prob_df.index, prob_df.probability, color=getColor(prob_df.index))
            ax.set_ylabel('Probabilities')
            ax.set_xlabel('Emotions')
            st.pyplot(fig)
    else:
        st.info(
                f"""
                    ⬆️ Enter text first.
                    """
            )
        st.stop()