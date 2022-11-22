#import libraries and helper functions
import streamlit as st
import pandas as pd
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from references.helper_functions import clean_tweets_with_lem, findEmotion


#store saved models into variables
model = pickle.load(open('references/models/cleaned_NLP_BoW_SGDC_model_92.pkl','rb'))
vectorizer = pickle.load(open('references/vectorizers/cleaned_NLP_BoW_SGDC_vectorizer_92.pkl','rb'))


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

##############################################################
#! probability estimates are not available for loss='hinge'
# agree = st.sidebar.checkbox('See Probability Estimates')

c1, c2, c3 = st.columns([1, 6, 1]) # establish margin
with c2:
    st.title('Emotion Analysis Model') #header tag
    text = st.text_input('Text Sample', '') # message, default
    if text is not '': #check if text var is an actually input 
        prediction = findEmotion(text, vectorizer, model) # finds the emotion behind the user input
        if len(text) > inputTextLimit:
            st.header(f"The text has the {prediction} emotion behind it") #presents prediction
            st.write(text)
        else:
            st.header(f"The text above has the {prediction} emotion behind it") #presents prediction
        # if agree:
        #     st.subheader("Emotion Probabilities")
        #     st.table(prob_df)
    else:
        st.info(
                f"""
                    ⬆️ Enter text first.
                    """
            )
        st.stop()