#import libraries
import re
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


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

def findEmotion(text, vectorizer, model): # Find emotion behind text
    emotions = ['anger','fear','joy','love','sadness','surprise']
    text = vectorizer.transform([clean_tweets_with_lem(text)])
    prediction = model.predict(text)[0]
    prob_df = pd.DataFrame( { "emotion": emotions, "probability": model.predict_proba(text)[0] } )
    prob_df.set_index('emotion', inplace=True)
    prob_df.sort_values(['probability'], inplace=True, ascending=False)
    return prediction, prob_df

def isNeutral(text, vectorizer, model):
    roundLimit = 3
    neturalLimit = 0.01

    emotions = ['anger','fear','joy','love','sadness','surprise']
    text = vectorizer.transform([clean_tweets_with_lem(text)])
    prediction = model.predict(text)[0]
    prob_df = pd.DataFrame( { "emotion": emotions, "probability": model.predict_proba(text)[0] } )
    prob_1, prob_2, prob_3, prob_4, prob_5, prob_6 = model.predict_proba(text)[0]

    prob_1 = round(prob_1, roundLimit)
    prob_2 = round(prob_2, roundLimit)
    prob_3 = round(prob_3, roundLimit)
    prob_4 = round(prob_4, roundLimit)
    prob_5 = round(prob_5, roundLimit)
    prob_6 = round(prob_6, roundLimit)

    # print(prediction == prob_df["emotion"].iloc[0])

    std = np.std([prob_1, prob_2, prob_3, prob_4, prob_5, prob_6])
    
    if std < neturalLimit:
        return True, prob_df
    
    return False, prob_df

def getColor(index):
    key = {
        'anger': 'red',
        'love': 'pink',
        'sadness': 'blue',
        'surprise': 'green',
        'joy': '#e6e600',
        'fear': 'purple',
    }
    stack = []
    for i in index:
        stack.append(key[i])
    return stack