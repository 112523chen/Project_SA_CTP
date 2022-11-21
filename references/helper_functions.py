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
    prob_df = pd.DataFrame( { "emotion": emotions, "probability": model.predict_proba(text)[0] } )
    prob_df.set_index('emotion', inplace=True)
    prob_df.sort_values(['probability'], inplace=True, ascending=False)
    return prediction, prob_df 