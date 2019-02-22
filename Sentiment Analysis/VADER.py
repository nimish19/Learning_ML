#Sentiment Analysis using VADER(Valence Aware Dictionary and Sentiment Reasoner)
#VADER is a lexicon and rule-based sentiment analysis tool 
#that is specifically attuned to sentiments expressed in social media

import pandas as pd

df = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

import re
import nltk 
nltk.download("punkt")
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

NEGATE =["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

#Cleaning the given reviews and storing them in corpus
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word in NEGATE or not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# checking polarity of Sentences in Corpus
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
nltk.download('vader_lexicon')
for sentence in corpus:
    vs = analyzer.polarity_scores(sentence)
    if vs['compound'] > 0.5 :
        print('POSITIVE')
    elif vs['compound'] <-0.5 :
        print('NEGATIVE')
    else :
        print('NEUTRAL')
    print("{:-<65} {}".format(sentence, str(vs)))
