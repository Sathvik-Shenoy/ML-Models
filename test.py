#Importing the packages
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import nltk
import re,string
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword=set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

#read the dataset with name "Fake_Real_Data.csv" and store it in a variable df
df = pd.read_csv("C:\\Users\\Admin\\Downloads\\twitter_data.csv")

#print the shape of dataframe
print(df.shape)

#print top 5 rows
df.head(5)
nlp = spacy.load("en_core_web_sm")
df["labels"] = df['class'].map({0: "Hate Speech", 1: "Offensive Speech", 2: "No Hate and Offensive Speech"})
df = df[["tweet", "labels"]]
def preprocess(text):
        test_list = text.split()
        text = ""
        for word in test_list:
            if word.endswith("ing"):
                word=word[:-3]
            text+=word
            text+=" "
        doc = nlp(text)
        filtered_tokens = []
        
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_tokens.append(token.lemma_)
            
        return " ".join(filtered_tokens)

df["tweet_new"] = df.tweet.apply(preprocess)

def clean(text):
    text = str (text). lower()
    text = re. sub('[.?]', '', text)
    text = re. sub('https?://\S+|www.\S+', '', text)
    text = re. sub('<.?>+', '', text)
    text = re. sub('[%s]' % re. escape(string. punctuation), '', text)
    text = re. sub('\n', '', text)
    text = re. sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ". join(text)
    text = [stemmer. stem(word) for word in text. split(' ')]
    text=" ". join(text)
    return text

df["tweet_new"] = df.tweet_new.apply(clean)

from sklearn.feature_extraction.text import TfidfVectorizer
x = np. array(df["tweet_new"])
y = np. array(df["labels"])
cv = TfidfVectorizer()
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#Model building
model = DecisionTreeClassifier()
#Training the model
model.fit(X_train,y_train)

import joblib
joblib.dump(model, 'trained_model.joblib')
joblib.dump(cv, 'fitted_vectorizer.joblib')

inp = "This is an example tweet that you want to classify."
inp = preprocess(inp)
print(inp)

with open('input_file.txt', 'w') as file:
    file.write(str(inp))