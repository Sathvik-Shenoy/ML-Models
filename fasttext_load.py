import fasttext
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load NLTK stopwords
nltk.download('stopwords')
stopword = set(stopwords.words('english'))

# Initialize NLTK SnowballStemmer
stemmer = nltk.SnowballStemmer("english")

df = pd.read_csv("C:\\Users\\Admin\\ML-Models\\hate_with_tweet_new.csv")

# Load SpaCy model
nlp = spacy.load("en_core_web_lg", disable=["tagger"])

# Load the FastText model
fasttext_model_path = "fasttext_model.bin"
fasttext_model = fasttext.load_model(fasttext_model_path)

# Load the ensemble model
svm_model_path = "rf_model.joblib"
svm_model = joblib.load(svm_model_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet_new'], df['labels'], test_size=0.2, random_state=42)

# Initialize label encoder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

def preprocess(text):
    test_list = text.split()
    text = ""
    for word in test_list:
        if word.endswith("ing"):
            word = word[:-3]
        text += word
        text += " "
    doc = nlp(text)
    filtered_tokens = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

def clean(text):
    text = str(text).lower()
    text = re.sub('[.?]', '', text)
    text = re.sub('https?://\S+|www.\S+', '', text)
    text = re.sub('<.?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

def predict_label(input_text):
    # Preprocess the user input
    user_input_processed = preprocess(input_text)

    # Transform the preprocessed input using the FastText model to get embeddings
    user_input_embedding = fasttext_model.get_sentence_vector(user_input_processed)

    # Make prediction using the ensemble model
    prediction = svm_model.predict([user_input_embedding])

    # Encode the predicted label back to its original form
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    return predicted_label

output = predict_label("hi sis")
print(output)
