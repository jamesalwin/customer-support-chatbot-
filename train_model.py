import json
import nltk
import random
import joblib
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

# Load the intents JSON file
with open('intents.json') as f:
    data = json.load(f)

lemmatizer = WordNetLemmatizer()

# Prepare training data
corpus = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern.lower())
        lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
        corpus.append(' '.join(lemmatized))
        labels.append(intent['tag'])

# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Train the model
model = LinearSVC()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("✅ Model trained and saved as chatbot_model.pkl, vectorizer.pkl, label_encoder.pkl")
