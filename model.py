import pickle
import nltk
import random
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

# Load models
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("intents.pkl", "rb") as f:
    meta = pickle.load(f)

vectorizer = meta["vectorizer"]
classes = meta["classes"]
intents_data = meta["data"]

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    return " ".join(tokens)

def get_bot_response(user_input):
    try:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        for intent in intents_data["intents"]:
            if intent["tag"] == prediction:
                return random.choice(intent["responses"])
        return "I'm not sure how to help with that."
    except Exception as e:
        print("Error:", e)

        return "Sorry, something went wrong!"
