# Required libraries
import random
import numpy as np
import pickle
import json
import tensorflow as tf
from flask import Flask, render_template, request
import os
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load data files
model_path = "chatbot_model.h5"
intents_path = "intents.json"
words_path = "words.pkl"
classes_path = "classes.pkl"

# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
if not os.path.exists(intents_path):
    raise FileNotFoundError(f"Intents file '{intents_path}' not found.")
if not os.path.exists(words_path):
    raise FileNotFoundError(f"Words file '{words_path}' not found.")
if not os.path.exists(classes_path):
    raise FileNotFoundError(f"Classes file '{classes_path}' not found.")

# Chat initialization
model = tf.keras.models.load_model(model_path)
with open(intents_path, 'r') as file:
    intents = json.load(file)
words = pickle.load(open(words_path, "rb"))
classes = pickle.load(open(classes_path, "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    if msg.startswith('my name is'):
        name = msg[11:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    elif msg.startswith('hi my name is'):
        name = msg[14:]
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
        res = getResponse(ints, intents)
    return res

# Chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

if __name__ == "__main__":
    app.run(debug=True)
