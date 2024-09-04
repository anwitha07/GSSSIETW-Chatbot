# imports
import random
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and download necessary NLTK data
lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# Load intents JSON file
with open("intents.json") as file:
    intents = json.load(file)

# Initialize lists
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Process intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and sort words and classes
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to pickle files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training data
for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word.lower()) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert training data to NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Extract train_x and train_y
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("model created")
