import os

from flask import render_template
import nltk
from nltk.stem import WordNetLemmatizer
from train_chatbot import modelCreating

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model

# loading created chatbot model
model = load_model('chatbot_model.h5')
model._make_predict_function()
import json
import random

intents = json.loads(open('intents.json').read())
# words.pkl file having work list ["'s", ',', 'a', 'adverse',
# 'all', 'anyone', 'are', 'awesome', 'be', 'behavior', 'blood', 'by', 'bye', 'can', 'causing',
# 'chatting', 'check', 'could', 'data', 'day', 'detail', 'do', 'dont', 'drug', 'entry', 'find', 'for', 'give', 'good', 'goodbye', 'have', 'hello', 'help', 'helpful', 'helping', 'hey', 'hi', 'history', 'hola', 'hospital', 'how', 'i', 'id', 'is', 'later', 'list', 'load', 'locate', 'log', 'looking', 'lookup', 'management', 'me', 'module', 'nearby', 'next', 'nice', 'of', 'offered', 'open', 'patient', 'pharmacy', 'pressure', 'provide', 'reaction', 'related', 'result', 'search', 'searching', 'see', 'show', 'suitable', 'support', 'task', 'thank', 'thanks', 'that', 'there', 'till', 'time', 'to', 'transfer', 'up', 'want', 'what', 'which', 'with', 'you']
words = pickle.load(open('words.pkl', 'rb'))
# laoding tag ['activity', 'bad_joke', 'doing_badly', 'doing_great', 'good_joke', 'goodbye', 'greeting', 'hate', 'how_are_you', 'joke', 'like', 'name', 'no', 'noanswer', 'options', 'real_bot', 'technical_training', 'thanks', 'your_thoughts']
classes = pickle.load(open('classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    model._make_predict_function()
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)

    res = getResponse(ints, intents)
    print(res)
    if res == "technical training":
        return render_template("botReply.html")
    elif res == "activity":
        return render_template("Iteam.html")
    return res


modelCreating()


def clearlogs():
    os.remove("logo.txt")
    os.remove("updatedata.txt")
    text_file = open("logo.txt", "a")
    updatedata1 = open("updatedata.txt", "a")
    updatedata1.write("Date;Role;Text")
    text_file.write("Date;Role;Text")
# chatbot_response("")
# chatbot_response("hi")
# chatbot_response("hi there")
