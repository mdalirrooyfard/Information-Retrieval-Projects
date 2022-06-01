from typing import List, Dict
import string
import math
import numpy
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')


def get_words_of_sentence(s):
    out = s.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    return [w for w in out.split() if w not in stop_words]


def calculate_tfs(training_data):
    for doc in training_data:
        words = get_words_of_sentence(doc["body"]+" "+doc["title"])
        category = doc["category"]
        classes_prior_numbers[category] += 1
        for w in words:
            vocab.add(w)
            if w in words_in_class[category]:
                words_in_class[category][w] += 1
            else:
                words_in_class[category][w] = 1
    for i in range(1,5):
        number_of_words_in_class[i] = sum(words_in_class[i].values())


def train(training_docs: List[Dict]):
    for doc in training_docs:
        words = get_words_of_sentence(doc["body"]+" "+doc["title"])
        category = doc["category"]
        classes_prior_numbers[category] += 1
        for w in words:
            vocab.add(w)
            if w in words_in_class[category]:
                words_in_class[category][w] += 1
            else:
                words_in_class[category][w] = 1
    for i in range(1,5):
        number_of_words_in_class[i] = sum(words_in_class[i].values())


def classify(doc: Dict) -> int:
    words = get_words_of_sentence(doc["body"]+" "+doc["title"])
    result = {1:0, 2:0, 3:0, 4:0}
    alpha = 0.1
    n = sum(classes_prior_numbers[r] for r in range(1,5))
    for i in range(1,5):
        result[i] += math.log10(classes_prior_numbers[i] / n)
    for w in words:
        if w in vocab:
            for i in range(1,5):
                if w in words_in_class[i]:
                    result[i] += math.log10((words_in_class[i][w] + alpha) / (number_of_words_in_class[i] + (alpha * len(vocab))))
                else:
                    result[i] += math.log10(alpha / (number_of_words_in_class[i] + (alpha * len(vocab))))
    return max(result, key=result.get)



words_in_class = {1:dict(), 2:dict(), 3:dict(), 4:dict()}
classes_prior_numbers = {1:0, 2:0, 3:0, 4:0}
number_of_words_in_class = {1:0, 2:0, 3:0, 4:0}
vocab = set()
