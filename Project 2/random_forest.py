import numpy
import string
import math
import json
from sklearn.ensemble import RandomForestClassifier


def get_words_of_sentence(s):
    out = s.translate(str.maketrans('', '', string.punctuation))
    return out.split()


def make_tf_idf_docs(training_data):
    docs_words_tfs = dict()
    whole_words = set()
    for x in range(len(training_data)):
        category = training_data[x]["category"]
        train_docs_classes.append(category)
        words = get_words_of_sentence(training_data[x]["body"]+" "+training_data[x]["title"])
        docs_words_tfs[x] = dict()
        for w in words:
            whole_words.add(w)
            if w in docs_words_tfs[x]:
                docs_words_tfs[x][w] += 1
            else:
                docs_words_tfs[x][w] = 1
        for w in docs_words_tfs[x]:
            if w in words_idf:
                words_idf[w] += 1
            else:
                words_idf[w] = 1
    for w in words_idf:
        words_idf[w] = math.log10(len(training_data) / words_idf[w])
    j = 0
    for w in whole_words:
        words_places[w] = j
        j += 1
    docs_matrix = numpy.zeros((len(training_data), len(whole_words)))
    for x in docs_words_tfs:
        for w in docs_words_tfs[x]:
            docs_matrix[x][words_places[w]] = docs_words_tfs[x][w] * words_idf[w]
    return docs_matrix


def make_test_matrix(valid_data):
    docs_words_tfs = dict()
    test_docs_matrix = numpy.zeros((len(valid_data), len(words_places)))
    for i in range(len(valid_data)):
        words = get_words_of_sentence(valid_data[i]["body"]+" "+valid_data[i]["title"])
        docs_words_tfs[i] = dict()
        for w in words:
            if w in words_places:
                if w in docs_words_tfs[i]:
                    docs_words_tfs[i][w] += 1
                else:
                    docs_words_tfs[i][w] = 1
        test_docs_classes.append(valid_data[i]["category"])
    for x in docs_words_tfs:
        for w in docs_words_tfs[x]:
            test_docs_matrix[x][words_places[w]] = docs_words_tfs[x][w] * words_idf[w]
    return test_docs_matrix


def random_forest(number_of_trees, depth):
    rfc = RandomForestClassifier(n_estimators=number_of_trees, max_depth=depth)
    rfc.fit(train_matrix, train_docs_classes)
    predict = rfc.predict(test_matrix)
    accuracy = 0
    for i in range(len(validation_data)):
        if predict[i] == test_docs_classes[i]:
            accuracy += 1
    accuracy /= len(validation_data)
    return accuracy


train_path = 'train.json'
validation_path = 'validation.json'

with open(train_path) as training_data_file:
    train_data = json.loads(training_data_file.read())
with open(validation_path) as validation_file:
    validation_data = json.loads(validation_file.read())

words_places = dict()
train_docs_classes = []
words_idf = dict()
train_matrix = make_tf_idf_docs(train_data)
test_docs_classes = []
test_matrix = make_test_matrix(validation_data)

d = {(5, 10):0, (5, 100):0, (10, 10):0, (10, 100):0}

for x in d:
    d[x] = random_forest(x[0], x[1])
    print("for number of trees: ", x[0], " and depth: ", x[1], ", accuracy: ", d[x])
best = max(d, key=d.get)
print("the best number of trees: ", best[0], " and depth: ", best[1], " with accuracy: ", d[best])
