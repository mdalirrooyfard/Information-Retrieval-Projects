import json
import string
import math
import numpy
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')


def get_words_of_sentence(s, kind):
    out = s.translate(str.maketrans('', '', string.punctuation))
    if kind == 0:
        #without context process
        return out.split()
    elif kind == 1:
        #stem
        stemmer = PorterStemmer()
        return list(map(stemmer.stem, out.split()))
    elif kind == 2:
        #lemmatization
        lemmatizer = WordNetLemmatizer()
        return list(map(lemmatizer.lemmatize, out.split()))
    else:
        #stopword removal
        stop_words = set(stopwords.words('english'))
        return [w for w in out.split() if w not in stop_words]


def make_tf_idf_docs(training_data, kind):
    docs_words_tfs = dict()
    whole_words = set()
    for x in range(len(training_data)):
        category = training_data[x]["category"]
        docs_classes[x] = category
        words = get_words_of_sentence(training_data[x]["body"]+" "+training_data[x]["title"], kind)
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


def cosine_normalize_matrix(old_matrix):
    y = numpy.zeros((old_matrix.shape[0], 1))
    for x in range(old_matrix.shape[0]):
        y[x][0] = math.sqrt(numpy.dot(old_matrix[x], numpy.transpose(old_matrix[x])))
    new_matrix = old_matrix / y
    return new_matrix


def find_closest_docs(k, method, vector):
    result = []
    if k == 1:
        if method == "max":
            result.append(numpy.argmax(vector))
        else:
            result.append(numpy.argmin(vector))
    else:
        temp = dict()
        for i in range(k):
            temp[i] = vector[i]
        if method == "max":
            for i in range(k, len(vector)):
                min_value_key = min(temp, key=temp.get)
                if vector[i] > temp[min_value_key]:
                    del temp[min_value_key]
                    temp[i] = vector[i]
        else:
            for i in range(k, len(vector)):
                max_value_key = max(temp, key=temp.get)
                if vector[i] < temp[max_value_key]:
                    del temp[max_value_key]
                    temp[i] = vector[i]
        for x in temp:
            result.append(x)
    return result


def find_most_category(nearest_docs):
    d = {1:0, 2:0, 3:0, 4:0}
    for doc in nearest_docs:
        d[docs_classes[doc]] += 1
    return max(d, key=d.get)


def knn(new_doc,method, kind):
    doc_vector = numpy.zeros((1, len(words_places)))
    words = get_words_of_sentence(new_doc["body"]+" " + new_doc["title"], kind)
    for w in words:
        if w in words_places:
            doc_vector[0][words_places[w]] += 1
    for w in words_places:
        doc_vector[0][words_places[w]] *= words_idf[w]
    if method == "cosine":
        doc_normal_vector = cosine_normalize_matrix(doc_vector)
        result = cosine_normal_matrix.dot(doc_normal_vector.transpose())
    else:
        part2 = numpy.dot(-2, numpy.dot(docs_tf_idf_matrix, doc_vector.transpose()))
        part3 = numpy.dot(doc_vector, doc_vector.transpose())
        result = part1_for_euclidean + part2 + part3
    return result


def calculate_metrics(confusion_matrix, accuracy, k):
    print("for k = ", k, ": ")
    print("accuracy: ", accuracy)
    print("confusion matrix: \n", confusion_matrix)
    F1 = 0
    for i in range(4):
        recall = confusion_matrix[i][i] / sum(confusion_matrix[i])
        print("recall for class ", i+1, ": ", recall)
        precision = confusion_matrix[i][i] / sum(confusion_matrix[j][i] for j in range(4))
        print("precision for class ", i+1, ": ", precision)
        F1 += ((2 * precision * recall) / (precision + recall))
    print("Macro averaged F1: ", F1 / 4)
    print("--------------------------------------------------------")


def find_best_k(method):
    accuracy_1 = 0
    accuracy_3 = 0
    accuracy_5 = 0
    temp = {"cosine":"max", "euclidean":"min"}
    confusion_matrix_1 = numpy.zeros((4,4))
    confusion_matrix_3 = numpy.zeros((4,4))
    confusion_matrix_5 = numpy.zeros((4,4))
    for doc in validation_data:
        result = knn(doc, method, 0)
        nearest_docs_1 = find_closest_docs(1, temp[method], result)
        nearest_docs_3 = find_closest_docs(3, temp[method], result)
        nearest_docs_5 = find_closest_docs(5, temp[method], result)
        predict_1 = find_most_category(nearest_docs_1)
        predict_3 = find_most_category(nearest_docs_3)
        predict_5 = find_most_category(nearest_docs_5)
        actual = doc["category"]
        if predict_1 == actual:
            accuracy_1 += 1
        if predict_3 == actual:
            accuracy_3 += 1
        if predict_5 == actual:
            accuracy_5 += 1
        confusion_matrix_1[actual - 1][predict_1 - 1] += 1
        confusion_matrix_3[actual - 1][predict_3 - 1] += 1
        confusion_matrix_5[actual - 1][predict_5 - 1] += 1
    print("method ", method, ": ")
    calculate_metrics(confusion_matrix_1, accuracy_1 / len(validation_data), 1)
    calculate_metrics(confusion_matrix_3, accuracy_3 / len(validation_data), 3)
    calculate_metrics(confusion_matrix_5, accuracy_5 / len(validation_data), 5)
    d = {1:accuracy_1 / len(validation_data), 3: accuracy_3 / len(validation_data), 5: accuracy_5 / len(validation_data)}
    print("the best k is ", max(d, key=d.get), "with accuracy: ", d[max(d, key=d.get)])
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    return accuracy_5 / len(validation_data)


def make_power_2_of_doc_vectors():
    res = numpy.zeros((1, len(train_data)))
    for i in range(len(train_data)):
        res[0][i] = docs_tf_idf_matrix[i].dot(docs_tf_idf_matrix[i].transpose())
    return res.transpose()


def context_process_impact(kind):
    accuracy = 0
    confusion_matrix = numpy.zeros((4,4))
    for doc in validation_data:
        results = knn(doc, "cosine", kind)
        closest_docs = find_closest_docs(5, "max", results)
        predict = find_most_category(closest_docs)
        actual = doc["category"]
        if actual == predict:
            accuracy += 1
        confusion_matrix[actual - 1][predict - 1] += 1
    accuracy = accuracy / len(validation_data)
    calculate_metrics(confusion_matrix, accuracy, 5)
    return accuracy


train_path = 'train.json'
validation_path = 'validation.json'

with open(train_path) as training_data_file:
    train_data_whole = json.loads(training_data_file.read())
    train_data = train_data_whole[0: len(train_data_whole) // 2]

with open(validation_path) as validation_file:
    validation_whole_data = json.loads(validation_file.read())
    validation_data = validation_whole_data[0:len(validation_whole_data) // 2]

# part1 of project - finding the best k
words_places = dict()
docs_classes = dict()
words_idf = dict()
docs_tf_idf_matrix = make_tf_idf_docs(train_data, 0)


cosine_normal_matrix = cosine_normalize_matrix(docs_tf_idf_matrix)
accuracy_normal = find_best_k("cosine")

part1_for_euclidean = make_power_2_of_doc_vectors()
find_best_k("euclidean")

# part3 of project - stemming impact
words_places = dict()
docs_classes = dict()
words_idf = dict()
docs_tf_idf_matrix = make_tf_idf_docs(train_data, 1)
cosine_normal_matrix = cosine_normalize_matrix(docs_tf_idf_matrix)
print("the impact of stemming: ")
accuracy_stem = context_process_impact(1)

# part3 of project - lemmatization impact
words_places = dict()
docs_classes = dict()
words_idf = dict()
docs_tf_idf_matrix = make_tf_idf_docs(train_data, 2)
cosine_normal_matrix = cosine_normalize_matrix(docs_tf_idf_matrix)
print("the impact of lemmatization: ")
accuracy_lemat = context_process_impact(2)

# part3 of project - stop words removal impact
words_places = dict()
docs_classes = dict()
words_idf = dict()
docs_tf_idf_matrix = make_tf_idf_docs(train_data, 3)
cosine_normal_matrix = cosine_normalize_matrix(docs_tf_idf_matrix)
print("the impact of stopwords removal: ")
accuracy_stopwords = context_process_impact(3)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("for k = 5 with cosine method:")
print("accuracy for normal job: ", accuracy_normal)
print("accuracy with stemming: ", accuracy_stem)
print("accuracy with lemmatization: ", accuracy_lemat)
print("accuracy with stopwords removal: ", accuracy_stopwords)

res = {"stopwords removal": accuracy_stopwords - accuracy_normal,
          "lemmatization": accuracy_lemat  - accuracy_normal,
          "stemming": accuracy_stem - accuracy_normal}

print("the most impact: ", max(res, key=res.get), " and the least impact: ",
      min(res, key=res.get))
