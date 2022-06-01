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
        # ans = []
        # for w, tag in pos_tag(out.split()):
        #     d = {"J":wordnet.ADJ, "N":wordnet.NOUN, "V":wordnet.VERB, "R":wordnet.ADV}
        #     if tag[0] in d:
        #         ans.append(lemmatizer.lemmatize(w, d[tag[0]]))
        #     else:
        #         ans.append(lemmatizer.lemmatize(w))
        # return ans
        return list(map(lemmatizer.lemmatize, out.split()))
    else:
        #stopword removal
        stop_words = set(stopwords.words('english'))
        return [w for w in out.split() if w not in stop_words]


def calculate_tfs(training_data, kind):
    for doc in training_data:
        words = get_words_of_sentence(doc["body"]+" "+doc["title"], kind)
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


def naive_bayes(new_doc, alpha, kind):
    words = get_words_of_sentence(new_doc["body"]+" "+new_doc["title"], kind)
    result = {1:0, 2:0, 3:0, 4:0}
    for i in range(1,5):
        result[i] += math.log10(classes_prior_numbers[i] / N)
    for w in words:
        if w in vocab:
            for i in range(1,5):
                if w in words_in_class[i]:
                    result[i] += math.log10((words_in_class[i][w] + alpha) / (number_of_words_in_class[i] + (alpha * len(vocab))))
                else:
                    result[i] += math.log10(alpha / (number_of_words_in_class[i] + (alpha * len(vocab))))
    return max(result, key=result.get)


def calculate_metrics(confusion_matrix, accuracy):
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
    print("------------------------------------------------------")


def find_best_alpha():
    accuracies = {0.1:0, 1:0, 10:0}
    for x in accuracies:
        print("alpha = ", x, ": ")
        confusion_matrix = numpy.zeros((4,4))
        for doc in validation_data:
            predict = naive_bayes(doc, x, 0)
            actual = doc["category"]
            if actual == predict:
                accuracies[x] += 1
            confusion_matrix[actual - 1][predict - 1] += 1
        accuracies[x] = accuracies[x] / len(validation_data)
        calculate_metrics(confusion_matrix, accuracies[x])
    best = max(accuracies, key=accuracies.get)
    print("the best alpha: ", best, " with accuracy: ", accuracies[best])
    return accuracies[0.1]


def context_process_impact(kind):
    accuracy = 0
    confusion_matrix = numpy.zeros((4,4))
    for doc in validation_data:
        predict = naive_bayes(doc, 0.1, kind)
        actual = doc["category"]
        if actual == predict:
            accuracy += 1
        confusion_matrix[actual - 1][predict - 1] += 1
    accuracy = accuracy / len(validation_data)
    calculate_metrics(confusion_matrix, accuracy)
    return accuracy


train_path = 'train.json'
validation_path = 'validation.json'

with open(train_path) as training_data_file:
    train_data = json.loads(training_data_file.read())
with open(validation_path) as validation_file:
    validation_data = json.loads(validation_file.read())

# part 2 of project - finding the best alpha
words_in_class = {1:dict(), 2:dict(), 3:dict(), 4:dict()}
classes_prior_numbers = {1:0, 2:0, 3:0, 4:0}
number_of_words_in_class = {1:0, 2:0, 3:0, 4:0}
vocab = set()
N = len(train_data)
calculate_tfs(train_data, 0)
accuracy_normal = find_best_alpha()

# part 3 of project - stemming impact
words_in_class = {1:dict(), 2:dict(), 3:dict(), 4:dict()}
classes_prior_numbers = {1:0, 2:0, 3:0, 4:0}
number_of_words_in_class = {1:0, 2:0, 3:0, 4:0}
vocab = set()
calculate_tfs(train_data, 1)
print("impact of stemming: ")
accuracy_stem = context_process_impact(1)

# part 3 of project - lemmatization impact
words_in_class = {1:dict(), 2:dict(), 3:dict(), 4:dict()}
classes_prior_numbers = {1:0, 2:0, 3:0, 4:0}
number_of_words_in_class = {1:0, 2:0, 3:0, 4:0}
vocab = set()
calculate_tfs(train_data, 2)
print("impact of lemmatization: ")
accuracy_lemat = context_process_impact(2)

# part 3 of project - stop words removal impact
words_in_class = {1:dict(), 2:dict(), 3:dict(), 4:dict()}
classes_prior_numbers = {1:0, 2:0, 3:0, 4:0}
number_of_words_in_class = {1:0, 2:0, 3:0, 4:0}
vocab = set()
calculate_tfs(train_data, 3)
print("impact of stopwords removal: ")
accuracy_stopwords = context_process_impact(3)

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("for naive bayes with alpha = 0.1:")
print("accuracy with normal job: ", accuracy_normal)
print("accuracy with stemming: ", accuracy_stem)
print("accuracy with lemmatization: ", accuracy_lemat)
print("accuracy with stopwords removal: ", accuracy_stopwords)

res = {"stopwords removal": accuracy_stopwords - accuracy_normal,
          "lemmatization": accuracy_lemat  - accuracy_normal,
          "stemming": accuracy_stem - accuracy_normal}

print("the best impact: ", max(res, key=res.get), " and the worst impact: ",
      min(res, key=res.get))
