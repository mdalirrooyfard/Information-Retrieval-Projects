import string
import json
import math
import numpy
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def get_words_of_sentence(s):
    out = s.translate(str.maketrans('', '', string.punctuation))
    return out.split()


def make_tf_idf_docs(data):
    docs_words_tfs = dict()
    whole_words = set()
    for x in range(len(data)):
        category = data[x]["category"]
        docs_classes[x] = category
        words = get_words_of_sentence(data[x]["body"] + " " + data[x]["title"])
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
        words_idf[w] = math.log10(len(data) / words_idf[w])
    j = 0
    for w in whole_words:
        words_places[w] = j
        j += 1
    docs_matrix = numpy.zeros((len(data), len(whole_words)))
    for x in docs_words_tfs:
        for w in docs_words_tfs[x]:
            docs_matrix[x][words_places[w]] = docs_words_tfs[x][w] * words_idf[w]
    return docs_matrix, len(whole_words)


def calculate_error(u, assignment):
    res = 0
    for j in assignment:
        for x in assignment[j]:
            res += math.pow(numpy.linalg.norm(data_matrix[x] - u[j]),2)
    return res


def k_means(current_u, max_iter):
    error = 0
    assignment = dict()
    for s in range(max_iter):
        assignment = {1:set(), 2:set(), 3:set(), 4:set()}
        for x in range(len(train_data)):
            distance = {r: numpy.linalg.norm(data_matrix[x] - current_u[r]) for r in range(1,5)}
            assignment[min(distance, key=distance.get)].add(x)
        for j in range(1,5):
            current_u[j] = sum(data_matrix[x] for x in assignment[j]) / len(assignment[j])
            norm = numpy.linalg.norm(current_u[j])
            current_u[j] /= norm
        new_error = calculate_error(current_u, assignment)
        if new_error == error:
            break
        error = new_error
    result = dict()
    for s in range(1,5):
        for x in assignment[s]:
            result[x] = s
    return result


def choose_data_for_tsne(number):
    count = {1:0, 2:0, 3:0, 4:0}
    mapping_index = dict()
    result = numpy.zeros((4 * number, N))
    s = 0
    x = 0
    while s < 4*number:
        if count[docs_classes[x]] < number:
            count[docs_classes[x]] += 1
            result[s] = data_matrix[x]
            mapping_index[s] = x
            s += 1
        x += 1
    return result, mapping_index


def show_plt(perplexity, length):
    embedded_data = TSNE(perplexity=perplexity).fit_transform(data_for_tsne)
    x = [embedded_data[r][0] for r in range(length)]
    y = [embedded_data[r][1] for r in range(length)]
    label_real = [docs_classes[mapping_index[r]] for r in range(length)]
    label_predict = [docs_class_results[mapping_index[r]] for r in range(length)]
    colors = {1:"blue", 2:"green", 3:"red", 4:"purple"}
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    g = sns.scatterplot(x, y, hue=label_real, palette=colors, ax=ax1)
    g.set(xlabel="real groups")
    g2 = sns.scatterplot(x,y, hue=label_predict, palette=colors, ax=ax2)
    g2.set(xlabel="predicted groups")
    plt.show()


def cosine_normalize_matrix(old_matrix):
    y = numpy.zeros((old_matrix.shape[0], 1))
    for x in range(old_matrix.shape[0]):
        y[x][0] = math.sqrt(numpy.dot(old_matrix[x], numpy.transpose(old_matrix[x])))
    new_matrix = old_matrix / y
    return new_matrix


train_path = 'train.json'

with open(train_path) as training_data_file:
    train_data_whole = json.loads(training_data_file.read())
    train_data = train_data_whole[0:len(train_data_whole) // 2]

words_places = dict()
docs_classes = dict()
words_idf = dict()
docs_tf_idf_matrix, N = make_tf_idf_docs(train_data)
data_matrix = cosine_normalize_matrix(docs_tf_idf_matrix)
u_random_indexes = random.sample(range(0, len(train_data)), 4)
initial_u = dict()
for i in range(4):
    initial_u[i+1] = data_matrix[u_random_indexes[i]]
docs_class_results = k_means(initial_u, 100)
data_for_tsne, mapping_index = choose_data_for_tsne(50)
show_plt(50, 200)
