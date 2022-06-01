import nltk
import string
from gensim.models import Word2Vec
import numpy
import json
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import random
#nltk.download('punkt')


def get_words_of_sentence(s):
    out = s.translate(str.maketrans('', '', string.punctuation))
    return out.split()


def make_doc_matrix(training_data, window_size, embedding_size):
    sentences = []
    docs_words = dict()
    for x in range(len(training_data)):
        docs_words[x] = dict()
        docs_classes[x] = training_data[x]["category"]
        text = training_data[x]["body"] + " " + training_data[x]["title"]
        doc_sentences = nltk.tokenize.sent_tokenize(text)
        for s in doc_sentences:
            words = get_words_of_sentence(s)
            sentences.append(words)
            for w in words:
                if w in docs_words[x]:
                    docs_words[x][w] += 1
                else:
                    docs_words[x][w] = 1
    wordtovec = Word2Vec(sentences=sentences, size=embedding_size, window=window_size, min_count=1, sg=1, hs=1)
    docs_matrix = numpy.zeros((len(training_data),embedding_size))
    for x in docs_words:
        count = 0
        for w in docs_words[x]:
            count += docs_words[x][w]
            docs_matrix[x] += docs_words[x][w] * (wordtovec.wv[w])
        docs_matrix[x] /= count
    return docs_matrix


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
        new_error = calculate_error(current_u, assignment)
        if new_error == error:
            break
        error = new_error
    result = dict()
    for s in range(1,5):
        for x in assignment[s]:
            result[x] = s
    return result


def choose_data_for_tsne(number, N):
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
    plt.title("embedding = "+str(embed_size)+" window = "+str(window), loc='center')
    plt.show()


train_path = 'train.json'

with open(train_path) as training_data_file:
    train_data_whole = json.loads(training_data_file.read())
    train_data = train_data_whole[0:len(train_data_whole) // 2]


u_random_indexes = random.sample(range(0, len(train_data)), 4)

embed_window_sizes = [(100, 5), (200, 5), (100, 10), (200, 10)]
for x in embed_window_sizes:
    embed_size = x[0]
    window = x[1]
    docs_classes = dict()
    data_matrix = make_doc_matrix(train_data, window, embed_size)
    initial_u = dict()
    for i in range(4):
        initial_u[i+1] = data_matrix[u_random_indexes[i]]
    docs_class_results = k_means(initial_u, 100)
    data_for_tsne, mapping_index = choose_data_for_tsne(50, embed_size)
    show_plt(50, 200)


