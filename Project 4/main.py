from sklearn import svm
import math
from statistics import mean
import numpy


def make_data(address):
    file = open(address, "r")
    lines = file.readlines()
    vectors = []
    vector_index_to_doc_query = dict()
    query_to_docs = dict()
    for line in lines:
        elements = line.split()
        qid = elements[1].split(":")[1]
        relevance = int(elements[0])
        doc_id = elements[50]
        doc_vector = []
        for i in range(2, 48):
            doc_vector.append(float(elements[i].split(":")[1]))
        if qid not in query_to_docs:
            query_to_docs[qid] = dict()
        query_to_docs[qid][doc_id] = [doc_vector, relevance]
    counter = 0
    y = []
    for q in query_to_docs:
        keys = list(query_to_docs[q].keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                doc1 = keys[i]
                doc2 = keys[j]
                vector_difference = numpy.array(query_to_docs[q][doc1][0]) - numpy.array(query_to_docs[q][doc2][0])
                temp = query_to_docs[q][doc1][1] - query_to_docs[q][doc2][1]
                if temp > 0:
                    score = 1
                elif temp == 0:
                    score = 0
                else:
                    score = -1
                vector_index_to_doc_query[counter] = (doc1, doc2, q)
                vectors.append(vector_difference)
                y.append(score)
                counter += 1
    return vectors, y, vector_index_to_doc_query, query_to_docs


def correct_arrangement(query_to_docs):
    results = dict()
    for q in query_to_docs:
        x = dict()
        for doc in query_to_docs[q]:
            x[doc] = query_to_docs[q][doc][1]
        x_sort = [(k,v) for k,v in sorted(x.items(), key=lambda item:(item[1], item[0]))]
        x_sort.reverse()
        results[q] = x_sort
    return results


def predicted_arrangement(vector_index_to_doc_query, scores):
    query_to_docs = dict()
    for i in range(len(scores)):
        x = vector_index_to_doc_query[i]
        if x[2] not in query_to_docs:
            query_to_docs[x[2]] = dict()
        query_to_docs[x[2]][(x[0], x[1])] = scores[i]
    results = dict()
    for q in query_to_docs:
        docs = dict()
        for pair in query_to_docs[q]:
            if pair[0] not in docs:
                docs[pair[0]] = 0
            if pair[1] not in docs:
                docs[pair[1]] = 0
            if query_to_docs[q][pair] == 0:
                if pair[0] > pair[1]:
                    docs[pair[0]] += 1
                else:
                    docs[pair[1]] += 1
            elif query_to_docs[q][pair] == 1:
                docs[pair[0]] += 1
            else:
                docs[pair[1]] += 1
        temp = [k for k,v in sorted(docs.items(), key=lambda item:item[1])]
        temp.reverse()
        results[q] = temp
    return results


def find_best_c():
    train_vectors, train_y, vector_index_to_doc_query, train_query_to_docs = make_data("data/train.txt")
    valid_vectors, valid_y, valid_vector_index_to_doc_query, valid_query_to_docs = make_data("data/vali.txt")
    possible_c_ndcg = {0.01:0,0.1:0, 1:0}
    possible_c_svm = {0.01:0, 0.1:0,  1:0}
    correct_query_arrangement = correct_arrangement(valid_query_to_docs)
    max_dcg = dict()
    for q in correct_query_arrangement:
        dcg = correct_query_arrangement[q][0][1]
        length = len(correct_query_arrangement[q])
        for i in range(1,min(5, length)):
            dcg += (correct_query_arrangement[q][i][1]/ math.log2(i+1))
        max_dcg[q] = dcg
    for c in possible_c_ndcg:
        classifier = svm.LinearSVC(C=c, max_iter=4000)
        classifier.fit(train_vectors, train_y)
        possible_c_svm[c] = classifier
        prediction_y = classifier.predict(valid_vectors)
        predicted_query_arrangement = predicted_arrangement(valid_vector_index_to_doc_query, prediction_y)
        predict_ndcg = dict()
        for q in predicted_query_arrangement:
            doc = predicted_query_arrangement[q][0]
            dcg = valid_query_to_docs[q][doc][1]
            length = len(predicted_query_arrangement[q])
            for i in range(1,min(5, length)):
                doc = predicted_query_arrangement[q][i]
                dcg += valid_query_to_docs[q][doc][1]
            if max_dcg[q] != 0:
                predict_ndcg[q] = dcg / max_dcg[q]
            else:
                predict_ndcg[q] = 1
        possible_c_ndcg[c] = mean(predict_ndcg.values())
    print("ndcg for different c:")
    print(possible_c_ndcg)
    best_c = max(possible_c_ndcg, key=possible_c_ndcg.get)
    print("The best c is "+ str(best_c))
    return possible_c_svm[best_c]


def test(classifier):
    test_vectors, test_y, test_index_to_doc_query, test_query_to_docs = make_data("data/test.txt")
    correct_test_arrangement = correct_arrangement(test_query_to_docs)
    max_dcg = dict()
    for q in correct_test_arrangement:
        dcg = correct_test_arrangement[q][0][1]
        length = len(correct_test_arrangement[q])
        for i in range(1,min(5, length)):
            dcg += (correct_test_arrangement[q][i][1]/ math.log2(i+1))
        max_dcg[q] = dcg
    prediction_y = classifier.predict(test_vectors)
    predicted_query_arrangement = predicted_arrangement(test_index_to_doc_query, prediction_y)
    predict_ndcg = dict()
    for q in predicted_query_arrangement:
        doc = predicted_query_arrangement[q][0]
        dcg = test_query_to_docs[q][doc][1]
        length = len(predicted_query_arrangement[q])
        for i in range(1,min(5, length)):
            doc = predicted_query_arrangement[q][i]
            dcg += test_query_to_docs[q][doc][1]
        if max_dcg[q] != 0:
            predict_ndcg[q] = dcg / max_dcg[q]
        else:
            predict_ndcg[q] = 1
    print("ndcg@5 for test: ", str(mean(predict_ndcg.values())))


best_classifier = find_best_c()
test(best_classifier)

