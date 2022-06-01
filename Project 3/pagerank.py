from elasticsearch import Elasticsearch, helpers
import json
import numpy


def calculate_page_rank(address, alpha):
    es = Elasticsearch(address)
    res = helpers.scan(es, query={"query":{"match_all":{}}}, index = "paper_index")
    id_to_matrix = dict()
    t = 0
    for x in res:
        id_to_matrix[x["_source"]["paper"]["id"]] = t
        t += 1
    probabilities = numpy.array([1/t for r in range(t)])
    matrix = numpy.zeros((t,t))
    res = helpers.scan(es, query={"query":{"match_all":{}}}, index = "paper_index")
    for x in res:
        row = id_to_matrix[x["_source"]["paper"]["id"]]
        z = 0
        for y in x["_source"]["paper"]["references"]:
            if y in id_to_matrix:
                z += 1
        for y in x["_source"]["paper"]["references"]:
            if y in id_to_matrix:
                col = id_to_matrix[y]
                matrix[row][col] = 1 / z
        if z == 0:
            for a in range(t):
                matrix[row][a] = 1 / t
    for a in range(t):
        for b in range(t):
            matrix[a][b] = ((1 - alpha)*matrix[a][b]) + (alpha / t)
    not_stop = True
    while not_stop:
        new_probabilities = numpy.dot(probabilities, matrix)
        if numpy.linalg.norm(new_probabilities - probabilities) < 0.001:
            not_stop = False
        probabilities = new_probabilities
    for x in id_to_matrix:
        res1 = es.search(body={"query": {"match": {"paper.id":x}}}, index = 'paper_index')
        page_rank = probabilities[id_to_matrix[x]]
        es.update(index="paper_index", id=res1['hits']['hits'][0]['_id'], body={"doc":{"paper":{"page_rank":page_rank}}}, refresh=True)

