from elasticsearch import Elasticsearch, helpers
import numpy
import heapq


def best_authors(address, number):
    es = Elasticsearch(address)
    res = helpers.scan(es, query={"query":{"match_all":{}}}, index = "paper_index")
    id_to_authors = dict()
    authors_to_index = dict()
    index_to_authors = dict()
    j = 0
    for x in res:
        id_to_authors[x["_source"]["paper"]["id"]] = x["_source"]["paper"]["authors"]
        for author in x["_source"]["paper"]["authors"]:
            if author not in authors_to_index:
                authors_to_index[author] = j
                index_to_authors[j] = author
                j += 1
    adjacency_matrix = numpy.zeros((len(authors_to_index), len(authors_to_index)))
    res = helpers.scan(es, query={"query":{"match_all":{}}}, index = "paper_index")
    for x in res:
        for author in x["_source"]["paper"]["authors"]:
            row = authors_to_index[author]
            for reference in x["_source"]["paper"]["references"]:
                if reference in id_to_authors:
                    for c in id_to_authors[reference]:
                        col = authors_to_index[c]
                        adjacency_matrix[row][col] = 1
    # initialize
    hubness = numpy.array([1 for r in range(len(authors_to_index))])
    authority = numpy.array([1 for r in range(len(authors_to_index))])
    for i in range(5):
        hubness = numpy.dot(adjacency_matrix, authority.transpose())
        authority = numpy.dot(adjacency_matrix.transpose(), hubness.transpose())
        h_norm = numpy.linalg.norm(hubness)
        hubness /= h_norm
        a_norm = numpy.linalg.norm(authority)
        authority /= a_norm
    heap = []
    for i in range(len(authority)):
        heapq.heappush(heap, (-authority[i], index_to_authors[i]))
    for j in range(number):
        x = heapq.heappop(heap)
        print(str(j+1), "- author: ", x[1], ", authority: ", str(-x[0]))


