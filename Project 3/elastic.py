from elasticsearch import Elasticsearch
import json


def store(address, json_file):
    es = Elasticsearch(address)
    body = {"mappings":
                {"properties":{
                    "paper":{
                        "properties":{
                                "id":{"type":"keyword"},
                                "title":{"type":"text"},
                                "authors":{"type":"text"},
                                "date":{"type":"text"},
                                "abstract":{"type":"text"},
                                "references":{"type":"keyword"}
                        }
                    }
                }},
    }

    es.indices.create("paper_index", body=body)
    s = json.loads(json_file)
    for x in s:
        a = dict()
        a["paper"] = x
        d = json.dumps(a)
        es.index(index="paper_index", body=d)


def delete(address):
    es = Elasticsearch(address)
    es.indices.delete(index='paper_index', ignore=[400, 404])
