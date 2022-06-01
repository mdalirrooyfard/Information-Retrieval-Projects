from elasticsearch import Elasticsearch


def search_docs(address, title_weight, title_query,  abstract_weight, abstract_query,
                year_weight, year_query, page_rank, page_rank_weight):
    es = Elasticsearch(address)
    body = {"query":{
                "function_score":{
                        "query":{
                "bool":{
                    "should":[{
                        "match":{
                            "paper.title":{
                                "query": title_query,
                                "boost": title_weight
                            }
                        }
                    },
                        {
                            "match":{
                                "paper.abstract":{
                                    "query":abstract_query,
                                    "boost": abstract_weight
                                }
                            }
                        },
                        {
                            "range":{
                                "paper.date":{
                                    "gte": year_query,
                                    "boost":year_weight
                                }
                            }
                        }
                    ]
                }
            }
                }
    }}
    if page_rank:
        body["query"]["function_score"]["field_value_factor"] = {"field":"paper.page_rank", "factor":page_rank_weight}
        body["query"]["function_score"]["boost_mode"] = "sum"
    body["_source"] = ["paper.title", "paper.abstract", "paper.authors", "paper.date"]
    res = es.search(index="paper_index", body= body)
    j = 1
    for x in res["hits"]["hits"]:
        print(str(j) + "- " + str(x["_source"]))
        j += 1



