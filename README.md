# Information-Retrieval-Projects

These are Projects for Sharif University of Techonology Modern Information Retrieval Course.

## Project 1:
This is an Information Retrieval System for Persian Wikipedia.

Details of Project:
  - Initial Data Processing including Text Unification, splitting words of each sentence, omitting punctuation marks and stemming words.
  - Constructing Positional and Bigram Indexes.
  - Implementing functions to add or delete a document from indexes.
  - Building functions to save indexes to a file and load them to prevent system overwork. 
  - Query spelling correction function.
  - Document Retrieval in tf-idf vector space with ltn-lnn and ltc-lnc scoring approaches.
  - Phrasal search.
  - Detailed search based on the title and the text of the documents.
  - System Evaluation with following metrics: MAP, F-measure, R-Precision, NDCG.
  - There is also a test jupiter file.
 

## Project 2
This is a Text Data Classification and Clustering Project.

Algorithms for Classification:
  - kNN with the following distance metrics: cosine similarity, euclidean distance
  - Naive Bayes with smoothing
  - Random Forest
  - SVM with linear kernel


The "K-Means" algorithm was implemented for the Clustering part. The clustering part was done with two vector systems: tf-idf and word2vec.

Additional Details:

  - Analyzing the influence of text pre-processing techniques such as stopword removal, lemmatization and stemming

  - Evaluating the results by the following metrics: Recall, Precision, Accuracy, Confusion Matrix, Macro averaged F

  - Analyzing the results by t-SNE approach.
  

## Project 3
This Project is an Article Crawl and Search system.
Project Details:
  - Building an article crawler for semanticscholar.org
  - Building an Index for the above articles with ElasticSearch.
  - Calculating Page Rank for the above articles and adding that to ElasticSearch.
  - Execute Weighted Search Queries on the data.
  - Finding the best writers by implemeting the "HITS" algorithm.


## Project 4
This is a supervised learning project to find the relation between queries and documents.

The implemented algorithm is "Ranking SVM".

The results on the test data is evaluated by the NDCG metric.
