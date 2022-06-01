from crawler import my_crawler
from elastic import store, delete
from pagerank import calculate_page_rank
from search import search_docs
from hits import best_authors
print("Enter the maximum numbers of articles for the first part")
number = int(input())
print("Enter the number of URLs for crawling")
m = int(input())
print("Enter each URL in a line with a space at the end, preventing from searching the url by your system.")
first_urls = []
for i in range(m):
    first_urls.append(input().strip())

print("Execution of first part (crawling)")
result = my_crawler(first_urls, number)
file = open('crawls.txt','w+')
file.write(result)

print("Enter number 2 to start the second part")
command = input()
while command != "2":
    print("Wrong instruction, please enter number 2")
    command = input()
print("Execution of second part, building index")
print("Write the local address of your elastic search. For example: localhost:9200")
elastic_address = input()
store(elastic_address, result)

print("Enter number 3 for the third part")
command = input()
while command != "3":
    print("Wrong instruction, enter number 3")
    command = input()
print("Third part, calculating page rank")
print("Enter the alpha parameter for page rank")
alpha = float(input())
calculate_page_rank(elastic_address, alpha)

elastic_address = "localhost:9200"
print("Enter number 4 for the forth part of project")
command = input()
while command != "4":
    print("Wrong instruction, enter number 4")
    command = input()
print("For search, enter search, for ending the part 4, enter end")
command = input()
while command != "end":
    if command != "search":
        print("Wrong instruction, enter search or end)
    else:
        print("You are in search part. Enter the following details.")
        title_query = input("title query:\n")
        title_weight = float(input("title weight:\n"))
        abstract_query = input("abstract query:\n")
        abstract_weight = float(input("abstract weight:\n"))
        year_query = input("year query:\n")
        year_weight = float(input("year weight:\n"))
        page_rank = input("Do you want page rank? answer with y/n\n")
        page_rank_weight = 0
        if page_rank == "y":
            page_rank_weight = float(input("page rank weight:\n"))
        search_docs(elastic_address, title_weight, title_query, abstract_weight, abstract_query,
                            year_weight, year_query, True, page_rank_weight)
        print("Enter search to search again or end to end the searching")
        command = input()
print("Enter number 5 for part 5")
command = input()
while command != "5":
    print("Wrong instruction. Please enter number 5")
    command = input()
print("Enter the number of writers")
n = int(input())
best_authors(elastic_address, n)

print("Do you want to delete the index from your elastic search? please enter y/n")
command = input()
if command == "y":
    delete(elastic_address)

print("End of the project")
