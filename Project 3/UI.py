from crawler import my_crawler
from elastic import store, delete
from pagerank import calculate_page_rank
from search import search_docs
from hits import best_authors
# print("برای بخش اول تعداد حداکثر مقالات را وارد کنید.")
# number = int(input())
# print("تعداد آدرس های اولیه را وارد کنید.")
# m = int(input())
# print("حال هر ادرس اولیه را در یک خط با جدا وارد کنید.اخر هر کدام یک اسپیس وارد کنید تا اینتر زدن منجر به جست و جوی ادرس در اینترنت نشود.")
# first_urls = []
# for i in range(m):
#     first_urls.append(input().strip())
#
# print("حال بخش اول اجرا می شود:")
# result = my_crawler(first_urls, number)
# file = open('crawls.txt','w+')
# file.write(result)
#
# print("بعد از اتمام بخش اول عدد 2 را برای شروع بخش دوم وارد کنید.")
# command = input()
# while command != "2":
#     print("دستور وارد شده اشتباه است. لطفا عدد 2 را وارد کنید.")
#     command = input()
# print("حال بخش دوم ")
# print("ادرسی که الستیک سرج تان روی آن اجرا میشود را وارد کنید. به شکل localhost:9200 برای مثال.")
# elastic_address = input()
# store(elastic_address, result)
#
# print("بعد از اتمام بخش دوم عدد 3 را برای شروع بخش سوم وارد کنید.")
# command = input()
# while command != "3":
#     print("دستور وارد شده اشتباه است. لطفا عدد 3 را وارد کنید.")
#     command = input()
# print("بخش سوم ")
# print("آلفا را برای محاسبه page rank وارد کنید.")
# alpha = float(input())
# calculate_page_rank(elastic_address, alpha)
#
elastic_address = "localhost:9200"
print("بعد از اتمام بخش سوم عدد 4 را برای شروع بخش چهارم وارد کنید.")
command = input()
while command != "4":
    print("دستور اشتباه وارد شده است. لطفا عدد 4 را وارد کنید.")
    command = input()
print("برای شروع جست و جو دستور search را وارد کنید. در صورتی که نمی خواهید جست و جو کنید دستور end را وارد کنید.")
command = input()
while command != "end":
    if command != "search":
        print("دستور اشتباه وارد شده است. لطفا دستور search و یا end را وارد کنید.")
    else:
        print("شما وارد قسمت جست و جو شده اید. موارد زیر را وارد کنید.")
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
        print("برای یک جست و جوی جدید عبارت search و برای اتمام جست و جو عبارت end را وارد کنید.")
        command = input()
print("برای شروع بخش پنجم عدد 5 را وارد کنید.")
command = input()
while command != "5":
    print("دستور اشتباه وارد شده است. لطفا عدد 5 را وارد کنید.")
    command = input()
print("تعداد نویسنده ها را وارد کنید.")
n = int(input())
best_authors(elastic_address, n)

print("آیا می خواهید ایندکس ساخته شده را دیلیت کنید؟ y/n")
command = input()
if command == "y":
    delete(elastic_address)

print("5 بخش این پروژه به پایان رسید.")
