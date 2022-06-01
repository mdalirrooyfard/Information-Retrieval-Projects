from scrapy.spiders import Spider
from scrapy import Request
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import json


def my_crawler(first_urls, number_of_papers = 2000):
    process = CrawlerProcess(get_project_settings())

    class my_spider(Spider):
        allowed_domains = ['semanticscholar.org']
        name = "semantic"
        seen_ids = set()
        next_ids_to_be_seen = set()
        information = dict()
        count = 0

        def start_requests(self):
            for u in self.start_urls:
                yield Request(url=u, callback=self.parse, dont_filter=True)

        def parse(self, response):
            if self.count < number_of_papers:
                new_id = response.url.split("/")[-1]
                if new_id not in self.seen_ids:
                    if new_id in self.next_ids_to_be_seen:
                        self.next_ids_to_be_seen.remove(new_id)
                    self.seen_ids.add(new_id)
                    self.count += 1
                    self.information[response.url] = dict()
                    self.information[response.url]["id"] = new_id
                    self.information[response.url]["title"] = response.xpath('//h1[@data-selenium-selector="paper-detail-title"]/text()').get()
                    temp = response.xpath('//pre[@class="bibtex-citation"]/text()').get()
                    elements = temp.split("author={")[1].split("\n")[0].replace("}","").replace(",","").split("and")
                    authors = []
                    for e in elements:
                        authors.append(e.strip())
                    self.information[response.url]["authors"] = authors
                    date = response.xpath('//span[@data-selenium-selector="paper-year"]/span/span/text()').get()
                    if date is not None:
                        self.information[response.url]["date"] = date
                    else:
                        self.information[response.url]["date"] = ""
                    abstract = response.xpath('//div[@class="text-truncator abstract__text text--preline"]/text()').get()
                    if abstract is not None:
                        self.information[response.url]["abstract"] = abstract
                    else:
                        self.information[response.url]["abstract"] = ""
                    references = response.xpath('//div[@data-selenium-selector="reference"]/'
                                                'div[@class="citation-list__citations"]/div[@class="paper-citation"]/'
                                   '/div/h2/a/@href').getall()
                    self.information[response.url]["references"] = []
                    for j in range(min(len(references), 10)):
                        references[j] = "http://www.semanticscholar.org" + references[j]
                        self.information[response.url]["references"].append(references[j].split("/")[-1])
                        reference_id = references[j].split("/")[-1]
                        if reference_id not in self.seen_ids and references[j] and reference_id not in self.next_ids_to_be_seen:
                            self.next_ids_to_be_seen.add(reference_id)
                            yield Request(url=references[j], callback=self.parse, dont_filter=True)
            else:
                process.stop()

    process.crawl(my_spider, start_urls=first_urls)
    process.start()
    information = my_spider.information
    d = json.dumps(list(information.values()))
    return d



