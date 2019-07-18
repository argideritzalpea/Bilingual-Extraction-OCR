# -*- coding: utf-8 -*-
import scrapy
import re
import os
import wget
from scrape_livelingua.items import Ebook
from scrapy.utils.project import get_project_settings

class LivelinguaCrawlerSpider(scrapy.Spider):
    name = 'livelingua_crawler'
    allowed_domains = ['www.livelingua.com']
    start_urls = ['https://www.livelingua.com/project/']

    custom_settings = {"ITEM_PIPELINES" : {'scrape_livelingua.pipelines.DownloadEbooksPipeline': 300},
                        "FILES_URLS_FIELD" : 'file_urls'}

    def parse(self, response):
        settings = get_project_settings()
        print("Your USER_AGENT is:\n%s" % (str(settings.get('ITEM_PIPELINES'))))

        language_links = response.css("div.col-md-4 a::attr(href)").getall()
        for link in language_links[2:4]:
            language = re.match('(.*)(?<=courses)(.*)', link).group(2)[1:-1]
            dir_path = "/Users/Chris/Documents/Grad School/UW/OCR/Bilingual-Extraction-OCR/scrape_livelingua/languages/" + language
            try:
                os.makedirs(dir_path)
            except FileExistsError:
                pass
            self.dir_path = dir_path
            request = response.follow(link, self.parseCourses)
            request.meta['dir'] = dir_path
            yield request

    def parseCourses(self, response):
        courses = response.css("span.thumb-info-caption h6 a::attr(href)").getall()
        dir_path = response.meta['dir']
        for course in courses:
            request = response.follow(course, self.parseEBooks)
            request.meta['dir'] = dir_path
            yield request
            #yield {"ebook": course,
            #"dir_path": dir_path}

    def parseEBooks(self, response):
        eBooks = response.css("div.col-md-6 ul li a::attr(href)").getall()
        dir_path = response.meta['dir']
        for eBook in eBooks:
            if eBook.endswith('.pdf'):
                item = Ebook()
                item['file_urls'] = 'https://www.livelingua.com' + eBook
                item['dir'] = response.meta['dir']
            #wget.download(eBook, response.meta['dir'])
                yield item

#<a href="https://www.livelingua.com/courses/Acholi/" class="btn btn-lg btn-info" style="margin-bottom:10px;width:220px;" title="Acholi">Acholi</a>
