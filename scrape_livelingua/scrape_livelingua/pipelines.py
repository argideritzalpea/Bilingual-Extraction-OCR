# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from scrapy.pipelines.files import FilesPipeline
from scrapy.utils.project import get_project_settings



class DownloadEbooksPipeline(FilesPipeline):
    headers = {
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept-Language': 'en-GB,en;q=0.9,nl-BE;q=0.8,nl;q=0.7,ro-RO;q=0.6,ro;q=0.5,en-US;q=0.4',
    }

    #def start_requests(self):
    #    self.settings=get_project_settings()

    def process_item(self, item, info):
        for ebook_url in item.get(self.files_urls_field, []):
            request = Request(url=ebook_url,
                              headers=self.headers)
            request.meta['dir'] = item['dir']
            yield request

    def file_path(self, request, response, info):
        return request.meta['dir']
