from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup


class PubMedScraper:
    def __init__(self):
        self._is_driver = False
        self.abstracts = dict()
        self.open_webdriver(headless=True)

    # Add (url, abstract) to cache
    def _cache_abstract(self, url, abstract):
        if url not in self.abstracts:
            self.abstracts[url] = abstract
        return

    def open_webdriver(self, headless=True):
        assert(not self._is_driver)

        # Javascript requires us to open the webpagewith a webdriver
        is_headless = Options()
        if headless:
            is_headless.add_argument('--headless')
        self._driver = webdriver.Chrome('./chromedriver', chrome_options=is_headless)
        self._is_driver = True
        return

    def close_webdriver(self):
        assert(self._is_driver)

        self._driver.quit()
        self._is_driver = False
        return

    # Get the relevant abstract for a PubMed url
    def get_abstract(self, url):
        assert(self._is_driver)

        # Check if we've previously scraped the abstracts
        if url in self.abstracts:
            return self.abstracts[url]
        # Extract abstract from PubMed url
        else:
            self._driver.get(url)
            soup = BeautifulSoup(self._driver.page_source, 'lxml')
            abstract = soup.abstracttext.text

            self._cache_abstract(url, abstract)
            return abstract

    # Return a list of abstracts
    def get_all_abstracts(self, urls):
        assert(self._is_driver)

        abstracts = []
        for url in urls:
            abstracts.append(self.get_abstract(url))
        return abstracts
