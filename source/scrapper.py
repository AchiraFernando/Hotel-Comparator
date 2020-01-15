import mysql.connector
import json
from json import JSONEncoder

class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class Scrapper:

    def scrap_reviews(self, hotel_name, platform):
        # START REVIEW SCRAPPER
        return
