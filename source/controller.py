from flask import Flask, jsonify
from flask_cors import CORS
from flask import request
import json
from json import JSONEncoder
from data_handler import DataHandler
from user_authenticator import UserAuthenticator
from runscrapper import RunScrapper, RevScrapperDetail


class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class User:
    def __init__(self, username, password, firstname, lastname, hotel_name):
        self.username = username
        self.password = password
        self.firstname = firstname
        self.lastname = lastname
        self.hotel_name = hotel_name


app = Flask(__name__)
CORS(app)
CORS(app, resources=r'/*', allow_headers='Content-Type')


@app.route('/register', methods=['POST'])
def register():
    print('Register started')
    json_data = request.json

    hotelname = json_data['hotelName'].replace(" ", "")

    user = User(
        username=json_data['username'],
        password=json_data['password'],
        firstname=json_data['firstname'],
        lastname=json_data['lastname'],
        hotel_name=hotelname,
    )
    ua = UserAuthenticator()
    result = ua.get_user(user.username)

    if result is not None:
        return 'user exists'

    result = ua.set_user(user)

    from review_analyzer import ReviewAnalyzer
    reviewAnalyzer = ReviewAnalyzer()

    platforms = ['TA', 'BC', 'ALL']
    for platform in platforms:
        reviewAnalyzer.get_analyzed_reviews(hotelname, platform)

    return json.dumps(result)


@app.route('/getuser', methods=['POST'])
def getuser():
    json_data = request.json

    username = json_data['username']

    ua = UserAuthenticator()
    result = ua.get_user(username)

    user = User(
        username=result[2],
        password=result[3],
        firstname=result[0],
        lastname=result[1],
        hotel_name=result[4],
    )

    return json.dumps(user, cls=MyEncoder)


@app.route('/get_own_hotel_details', methods=['POST'])
def get_own_hotel_details():
    json_data = request.json
    username = json_data['username']
    hotel_name = json_data['hotel_name']
    platforms = json_data['platforms']
    print(platforms)

    platform = ''
    if 'TA' and 'BC' in platforms:
        platform = 'ALL'
    elif 'TA' in platforms:
        platform = 'TA'
    else:
        platform = 'BC'

    data_handler = DataHandler()
    reviews = data_handler.get_analyzed_reviews(hotel_name, platform)

    return reviews[2]


@app.route('/get_analyzed_com_reviews', methods=['POST'])
def get_analyzed_com_reviews():
    json_data = request.json
    username = json_data['username']
    hotel_name = json_data['hotel_name']
    platforms = json_data['platforms']

    hotelname = hotel_name.replace(" ", "")

    print(hotelname)

    platform = ''
    if 'TA' and 'BC' in platforms:
        platform = 'ALL'
    elif 'TA' in platforms:
        platform = 'TA'
    else:
        platform = 'BC'

    # IF ALREADY SCRAPPED CALL THE ANALYZER
    from review_analyzer import ReviewAnalyzer
    reviewAnalyzer = ReviewAnalyzer()
    analyzed_reviews = reviewAnalyzer.get_analyzed_reviews(hotelname, platform)

    # AND RETURN THE ANALYZED JSON
    return analyzed_reviews


@app.route('/extract_reviews', methods=['POST'])
def extract_reviews():
    json_data = request.json

    rev_scrapper_detail = RevScrapperDetail()

    rev_scrapper_detail.hotel_name = json_data['hotel_name']
    rev_scrapper_detail.review_count = json_data['review_count']
    rev_scrapper_detail.tripadvisor_url = json_data['tripadvisor_url']
    rev_scrapper_detail.bookingdotcom_url = json_data['bookingdotcom_url']

    rev_scrapper_detail.hotel_name = rev_scrapper_detail.hotel_name.replace(" ", "")

    print(rev_scrapper_detail.hotel_name)
    print(rev_scrapper_detail.review_count)
    print(rev_scrapper_detail.tripadvisor_url)
    print(rev_scrapper_detail.bookingdotcom_url)

    runscrapper = RunScrapper()
    runscrapper.remove_already_scrapped(rev_scrapper_detail)
    result = runscrapper.run(rev_scrapper_detail)

    return json.dumps(result)


@app.route('/get_all_scrapped_files', methods=['GET'])
def get_all_scrapped_files():
    datahandler = DataHandler()
    all_files = datahandler.all_existing_drops()

    return json.dumps(all_files)


@app.route('/change_password', methods=['POST'])
def change_password():
    json_data = request.json
    username = json_data['username']
    password = json_data['password']

    ua = UserAuthenticator()
    result = ua.change_password(username, password)

    return json.dumps(result)


@app.route('/testapi', methods=['GET'])
def testapi():
    import time
    time.sleep(179)

    return json.dumps('PASSEE')


if __name__ == '__main__':
    app.run(debug=True, port=5500)


