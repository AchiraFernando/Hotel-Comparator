from data_handler import DataHandler
from predictions_generator import PredictionsGenerator
from aspect_detail_builder import AspectDetailBuilder
from controller import MyEncoder
from review_summarizer import ReviewSummarizer
import json
import time

class ReviewAnalyzer:

    predicted_reviews = []
    aspect_details = {}
    hotel1_name = 'GalleFaceHotel'
    hotel2_name = 'ShangrilaHotel'

    def get_analyzed_reviews(self, hotel_name, platforms):
        data_handler = DataHandler()
        reviews = data_handler.get_analyzed_reviews(hotel_name, platforms)
        print(hotel_name)
        print(reviews)
        if reviews is None:
            self.analyze_reviews(hotel_name, platforms)
            reviews = data_handler.get_analyzed_reviews(hotel_name, platforms)

            if reviews is None:
                return None

        return reviews[2]

    def analyze_reviews(self, hotel_name, platform):
        start = time.time()

        pred_start = time.time()
        make_predictions = PredictionsGenerator()
        predicted = make_predictions.make_predictions(hotel_name, platform)
        # predicted = make_predictions.mock_predictions(hotel_name)
        self.predicted_reviews = predicted
        print('Finished Predicting Ratings')
        pred_end = time.time()
        print('Total time taken for predictions: ', pred_end - pred_start, ' seconds')

        aspect_d_start = time.time()
        self.aspect_details = self.get_aspect_details(self.predicted_reviews)
        print('Finished Categorizing To Aspects')
        aspect_d_end = time.time()
        print('Total time taken for aspect detail generation: ', aspect_d_end - aspect_d_start, ' seconds')

        summ_start = time.time()
        self.aspect_details = self.get_summarized_reviews(self.aspect_details)
        print('Finished Summarizing Reviews')
        summ_end = time.time()
        print('Total time taken for summarization: ', summ_end - summ_start, ' seconds')

        self.store_analyzed_reviews(hotel_name, self.aspect_details, platform)
        print('Finished Storing To Database')

        end = time.time()
        print('Total time taken for review analysis: ', end-start, ' seconds')

        return self.aspect_details

    def get_aspect_details(self, reviews_df):
        reviews_per_aspect = AspectDetailBuilder()

        details = reviews_per_aspect.get_aspect_details(reviews_df)

        return details

    def get_summarized_reviews(self, aspect_details):
        rs = ReviewSummarizer()

        details = rs.summarized_reviews(aspect_details)

        return details

    def store_analyzed_reviews(self, hotel_name, aspect_details, platforms):
        for key, value in self.aspect_details.items():
            value.review_list = {}

        to_json = json.dumps(aspect_details, cls=MyEncoder)
        data_handler = DataHandler()

        data_handler.set_analyzed_reviews(hotel_name, to_json, platforms)


# ra = ReviewAnalyzer()
# ra.get_analyzed_reviews('Kingsbury', ['ALL'])