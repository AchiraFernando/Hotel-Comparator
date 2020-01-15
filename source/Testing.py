import pandas as pd
import os

from aspect_detail_builder import AspectDetailBuilder
from common_terms_generator import CommonTermsGenerator
from mentions_generator import MentionsGenerator
from nn_model import NNModel
from preprocessor import PreProcessor


def check_predict_ratings():
    data = pd.read_csv('test_data/test_revs.csv')

    nn_model = NNModel()

    predicted_df = nn_model.generate_predictions(data)

    predicted_df.to_csv('test_data/predicted_revs.csv')

    data = pd.read_csv('test_data/predicted_revs.csv')

    if data['pred_rating'] is not None:
        print('Testing passed - Reviews Precited Successfully !')
        os.remove('test_data/predicted_revs.csv')
    else:
        print('Review Prediction has failed')

def check_review_preprocess():
    data = pd.read_csv('test_data/test_revs.csv')

    preprocess = PreProcessor()
    preprocessed_revs = preprocess.preprocess_review_list(data['text'])

    if len(preprocessed_revs) > 0:
        print('Testing passed - Preprocessing is Successfully !')
    else:
        print('Preprocessing has failed')

def check_aspect_classification():
    data = pd.read_csv('test_data/after_prediction.csv')

    aspect_detail = AspectDetailBuilder()
    detailed_dic = aspect_detail.get_aspect_details(data)

    if detailed_dic is not None:
        print('Testing passed - Aspect Classification is Successfully !')
    else:
        print('Aspect Classification has failed')

def check_common_terms_and_sentiment():
    data = pd.read_csv('test_data/after_prediction.csv')

    generate_mentions = MentionsGenerator()
    common_mentions = generate_mentions.get_common_mentions(data['text'], 20)

    if common_mentions is not None:
        print('Testing passed - Common Term Extraction is Successfully !')
    else:
        print('Common Term Extraction has failed')

def check_common_mentions_extraction():
    data = pd.read_csv('test_data/after_prediction.csv')

    generate_common_terms = CommonTermsGenerator()
    common_term_details = generate_common_terms.get_common_term_details(data['text'], 20)

    if common_term_details is not None:
        print('Testing passed - Common Mentions Extraction is Successfully !')
    else:
        print('Common Mentions Extraction has failed')

# To check if reviews get predicted
# check_predict_ratings()

# To check whether review texts gets preprocessed
# check_review_preprocess()

# To divide reviews into aspects
# check_aspect_classification()

# To find all common terms with sentiment
# check_common_terms_and_sentiment()

# To find all the common mentions within reviews
# check_common_mentions_extraction()

