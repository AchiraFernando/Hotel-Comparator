from nltk import word_tokenize, pos_tag

from aspect_synonym_bank import AspectSynonymBank
from common_terms_generator import CommonTermsGenerator
from mentions_generator import MentionsGenerator
from preprocessor import PreProcessor
import time

class AspectDetail:
    aspect_name = ''
    aspect_rating = 0
    review_list = {}
    common_mentions = {}
    common_term_details = []
    review_summary = {}


class AspectDetailBuilder:

    aspects = ["location", "staff", "food", "room", "service", "value", "cleanliness", "facilities"]
    reviews_per_aspect = {}

    def is_review_in_aspect(self, aspect, review):
        pp = PreProcessor()
        review = pp.preprocess_review(review, False)
        tokens = word_tokenize(review)

        # use pos tag to find all NN and NNS
        tagged_words = pos_tag(tokens)

        is_noun = lambda pos: pos[:2] == 'NN'
        is_nouns = lambda pos: pos[:2] == 'NNS'
        nouns = [word for (word, pos) in tagged_words if is_noun(pos) or is_nouns(pos)]

        asb = AspectSynonymBank()
        aspect_syn = asb.get_synonyms_for_aspect(aspect)

        match = [word for word in nouns if word in aspect_syn]

        if len(match) > 0:
            return True

        return False

    def get_aspect_details(self, review_df):
        review_df['text'].dropna(inplace=True)

        self.reviews_per_aspect = {}

        for index, review in review_df['text'].iteritems():
            review_text = review
            review_rating = review_df['pred_rating'][index]

            # check if the aspect with synonyms is present within the broken down nouns
            for aspect in self.aspects:

                if aspect not in self.reviews_per_aspect.keys():
                    # print(self.reviews_per_aspect.keys())
                    aspect_detail = AspectDetail()
                    aspect_detail.aspect_name = aspect
                    aspect_detail.aspect_rating = 0
                    aspect_detail.review_list = {}
                    aspect_detail.review_summary = {}
                    aspect_detail.common_term_details = []

                    self.reviews_per_aspect[aspect] = aspect_detail

                # if so, add the review to the dictionary with aspect as the key
                if self.is_review_in_aspect(aspect, review_text):
                    reviews = self.reviews_per_aspect[aspect].review_list.keys()
                    if review_text not in reviews:
                        self.reviews_per_aspect[aspect].review_list[review_text] = int(review_rating)

        self.aspect_common_details(20)
        self.update_aspect_rating()

        return self.reviews_per_aspect

    def update_aspect_rating(self):

        for key, value in self.reviews_per_aspect.items():
            all_ratings = value.review_list.values()
            rating_numbers = [int(rating) for rating in all_ratings]

            if len(all_ratings) == 0:
                avg = 0
            else:
                avg = sum(rating_numbers) / len(all_ratings)

            value.aspect_rating = avg

    def aspect_common_details(self, threshold):
        pp = PreProcessor()

        for aspect, aspect_detail in self.reviews_per_aspect.items():
            reviews = aspect_detail.review_list

            reviews = [pp.preprocess_review(review, False) for review in reviews]

            generate_mentions = MentionsGenerator()
            common_mentions = generate_mentions.get_common_mentions(reviews, threshold)
            aspect_detail.common_mentions = common_mentions

            generate_common_terms = CommonTermsGenerator()
            common_term_details = generate_common_terms.get_common_term_details(reviews, threshold)
            aspect_detail.common_term_details = common_term_details

