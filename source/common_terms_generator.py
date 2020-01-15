from nltk import sent_tokenize, word_tokenize
from collections import Counter

from feature_polarity_extractor import FeaturePolarityExtractor


class CommonTermDetail:
    term = ''
    term_polarity = 0
    term_count = 0


class CommonTermsGenerator:

    all_terms = []
    common_terms = {}

    common_term_details = {}

    def find_all_terms(self, reviews):
        self.all_terms = []

        for rev in reviews:
            self.all_terms += word_tokenize(rev)

        return self.all_terms

    def fetch_common_terms(self, threshold):
        self.common_terms = {}

        counter = Counter(self.all_terms)
        self.common_terms = counter.most_common(threshold)

        return self.common_terms

    def generate_polarity(self, reviews):
        self.common_term_details = {}

        for term in self.common_terms:
            common_term = CommonTermDetail()
            common_term.term = term[0]
            common_term.term_count = term[1]

            filtered_reviews = [rev for rev in reviews if common_term.term in rev]

            filtered_sents = []
            for rev in filtered_reviews:
                sent_rev = sent_tokenize(rev)
                sents = [rev for rev in sent_rev if common_term.term in rev]
                filtered_sents += sents

            total_score = 0
            fpe = FeaturePolarityExtractor()
            score = fpe.sentiment_analyzer_scores(filtered_sents)
            total_score += score

            common_term.term_polarity = total_score

            self.common_term_details[common_term.term] = common_term

        return self.common_term_details

    def get_common_term_details(self, reviews, threshold):
        self.find_all_terms(reviews)
        self.fetch_common_terms(threshold)
        self.generate_polarity(reviews)

        return self.common_term_details

