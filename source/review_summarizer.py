from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
import heapq

from preprocessor import PreProcessor


class ReviewSummarizer:

    def generate_summary(self, review):
        pp = PreProcessor()

        formatted_text = pp.preprocess_review(review, True)
        sentences = sent_tokenize(review)

        word_frequencies = {}
        for word in word_tokenize(formatted_text):
            stop_words = set(stopwords.words('english'))
            if word not in stop_words:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        maximum_frequency = max(word_frequencies.values())

        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 80:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]

        sent_summary = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
        summarized = ' '.join(sent_summary)

        return summarized

    def summarized_reviews(self, aspect_details):
        for aspect, detail in aspect_details.items():

            for rev in detail.review_list.keys():
                summarized_review = self.generate_summary(rev)
                rating = detail.review_list[rev]
                detail.review_summary[summarized_review] = rating

        return aspect_details
