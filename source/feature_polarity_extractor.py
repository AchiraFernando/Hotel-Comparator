from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem import PorterStemmer
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocessor import PreProcessor

import time


class FeaturePolarityExtractor:

    def convert_to_wn_tags(self, tag):
        if tag.startswith('J'):
            return wn.ADJ
        elif tag.startswith('N'):
            return wn.NOUN
        elif tag.startswith('R'):
            return wn.ADV
        elif tag.startswith('V'):
            return wn.VERB
        return None

    def detect_polarity(self, sentence):
        pp = PreProcessor()
        lemmatizer = WordNetLemmatizer()

        sentiment = 0.0
        token_count = 0

        # preprocessed_sent = pp.preprocess_review(sentence, False)
        tokenized_sent = word_tokenize(sentence)
        tagged_sentence = pos_tag(tokenized_sent)

        for word, tag in tagged_sentence:
            wordnet_tag = self.convert_to_wn_tags(tag)
            if wordnet_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue

            lemma = lemmatizer.lemmatize(word, pos=wordnet_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wordnet_tag)
            if not synsets:
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            token_count += 1

        return sentiment

    lemmatizer = WordNetLemmatizer()

    def get_sentiment(self, word, tag):
        """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

        wn_tag = self.convert_to_wn_tags(tag)

        if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
            return []

        lemma = self.lemmatizer.lemmatize(word, pos=wn_tag)
        if not lemma:
            return []

        synsets = wn.synsets(word, pos=wn_tag)
        if not synsets:
            return []

        # Take the first sense, the most common
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())

        return swn_synset.pos_score() - swn_synset.neg_score()


    def total_polarity_score(self, sentences):
        score = 0

        for sent in sentences:
            score += self.detect_polarity(sent)

        return score

    def total_score_per_review(self, review):
        sentences = sent_tokenize(review)
        total_score = self.total_polarity_score(sentences)
        print(total_score)

        return total_score


    def read_csv_to_list(self, path):
        path = path
        data = pd.read_csv(path)
        reviews = data

        return reviews

    def sentiment_analyzer_scores(self, sentences):
        analyser = SentimentIntensityAnalyzer()
        score = 0

        for sent in sentences:
            sc = analyser.polarity_scores(sent)
            total_sc = sc['pos'] - sc['neg']
            score += total_sc

        return score



# fpe = FeaturePolarityExtractor()
#
# start = time.time()
#
# data = fpe.read_csv_to_list("drops/GFH-tripadvisor.csv")['text'].tolist()
# data = data[0:20]
# #
# total_score = 0
#
# for rev in data:
#     sent_tokens = sent_tokenize(rev)
#     score = fpe.sentiment_analyzer_scores(sent_tokens)
#     total_score += score
#
# end = time.time()
#
# print(total_score)
# print(len(data))
# print('TOTAL TIME ', end - start)