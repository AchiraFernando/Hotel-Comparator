from collections import Counter

class MentionsGenerator:

    all_mentions = []
    common_mentions = {}

    def find_all_mentions(self, reviews):
        all_bigrams = [b for l in reviews for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        self.all_mentions = all_bigrams

        return self.all_mentions

    def count_common_mentions(self, threshold):
        counter = Counter(self.all_mentions)
        commons = counter.most_common(threshold)
        self.common_mentions = commons

        return self.common_mentions

    def get_common_mentions(self, reviews, threshold):
        self.find_all_mentions(reviews)
        self.count_common_mentions(threshold)

        dic = {}
        for key, value in self.common_mentions:

            if value > 1:
                key1 = key[0]
                key2 = key[1]
                formatted_key = key1 + ' ' + key2
                dic[formatted_key] = value

        return dic