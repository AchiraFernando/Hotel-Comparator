from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from string import punctuation

class PreProcessor:

    def remove_symbols(self, review):
        symbols = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")

        # for line in reviews:
        if type(review) is str:
            review = symbols.sub('', review)
            review = re.sub(r'https?:\/\/.*[\r\n]*', '', review, flags=re.MULTILINE)
            review = re.sub(r'\<a href', ' ', review)
            review = re.sub(r'&amp;', '', review)
            review = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', review)
        else:
            print('invalid line')

        return review

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)

        return string.strip().lower()

    def clean_review(self, doc):
        doc = doc.lower()
        # split into tokens by white space
        tokens = word_tokenize(doc)
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]

        tokens = ' '.join(tokens)
        return tokens

    def remove_only_stopwords(self, review):
        review = review.lower()
        tokens = word_tokenize(review)
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if (not w in stop_words) and len(w) > 1]
        tokens = ' '.join(tokens)
        return tokens

    def clean_review_ignore_sw(self, doc):
        doc = doc.lower()
        # split into tokens by white space
        tokens = word_tokenize(doc)
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]

        tokens = ' '.join(tokens)
        return tokens

    def preprocess_review(self, review, ignore_stopwords):
        rev = self.remove_symbols(review)

        if ignore_stopwords:
            rev = self.clean_review_ignore_sw(rev)
        else:
            rev = self.clean_review(rev)

        return rev

    def preprocess_review_list(self, reviews):
        processed_revs = []
        for review in reviews:
            rev = self.remove_symbols(review)
            rev = self.clean_review(rev)

            processed_revs.append(rev)

        return processed_revs

