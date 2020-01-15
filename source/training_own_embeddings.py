from string import punctuation
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score
import numpy as np
import keras
from keras.callbacks import TensorBoard
from time import time

import pandas as pd

class PrintDot(Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

class TrainAndTest:

    def read_csv_to_list(self, path):
        # path = 'data/new/new_reviews.csv'
        path = path
        data = pd.read_csv(path)
        reviews = data

        return reviews

    # load doc into memory
    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, 'r', encoding='utf-8')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # turn a doc into clean tokens
    def clean_doc(self, doc, vocab):
        # split into tokens by white space
        doc = doc.lower()
        tokens = doc.split()
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # filter out tokens not in vocab
        tokens = [w for w in tokens if w in vocab]
        tokens = ' '.join(tokens)
        return tokens

    def reviews_to_tokens(self, reviews, vocab):
        all_review_tokens = []
        for review_text in reviews:
            # clean doc
            tokens = self.clean_doc(review_text, vocab)
            # update counts
            all_review_tokens.append(tokens)
        return all_review_tokens

    # converting points to classes
    def points_to_class(self, points):
        if float(points) < 1.5:
            return 0
        elif float(points) < 2.5:
            return 1
        elif float(points) < 3.5:
            return 2
        elif float(points) < 4.5:
            return 3
        else:
            return 4

    def make_predictions(self, reviews_df):
        trained_model = keras.models.load_model('sentiment_model_simple_val-loss-0.86_val-acc-0.732.h5')
        print(trained_model.summary())

        # load the vocabulary
        vocab_filename = 'new_vocab.txt'
        vocab = self.load_doc(vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)

        all_text_tokens = self.reviews_to_tokens(reviews_df['text'], vocab)

        # train tokenizer
        tokenizer = Tokenizer(num_words=None)
        tokenizer.fit_on_texts(all_text_tokens)
        sequences_test = tokenizer.texts_to_sequences(all_text_tokens)

        X_test = pad_sequences(sequences_test, maxlen=1939, padding='post')
        pred_test = trained_model.predict(X_test)

        predicted_ratings = []
        for prediction in pred_test:
            predicted_ratings.append(np.argmax(prediction) + 1)

        reviews_df = reviews_df.assign(pred_rating=predicted_ratings)

        return reviews_df

    def load_and_test(self):
        new_model = keras.models.load_model('training_own_embeddings-multichannel-65%.h5')

        print(new_model.summary())

        # load the vocabulary
        vocab_filename = 'vocab.txt'
        vocab = self.load_doc(vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)

        text = [
            'Good rooms. Good food',
            'Bad rooms. Very poor service',
            'Room sizes were small and no space after the bed.',
            'A room with a balcony would have been nice but overall good',
            'The food was very bad and the location is not attractive',
            '5 of us stayed at the Comfort Inn Sept. 2015 as the result of having business in Joplin. Check in was quick and the clerk was very helpful. Our rooms were clean and the beds comfortable. The breakfast bar had the usual items to include biscuits and gravy and the food was about what you get at most other places... More',
            '5 star accommodations as well as service. Even during the floods- they had staff to help us. We LOVED our time there',
            '5 star service, food tops...simply a home run.',
            '50 dollar fee that is charged without notice but then returned.'
        ]

        all_text_tokens = self.reviews_to_tokens(text, vocab)

        # train tokenizer
        tokenizer = Tokenizer(num_words=None)
        tokenizer.fit_on_texts(all_text_tokens)
        sequences_test = tokenizer.texts_to_sequences(all_text_tokens)

        X_test = pad_sequences(sequences_test, maxlen=553, padding='post')
        pred_test = new_model.predict(X_test)

        print(pred_test)

        print(text[0])
        print(np.argmax(pred_test[0]) + 1)
        print(text[1])
        print(np.argmax(pred_test[1]) + 1)
        print(text[2])
        print(np.argmax(pred_test[2]) + 1)
        print(text[3])
        print(np.argmax(pred_test[3]) + 1)
        print(text[4])
        print(np.argmax(pred_test[4]) + 1)
        print(text[5])
        print(np.argmax(pred_test[5]) + 1)
        print(text[6])
        print(np.argmax(pred_test[6]) + 1)
        print(text[7])
        print(np.argmax(pred_test[7]) + 1)
        print(text[8])
        print(np.argmax(pred_test[8]) + 1)

    def check_data_shape(self):
        train_review_file = self.read_csv_to_list('new_reviews.csv')
        print(train_review_file.shape)
        classified_ratings = train_review_file['rating'].apply(self.points_to_class)
        print(classified_ratings.value_counts())

        test_review_file = self.read_csv_to_list('hotel_reviews_test.csv')
        print(test_review_file.shape)
        classified_ratings = test_review_file['rating'].apply(self.points_to_class)
        print(classified_ratings.value_counts())

    def train_and_test(self):
        num_classes = 5
        NAME = 'Conv1D-Own_Embeddings-20Epochs-{}'.format((int(time())))

        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

        # load the vocabulary
        vocab_filename = 'vocab.txt'
        vocab = self.load_doc(vocab_filename)
        vocab = vocab.split()
        vocab = set(vocab)

        train_review_file = self.read_csv_to_list('new_reviews.csv')

        all_text_tokens = self.reviews_to_tokens(train_review_file['text'].tolist(), vocab)

        # create the tokenizer
        tokenizer = Tokenizer()
        # fit the tokenizer on the documents
        tokenizer.fit_on_texts(all_text_tokens)

        # sequence encode
        encoded_docs = tokenizer.texts_to_sequences(all_text_tokens)

        # pad sequences
        max_length = max([len(s.split()) for s in all_text_tokens])
        Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        # define training labels
        classified_ratings = train_review_file['rating'].apply(self.points_to_class)
        print(classified_ratings.value_counts())
        ytrain = to_categorical(classified_ratings, num_classes)

        test_review_file = self.read_csv_to_list('hotel_reviews_test.csv')
        test_text_tokens = self.reviews_to_tokens(test_review_file['text'].tolist(), vocab)

        # sequence encode
        test_encoded_docs = tokenizer.texts_to_sequences(test_text_tokens)
        # pad sequences
        Xtest = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')
        # define training labels
        test_classified_ratings = test_review_file['rating'].apply(self.points_to_class)
        print(test_classified_ratings.value_counts())
        ytest = to_categorical(test_classified_ratings, num_classes)

        # define vocabulary size (largest integer value)
        vocab_size = len(tokenizer.word_index) + 1

        # define model
        model = Sequential()
        model.add(Embedding(vocab_size, 100, input_length=max_length))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        print(model.summary())

        # compile network
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        callback = [PrintDot(), tensorboard]

        # fit network
        model.fit(Xtrain, ytrain, epochs=20, verbose=2, validation_data=(Xtest, ytest), callbacks=callback)

        print('....---------------------testing started---------------------....')

        # Predictions
        pred_test = model.predict(Xtest)
        print(pred_test)

        print(test_review_file['text'][0])
        print('actual - ', test_review_file['rating'][0])
        print('pred - ', np.argmax(pred_test[0]))

        print(test_review_file['text'][1])
        print('actual - ', test_review_file['rating'][1])
        print('pred - ', np.argmax(pred_test[1]))

        print(test_review_file['text'][2])
        print('actual - ', test_review_file['rating'][2])
        print('pred - ', np.argmax(pred_test[2]))

        print(test_review_file['text'][3])
        print('actual - ', test_review_file['rating'][3])
        print('pred - ', np.argmax(pred_test[3]))

        print(test_review_file['text'][4])
        print('actual - ', test_review_file['rating'][4])
        print('pred - ', np.argmax(pred_test[4]))

        pred_test = [np.argmax(x) for x in pred_test]

        # Actual
        true_test = ytest
        true_test = [np.argmax(x) for x in true_test]

        # Find accuracies
        accuracy = accuracy_score(true_test, pred_test)

        print("The total accuracy is : ", accuracy)

        model.save('training_own_embeddings-multichannel-65%.h5')

    def balance_dataframe(self, df):
        lowest_count = df.groupby('rating').apply(lambda x: x.shape[0]).min()
        df = df.groupby('rating').apply(
            lambda x: x.sample(lowest_count)).drop('rating', axis=1).reset_index().set_index('level_1')

        df.sort_index(inplace=True)

        return df

tap = TrainAndTest()

# tap.check_data_shape()
# tap.train_and_test()
# tap.load_and_test()