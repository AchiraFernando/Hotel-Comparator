import pandas as pd
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Embedding, Conv1D, Dropout, Input, GlobalMaxPooling1D, concatenate, Activation
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt
from keras import backend
import pickle

from preprocessor import PreProcessor


class PrintDot(Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


class NNModel:

    def read_csv_to_list(self, path):
        path = path
        data = pd.read_csv(path)
        reviews = data

        return reviews

    def tokenize_reviews(self, all_reviews):
        all_review_tokens = []
        preprocessing = PreProcessor()
        for review_text in all_reviews:
            # remove symbols
            review_text = preprocessing.remove_symbols(review_text)
            # clean review
            tokens = preprocessing.clean_review(review_text)
            # collect all the tokens
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

    def create_model(self, max_length, vocab_size, num_classes):

        input_layer = Input(shape=(max_length,), dtype='int32')

        embedding = Embedding(vocab_size, 200, input_length=max_length)(input_layer)

        channel1 = Conv1D(filters=128, kernel_size=2, activation='relu')(embedding)
        channel1 = Dropout(0.2)(channel1)
        channel1 = GlobalMaxPooling1D()(channel1)
        channel1 = Dense(128, activation='relu')(channel1)

        channel2 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding)
        channel2 = Dropout(0.2)(channel2)
        channel2 = GlobalMaxPooling1D()(channel2)
        channel2 = Dense(128, activation='relu')(channel2)

        channel3 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding)
        channel3 = Dropout(0.2)(channel3)
        channel3 = GlobalMaxPooling1D()(channel3)
        channel3 = Dense(128, activation='relu')(channel3)

        merged = concatenate([channel1, channel2, channel3], axis=1)

        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(num_classes)(merged)
        output = Activation('softmax')(merged)
        model = Model(inputs=[input_layer], outputs=[output])

        return model

    def testing_model(self, model, final_X_test, final_Y_test):
        # Predictions
        pred_test = model.predict(final_X_test)
        pred_test = [np.argmax(x) for x in pred_test]

        # Actual
        true_test = final_Y_test
        true_test = [np.argmax(x) for x in true_test]

        # Find accuracies
        accuracy = accuracy_score(true_test, pred_test)

        print("The total accuracy is : ", accuracy)

        confusion_matrix = ConfusionMatrix(true_test, pred_test)
        print("Confusion matrix:\n%s" % confusion_matrix)

        confusion_matrix.plot(normalized=True)
        plt.show()

    def train_and_test(self):

        num_classes = 5

        train_review_file = pd.read_csv('combined_reviews.csv')

        print('....--------------SHUFFLING DATASET--------------....')

        # It’s a good practice to shuffle the data before splitting between a train and test set.
        train_review_file = train_review_file.reindex(np.random.permutation(train_review_file.index))

        print('....--------------SPLITTING DATASET TO 80% TRAIN & 20% TEST--------------....')

        train_dataset, test_dataset = train_test_split(
            train_review_file, train_size=0.8, test_size=0.2
        )

        print('....--------------SHUFFLING TRAIN DATASET--------------....')

        # It’s a good practice to shuffle the data before splitting between a train and test set.
        train_dataset = train_dataset.reindex(np.random.permutation(train_dataset.index))

        print('....--------------SPLITTING TRAIN DATASET TO 90% TRAIN & 10% VALIDATION--------------....')

        train_data, validation_data = train_test_split(
            train_dataset, train_size=0.9, test_size=0.1
        )

        print('....--------------CATEGORIZING LABELS TO CLASSES--------------....')

        test_dataset.loc[:, 'rating'] = test_dataset['rating'].apply(self.points_to_class)
        train_data.loc[:, 'rating'] = train_data['rating'].apply(self.points_to_class)
        validation_data.loc[:, 'rating'] = validation_data['rating'].apply(self.points_to_class)

        print('....--------------CLEANING TRAIN DATA--------------....')

        all_train_tokens = self.tokenize_reviews(train_data['text'].tolist())

        print('....--------------TOKENIZING TRAIN DATA--------------....')

        # create the tokenizer
        tokenizer_train = Tokenizer()
        # fit the tokenizer on the train tokens
        tokenizer_train.fit_on_texts(all_train_tokens)
        # sequence encode
        train_encoded_docs = tokenizer_train.texts_to_sequences(all_train_tokens)
        # max length
        max_length = max([len(s.split()) for s in all_train_tokens])
        # pad sequences
        final_X_train = pad_sequences(train_encoded_docs, maxlen=max_length, padding='post')

        # define training labels
        print('Train Rating Count: ', train_data['rating'].value_counts())
        final_Y_train = to_categorical(train_data['rating'], num_classes)

        print('....--------------CLEANING VALIDATION DATA--------------....')

        all_val_tokens = self.tokenize_reviews(validation_data['text'].tolist())

        print('....--------------TOKENIZING VALIDATION DATA--------------....')

        # sequence encode
        val_encoded_docs = tokenizer_train.texts_to_sequences(all_val_tokens)
        # pad sequences
        final_X_val = pad_sequences(val_encoded_docs, maxlen=max_length, padding='post')

        # define validation labels
        print('Validation Rating Count: ', validation_data['rating'].value_counts())
        final_Y_val = to_categorical(validation_data['rating'], num_classes)

        print('....--------------CLEANING TESTING DATA--------------....')

        all_test_tokens = self.tokenize_reviews(test_dataset['text'].tolist())

        print('....--------------TOKENIZING TESTING DATA--------------....')

        # sequence encode
        test_encoded_docs = tokenizer_train.texts_to_sequences(all_test_tokens)

        # pad sequences
        final_X_test = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')

        # define training labels
        print('Test Rating Count: ', test_dataset['rating'].value_counts())
        final_Y_test = to_categorical(test_dataset['rating'], num_classes)

        print('....--------------DEFINING THE VOCAB SIZE--------------....')

        # define vocabulary size (largest integer value)
        vocab_size = len(tokenizer_train.word_index) + 1

        NAME = 'Conv1D-60Epochs-512batchsize-{}'.format((int(time())))
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

        print('....--------------CREATING MODEL--------------....')

        model = self.create_model(max_length, vocab_size, num_classes)

        print(model.summary())

        # compile network
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        filepath = "CNN_best_weights.{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.4f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='max')

        earlyStopping = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=0, verbose=0,
                                      mode='auto', baseline=None, restore_best_weights=False)

        callback = [PrintDot(), tensorboard, checkpoint, earlyStopping]

        print('....--------------TRAINING AND VALIDATING MODEL--------------....')

        # fit network
        model.fit(final_X_train, final_Y_train, epochs=60, batch_size=512, validation_data=(final_X_val, final_Y_val),
                  callbacks=callback, shuffle=True)

        model.save('final_conv1D_model.h5')

        print('....--------------TESTING STARTED--------------....')

        self.testing_model(model, final_X_test, final_Y_test)

    def generate_predictions(self, reviews_df):
        # load the tokenizer and the model
        with open("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/trained_model/tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)

        trained_model = load_model('C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/trained_model/cnn_saved_model.hdf5')

        all_test_tokens = self.tokenize_reviews(reviews_df['text'].tolist())

        # note that we shouldn't call "fit" on the tokenizer again
        sequences = tokenizer.texts_to_sequences(all_test_tokens)
        data = pad_sequences(sequences, maxlen=1939, padding='post')

        pred_test = trained_model.predict(data)

        predicted_ratings = []
        for prediction in pred_test:
            predicted_ratings.append(np.argmax(prediction) + 1)

        reviews_df = reviews_df.assign(pred_rating=predicted_ratings)

        # reviews_df.to_csv('GFH.csv')

        backend.clear_session()

        return reviews_df


    def testing_model(self):
        trained_model = load_model('trained_model/rating_model-0.6893-0.7268.h5')

        # compile network
        trained_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

        # load tokenizer
        tokenizer = Tokenizer()
        with open('trained_model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        test_revs = self.read_csv_to_list('trained_model/test_dataset.csv')

        test_revs.loc[:, 'rating'] = test_revs['rating'].apply(self.points_to_class)

        actual_texts = test_revs['text']
        actual_ratings = test_revs['rating']

        final_Y_test = to_categorical(actual_ratings, 5)

        actual_text_tokens = self.tokenize_reviews(actual_texts)

        # train tokenizer
        # tokenizer = Tokenizer(num_words=None)
        # tokenizer.fit_on_texts(actual_text_tokens)
        sequences_test = tokenizer.texts_to_sequences(actual_text_tokens)

        X_test = pad_sequences(sequences_test, maxlen=1939, padding='post')

        # Predictions
        pred_test = trained_model.predict(X_test)
        pred_test = [np.argmax(x) for x in pred_test]

        # Actual
        true_test = final_Y_test
        true_test = [np.argmax(x) for x in true_test]

        # Find accuracies
        accuracy = accuracy_score(true_test, pred_test)

        print("The total accuracy is : ", accuracy)




# tensorboard --logdir [your log dir] --host=127.0.0.1