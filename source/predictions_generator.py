import pandas as pd
import numpy as np
from nn_model import NNModel


class PredictionsGenerator:

    def read_csv_to_list(self, path):
        reviews = pd.read_csv(path, encoding="latin")
        return reviews

    def read_from_platforms(self, hotel_name, platform):
        all_reviews = []
        print('PLATFORM in read_from_platforms', platform)
        print('HOTEL NAME in read_from_platforms', hotel_name)
        # if 'TA' in platforms:
        if platform == 'TA':
            data = self.read_csv_to_list("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/" + hotel_name + "-tripadvisor.csv")
            all_reviews = data['text'].tolist()

        if platform == 'BC':
            data = self.read_csv_to_list("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/" + hotel_name + "-bookingscom.csv")
            all_reviews += data['text'].tolist()

        if platform == 'ALL':
            print('ALL PART REACHED')
            data = self.read_csv_to_list("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/" + hotel_name + "-tripadvisor.csv")
            all_reviews = data['text'].tolist()
            print(len(all_reviews))
            data = self.read_csv_to_list("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/" + hotel_name + "-bookingscom.csv")
            all_reviews += data['text'].tolist()
            print(len(all_reviews))
        return all_reviews

    def make_dataframe(self, review_list):
        np_reviews = np.asarray([review_list]).T
        categories = ['text']
        df_reviews = pd.DataFrame(np_reviews, columns=categories)

        return df_reviews

    def make_predictions(self, hotel_name, platforms):
        all_reviews = self.read_from_platforms(hotel_name, platforms)
        reviews_df = self.make_dataframe(all_reviews)

        nnm = NNModel()
        predited_df = nnm.generate_predictions(reviews_df)

        print(predited_df.shape)

        predited_df.to_csv('C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/predicted_data/' + hotel_name + '_' + platforms + '_predicted.csv')

        return predited_df

    def mock_predictions(self, hotel_name):
        rev_data = self.read_csv_to_list("predicted_data/" + hotel_name + "_mockpred.csv")

        return rev_data
