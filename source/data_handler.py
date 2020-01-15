import mysql.connector
import pandas as pd
import glob, os
from scrapper import Scrapper


class DataHandler:

    def read_csv_to_list(self, path):
        reviews = pd.read_csv(path, encoding="latin")
        return reviews

    def get_platform(self, platforms):
        if 'ALL' in platforms:
            return 'ALL'
        elif 'TA' in platforms:
            return 'TA'
        else:
            return 'BC'

    def pick_platform(self, platforms):
        if ('TA' and 'BC') or 'ALL' in platforms:
            return 'ALL'
        elif 'TA' in platforms:
            return 'TA'
        else:
            return 'BC'

    def get_analyzed_reviews(self, hotel_name, platforms):

        print(platforms)

        platform = self.get_platform(platforms)

        print(platform)

        conn = mysql.connector.connect(host='localhost',
                                       database='comparator_dash',
                                       user='root',
                                       password='')

        cursor = conn.cursor(buffered=True)
        query = "SELECT * FROM analyzed_reviews WHERE (hotel_name = '" + hotel_name + "' AND platform = '" + platform + "')"
        cursor.execute(query)

        result = cursor.fetchone()
        conn.commit()

        print(result)
        return result

    def set_analyzed_reviews(self, hotel_name, review_json, platform):

        print('Storing ', review_json, ' for ', platform)
        print('Storing ', hotel_name, ' for ', platform)
        # platform = self.get_platform(platforms)

        conn = mysql.connector.connect(host='localhost',
                                       database='comparator_dash',
                                       user='root',
                                       password='')

        cursor = conn.cursor()
        query = "INSERT INTO analyzed_reviews (hotel_name, platform, review_json) VALUES (%s, %s, %s)"
        val = (hotel_name, platform, review_json)

        try:
            cursor.execute(query, val)
            conn.commit()
            print('Analyzed review successfully stored')
            status = 'Analyzed review successfully stored'
        except:
            print('Error when storing reviews')
            status = 'Error when storing reviews'
        finally:
            conn.close()

        return status

    def delete_reviews(self, hotel_name):
        conn = mysql.connector.connect(host='localhost',
                                       database='comparator_dash',
                                       user='root',
                                       password='')

        cursor = conn.cursor()
        query = "DELETE FROM analyzed_reviews WHERE (hotel_name = '" + hotel_name + "')"
        val = (hotel_name)

        try:
            cursor.execute(query)
            conn.commit()
            print('Delete Successful')
            status = 'Delete Successful'
        except:
            print('Error when deleting')
            status = 'Error when deleting'
        finally:
            conn.close()

        return status

    # CHECK IF REVIEWS ARE ALREADY SCRAPPED FOR PLATFORMS
    def check_and_scrap_reviews(self, hotel_name, platforms):
        for platform in platforms:
            if (platform == 'TA'):
                data = self.read_csv_to_list("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/" + hotel_name + "-tripadvisor.csv")

                if data is None:
                    # NOW CALL THE SCRAPPER TO SCRAP REVIEWS TO drops
                    scrapper = Scrapper()
                    scrapper.scrap_reviews(hotel_name, platform)

            if (platform == 'BC'):
                data = self.read_csv_to_list("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/" + hotel_name + "-bookingscom.csv")

                if data is None:
                    # NOW CALL THE SCRAPPER TO SCRAP REVIEWS TO drops
                    scrapper = Scrapper()
                    scrapper.scrap_reviews(hotel_name, platform)


    def all_existing_drops(self):
        all_files = []
        os.chdir("C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/")
        for file in glob.glob("*.csv"):
            all_files.append(file)

        return all_files

# data_handler = DataHandler()
# data_handler.delete_reviews('GFH')
# data_handler.delete_reviews('GalleFaceHotel')