import subprocess
import glob
import os

class RevScrapperDetail:
    hotel_name = ''
    review_count = 0
    tripadvisor_url = ''
    bookingdotcom_url = ''

class RunScrapper:
    # SET
    # HOTEL_NAME = % ~1
    # SET
    # START_URL = % ~2
    # SET
    # REVIEW_COUNT = % ~3

    def trip_advisor(self, hotel_name, hotel_url, review_count):

        # hotel_name = "GFH"
        # hotel_url = "https://www.tripadvisor.com/Hotel_Review-g293962-d2038179-Reviews-Galle_Face_Hotel_Colombo-Colombo_Western_Province.html"
        # review_count = "5"

        item = subprocess.Popen(["C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/scrapper/tripadvisor_scrap.bat", hotel_name, hotel_url, review_count], shell=True, stdout=subprocess.PIPE)

        for line in item.stdout:
            print(line)

    def booking_dot_com(self, hotel_name, hotel_url, review_count):

        # hotel_name = "GFH"
        # hotel_url = "https://www.booking.com/hotel/lk/galle-face.en-gb.html"
        # review_count = "5"
        print(hotel_name)
        print(hotel_url)
        print(review_count)
        item = subprocess.Popen(["C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/scrapper/bookingdotcom_scrap.bat", hotel_name, hotel_url, review_count], shell=True, stdout=subprocess.PIPE)
        for line in item.stdout:
            print(line)

    def remove_already_scrapped(self, rev_scrapper_detail):
        all_rev_paths = []
        for fpath in glob.glob('C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/drops/' + rev_scrapper_detail.hotel_name + '*'):
            all_rev_paths.append(fpath)

        ta_file = []

        if rev_scrapper_detail.tripadvisor_url != '':
            ta_file = [s for s in all_rev_paths if (rev_scrapper_detail.hotel_name + '-tripadvisor.csv') in s]

        if len(ta_file) == 1:
            os.remove(ta_file[0])

        bdc_file = []

        if rev_scrapper_detail.bookingdotcom_url != '':
            bdc_file = [s for s in all_rev_paths if (rev_scrapper_detail.hotel_name + '-bookingscom.csv') in s]

        if len(bdc_file) == 1:
            os.remove(bdc_file[0])

    def run(self, rev_scrapper_detail):
        try:
            if str(rev_scrapper_detail.tripadvisor_url) != '' and str(rev_scrapper_detail.bookingdotcom_url) != '':
                self.trip_advisor(str(rev_scrapper_detail.hotel_name), str(rev_scrapper_detail.tripadvisor_url), str(rev_scrapper_detail.review_count))
                self.booking_dot_com(str(rev_scrapper_detail.hotel_name), str(rev_scrapper_detail.bookingdotcom_url), str(rev_scrapper_detail.review_count))
            elif str(rev_scrapper_detail.tripadvisor_url) != '':
                self.trip_advisor(str(rev_scrapper_detail.hotel_name), str(rev_scrapper_detail.tripadvisor_url), str(rev_scrapper_detail.review_count))
            else:
                self.booking_dot_com(str(rev_scrapper_detail.hotel_name), str(rev_scrapper_detail.bookingdotcom_url), str(rev_scrapper_detail.review_count))
            return 'success'
        except:
            return 'fail'

    def booking_dot_com_testt(self):

        hotel_name = "GFH_DeLeTe"
        hotel_url = "https://www.booking.com/hotel/lk/galle-face.en-gb.html"
        review_count = "100"
        print(hotel_name)
        print(hotel_url)
        print(review_count)
        item = subprocess.Popen(["C:/Users/acfelk/Documents/IIT_Files/final year/FYP/fyp_workfiles/final_project/backend/scrapper/bookingdotcom_scrap.bat", hotel_name, hotel_url, review_count], shell=True, stdout=subprocess.PIPE)
        for line in item.stdout:
            print(line)


# rs = RunScrapper()
# # rs.booking_dot_com_testt()