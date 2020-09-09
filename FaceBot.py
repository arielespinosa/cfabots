import os
import json
import time
from datetime import datetime
import facebook

DEFAULT_MSG = 'Pronóstico para el {} de {} del {} a las {}:00 horas del modelo SisPI'
FOLDER_NAME = 'wrfout_{}'
RAIN_VARNAME = 'wrfout_{}_d3_rain_sfc_22.png'
WIND_VARNAME = 'wrfout_{}_d3_Wind_sfc_22.png'
TEMP_VARNAME = 'wrfout_{}_d3_T_sfc_22.png'
CLOUD_VARNAME = ''

POST_ON_FACEBOOK_FLAG = 0

# https://www.facebook.com/CFA-108484500941829/
CONFIGURATION = {
    'sispi_output_directory':'/var/www/html/INSMET-webs/models/static/models/plots/sispi',
    }

class Facebook:

    page_id = '108484500941829'
    user_access_token = 'EAAEWQo92a8oBAE73xe7ZAupGCQJhV7c9Xbq3mqhbgcaoHGZB8C8iZC91DKaqN2HYgTtZAC1tA55izsl3FrZCzrVv4bunfv5zzPc9ZBjBnTzjFZCMhwzJEZAv5ycFoWsZBaWjzt4vZCNxZApGv6gXFJPycnDzcs7wmCf3SIEpHmo7L3tBE0zKQhvlN9k0ZC4aGCaMIlkndZBCGiJPpZBAZDZD'
    page_access_token = 'EAAEWQo92a8oBADMh9B1jWIhZAvOj551VDLVXaAgJC8AfB3Q4pXdMCD37ggBIxkEZCyS43YehgGOHh2hIthsAzdg9vHzzYEL2XDeW7vIni7B4kHK8kZB9a3hzokWZCOgi3m5i5Mv2Uyrzxpiqy1oD1u4kfrBBV0dejpTjxD8Gpjt2Q7ZCaHAqL'
    
    def __init__(self, p_page_id=None, p_user_access_token=None, p_page_access_token=None):

        if p_page_id:
            self.page_id = p_page_id

        if p_user_access_token:
            self.user_access_token = p_user_access_token

        if p_page_access_token:
            self.page_access_token = p_page_access_token

        try:
            self.fb = facebook.GraphAPI(access_token=self.page_access_token)
        except:
            self.fb = None

    def post_message(self, msg):
        try:            
            self.fb.put_object(parent_object=self.page_id, connection_name='feed', message=msg)
            return True
        except:
            return False

    def post_photos(self, images, msg):
        imgs_id, args = [], {}

        for image in images:
            photo = open(image, "rb")
            imgs_id.append(self.fb.put_photo(photo, album_id='me/photos', published=False)['id'])
            photo.close()

        for img_id in imgs_id:
            key = "attached_media["+str(imgs_id.index(img_id))+"]"
            args[key] = "{'media_fbid':'"+img_id+"'}"

        try:
            args["message"] = msg
            self.fb.request(path='/me/feed', args=None, post_args=args, method='POST')
            return True
        except:
            return False


def post_on_facebook(hour_initialization=None):
  
    if hour_initialization:
        date = datetime.today().strftime('%Y%m%d') + hour_initialization
    
        output_folder = FOLDER_NAME.format(date)

        while not output_folder in os.listdir(CONFIGURATION['sispi_output_directory']):
            print("Wating for output folder on directory...")        
            time.sleep(120)
        
        rain_plot = os.path.join(CONFIGURATION['sispi_output_directory'], output_folder, RAIN_VARNAME.format(date))
        wind_plot = os.path.join(CONFIGURATION['sispi_output_directory'], output_folder, WIND_VARNAME.format(date))
        temp_plot = os.path.join(CONFIGURATION['sispi_output_directory'], output_folder, TEMP_VARNAME.format(date))
        #cloud_plot = os.path.join(CONFIGURATION['sispi_output_directory'], output_folder, CLOUD_VARNAME.format(date))

        msg = DEFAULT_MSG.format(date[6:8], date[4:6], date[:4], date[8:])
        fb = Facebook()
        if fb.post_photos(images=[rain_plot, wind_plot, temp_plot], msg=msg):
            POST_ON_FACEBOOK_FLAG = 1
        else:
            print("Something went wrong trying to post on Facebook :(")


if __name__ == '__main__':

    hour_initialization = ['00', '06', '12', '18']
    post_on_facebook("18")
    """
    while True:
        for hour in hour_initialization:
            post_on_facebook(hour)
    """
            


