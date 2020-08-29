#!/usr/bin/env python

from TelBot import TelBot
import json
import sys
import time
from glob import glob
from os.path import join, exists
from scripts.imgconvert import pngs2mp4


def publicar():
    
    bot = TelBot("1314850663:AAFuBzMDs5niJiUXHvH6ZaWI9rXHaz7GX8A")
    channel_id = 572031301
    cycles = ['00', '06', '12', '18']

    nowlcl = time.localtime()
    nowgmt = time.gmtime()

    currcycle = cycles[int(nowgmt.tm_hour / 6)]   # ultima corrida

    outputdir = "/opt/sispi/OUTPUTS_1W/outputs"
    curoutput = time.strftime('%Y%m%d') + currcycle

    if not exists(join(outputdir, curoutput, "wrfout_" + curoutput)):
        print("La última corrida no está lista")
        sys.exit()

    #  RAIN
    lluviafiles = join(outputdir, curoutput, "wrfout_" + curoutput, 'SFC/RAIN',
                       "wrfout_" + curoutput + "_d3_rain_sfc_*")
    count = 24                                                     
    
    if len(glob(lluviafiles)) != count:                                  # espera la cantidad fijada de archivos
        print("Los últimos gráficos no están listos")
        sys.exit()

    vidfile = pngs2mp4(lluviafiles, imagesize='480x320')
    bot.sendVideo(channel_id, video=open(vidfile, 'rb'), width=480, height=320)

    

if __name__ == '__main__':
    publicar()
    vidfile = '/home/miguel/Projects/cfabots/images/wrfout_2020082806/SFC/RAIN/wrfout_2020082806_d3_rain_sfc_.mp4'

    # bot.sendVideo(572031301, video=open(vidfile, 'rb'), width=480, height=320)

    
    
    