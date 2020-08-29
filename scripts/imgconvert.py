#!/usr/bin/env python

import os
from os.path import basename
from glob import glob

def pngs2mp4(fileexpr,imagesize='640x480'):
    """ 
    input: path(full or not) to files  /path/wrfout_sfc_*
    output: full path to mp4
    """
    newfilename = fileexpr[:-1]
    
    
    
    
    cmd1="""convert -delay 100 {} {}.gif""".format(fileexpr,newfilename)
    print(cmd1)    
    os.system(cmd1)
    cmd2="""ffmpeg -f gif -i {}.gif -c:v libx264 -b 20k -minrate 20k -maxrate 20k -s {} {}.mp4""".format(newfilename,imagesize,newfilename)
    print(cmd2)
    os.system(cmd2)


if __name__ =="__main__":
    pngs2mp4("/home/miguel/Projects/cfabots/images/wrfout_2020082806/SFC/RAIN/wrfout_2020082806_d3_rain_sfc_*",imagesize='640x480')