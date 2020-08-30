#!/usr/bin/env python

import os
from os.path import basename
from glob import glob


def pngs2mp4(fileexpr, imagesize='640x480'):
    """ 
    input: path to files  /path/wrfout_sfc_*
    output: full path to mp4
    """
    newfilename = fileexpr[:-1]

    cmd1 = """convert -delay 100 {} {}.gif""".format(fileexpr, newfilename)
    print(cmd1)
    # os.system(cmd1)
    #cmd2 = """/usr/bin/ffmpeg -i {}.gif -c:v libx265 -crf 42 -s {} {}.mp4""".format(
    #    newfilename, imagesize, newfilename)
    #cmd2 = """/usr/bin/ffmpeg -y -f image2 -pattern_type glob -framerate 1 -i {}.png  -crf 42.0 -vcodec libx264 -filter:v scale=640:480  -profile baseline  -coder 1 -flags +loop -cmp chroma -partitions +parti4x4+partp8x8+partb8x8 -me_method hex -subq 6 -me_range 16 -g 250 -keyint_min 25 -sc_threshold 40 -i_qfactor 0.71 -b_strategy 1 -threads 0 "{}.mp4" """.format(
        # fileexpr, newfilename)
    cmd2 = """/usr/bin/ffmpeg -y -i {}.gif  -crf 42.0 -vcodec libx264 -filter:v scale=640:480   "{}.mp4" """.format(
        newfilename, newfilename)
    cmd2 = """cat {}.png|/usr/bin/ffmpeg -y -framerate 1 -i -  -crf 42.0 -vcodec libx264 -filter:v scale=640:480  -coder 1 -flags +loop -cmp chroma -partitions +parti4x4+partp8x8+partb8x8 -me_method hex -subq 6 -me_range 16 -g 250 -keyint_min 25 -sc_threshold 40 -i_qfactor 0.71 -b_strategy 1 -threads 0 "{}.mp4" """.format(
        fileexpr, newfilename)
   
   
    print(cmd2)
    os.system(cmd2)
    return """{}.mp4""".format(newfilename)


if __name__ == "__main__":
    pngs2mp4(
        "/home/miguel/Projects/cfabots/images/wrfout_2020082800/SFC/RAIN/wrfout_2020082800_d3_rain_sfc_*",
        imagesize='640x480')
    pngs2mp4('/home/miguel/Pictures/.webcam/20200827/shot*')