#!/usr/bin/env python

import os
from os.path import basename
from glob import glob


def pngs2mp4(fileexpr, imagesize='640:480'):
    """ 
    input: globpath to files  /path/wrfout_sfc_*
    output: full path to mp4
    """
    newfilename = fileexpr[:-1]

    cmd1 = """convert -delay 100 {} {}.gif""".format(fileexpr, newfilename)
    # print(cmd1)
    # os.system(cmd1)

    # cmd2 = """cat {}.png|/usr/bin/ffmpeg -y -framerate 1 -i -  -crf 42.0 -vcodec libvpx -b:v 50k -v:f scale={} "{}.webm" """.format(
    # fileexpr,imagesize, newfilename)

    cmd1 = """cat {}.png|/usr/bin/ffmpeg -y -framerate 1 -i -  -vcodec libvpx -b:v 50k -s 640x480  "{}.webm" """.format(
        fileexpr, newfilename)

    cmd2 = """/usr/bin/ffmpeg -y  -i {}.webm  -b:v 8k -s 640x480 "{}.gif" """.format(
        fileexpr, newfilename)

    print(cmd2)
    os.system(cmd1)
    # os.system(cmd2)
    return """{}.webm""".format(newfilename)


if __name__ == "__main__":
    # pngs2mp4(
    # "/home/miguel/Projects/cfabots/images/wrfout_2020082800/SFC/RAIN/wrfout_2020082800_d3_rain_sfc_*",
    # imagesize='640x480')

    vidfile = pngs2mp4(
        "/home/miguel/Projects/cfabots/images/wrfout_2020082800/SFC/RAIN/wrfout_2020082800_d3_rain_sfc_*",
        imagesize='640:480')

    # pngs2mp4('/home/miguel/Pictures/.webcam/20200827/shot*')