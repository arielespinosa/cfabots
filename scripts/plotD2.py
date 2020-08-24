#!/usr/bin/env python
#-*- coding: utf-8 -*-
# encoding: utf-8
from __future__ import unicode_literals

# Working script to generate maps from wrfout netCDF files
# using matplot lib with basemap
# Basemap coding from David John Gagne II
# Written by Luke Madaus for use with operational WRF domains

import matplotlib
matplotlib.use('agg')
from matplotlib.colors import LinearSegmentedColormap
import sys,getopt
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime, timedelta
import coltbls as coltbls
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid import make_axes_locatable
import matplotlib.axes as maxes
import glob, os
import matplotlib.colors as mcolors
from cpt_convert import loadCPT
import matplotlib as mpl
import matplotlib.colors as colors
import multiprocessing
import time
from matplotlib.patches import Polygon
from wrf import getvar, interplevel, to_np, get_basemap, latlon_coords
from mpl_toolkits.basemap import Basemap, addcyclic
from scipy.ndimage.filters import minimum_filter, maximum_filter

#------------------------------------------------------------------------------
# CPU cores number
#------------------------------------------------------------------------------
PROCESS_LIMIT = 8


def timestring(wrftime,curtime):
    curtime_str = '%02.0f' % curtime
    wrfdt = datetime.strptime(wrftime,'%Y-%m-%d_%H:%M:%S')
    outtime = '%sZ F+%s Hours' % (wrfdt.strftime('%a %Y-%m-%d/%H%M'),curtime_str)
    return outtime


def steps(llcrnrlon, llcrnrlat, m, nsteps, nowstep):

    print nsteps, nowstep
    factor = 0.35
    xx, yy = m(llcrnrlon+5*factor+0.1, llcrnrlat+1*factor)  # lon/lat esquina iquierda - abajo
    x1,y1 = m(xx-5*factor,yy-1*factor)
    x2,y2 = m(xx+5*factor,yy-1*factor)
    x3,y3 = m(xx+5*factor,yy+1*factor)
    x4,y4 = m(xx-5*factor,yy+1*factor)

    xdif = x3-x1
    ydif = y3-y1

#    nowstep = 4
#    nsteps = 25
    dx = xdif/25
    dy = ydif/4*factor

    xl,yl,xr,yr = (x1,y1+3*dy,x1,y1+7*dy)

    for i in range(nsteps):
        if i == 0:
            xr=xl+dx
            yr=yr
            if i == nowstep:
                poly = Polygon([(xl,yl),(xr,yl),(xr,yr),(xl,yr)],facecolor='red',edgecolor='#b9eac4',linewidth=0.5)
                plt.gca().add_patch(poly)
            else:
                poly = Polygon([(xl,yl),(xr,yl),(xr,yr),(xl,yr)],facecolor='#4eed71',edgecolor='#b9eac4',linewidth=0.5)
                plt.gca().add_patch(poly)
        else:
            xr=xr+dx
            yr=yr
            xl=xl+dx
            yl=yl
            if i == nowstep:
                poly = Polygon([(xl,yl),(xr,yl),(xr,yr),(xl,yr)],facecolor='red',edgecolor='#b9eac4',linewidth=0.5)
                plt.gca().add_patch(poly)
            else:
                poly = Polygon([(xl,yl),(xr,yl),(xr,yr),(xl,yr)],facecolor='#4eed71',edgecolor='#b9eac4',linewidth=0.5)
                plt.gca().add_patch(poly)



def drawmap(m,curtimestring,dom,a,DATA,TITLESTRING,PROD,UNITS, llcrnrlon, llcrnrlat, ftimes):

    F = plt.gcf()  # Gets the current figure

    m.drawmeridians(range(0, 360, 4),labels=[1,0,0,1],fontsize=8, linewidth=0)
    m.drawparallels(range(-180, 180, 4),labels=[1,0,0,1],fontsize=8, linewidth=0)

#    m.drawrivers(color='#0000ff', linewidth=0.15)

#    m.readshapefile('/home/adrian/Desktop/RogerData/shp/waterways', 'river', color='#0000ff', linewidth=0.15)
    m.readshapefile('/home/adrian/Desktop/RogerData/shp/Provincias/provi', 'roads', color='k', linewidth=0.9)
    m.readshapefile('/home/adrian/Desktop/RogerData/shp/Municipios/muni', 'municipios', color='grey', linewidth=0.25)

    m.drawstates(color='gray', linewidth=0.25)
    m.drawcoastlines(color='k', linewidth=0.9)
    m.drawcountries(color='k', linewidth=0.9)
#    m.scatter(xlat,ylons)  # Para resaltar algun punto de interes
    
    plt.title('WRF-ARW %s (%s)   Valid: %s' % (TITLESTRING, UNITS, curtimestring), \
        fontsize=11,bbox=dict(facecolor='white', alpha=0.65),\
        x=0.5,y=.94,weight = 'demibold',style='oblique', \
        stretch='normal', family='sans-serif')

    # Code to make the colorbar outside of the main axis, on the bottom, and lined up
    ax = plt.gca()  # Gets the current axes
    divider = make_axes_locatable(ax)

    file_id = '%s_%s_f%02d' % (dom, PROD, a)
    filename = '%s.png' % (file_id)

    steps(llcrnrlon, llcrnrlat, m, ftimes, a)

    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Convert the figure to a gif file
    os.system('convert -render -flatten %s %s.gif' % (filename, file_id))
    os.system('rm -f %s' % filename)


def plot_comp_reflect(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):
    print("    COMP REFLECTIVITY")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)
    
    QR = nc.variables['QRAIN']
    try:
        QS = nc.variables['QSNOW']
    except:
        QS = np.zeros(np.shape(QR))

    # Define 'constant' densities (kg m-3)
    rhor = 1000
    rhos = 100
    rhog = 400
    rhoi = 917

    # Define "fixed intercepts" (m-4)
    Norain = 8.0E6
    #Nosnow = 2.0E7
    Nosnow = 2.0E6*np.exp(-0.12 * (temps[time]-273))
    Nograu = 4.0E6


    # First, find the density at the first sigma level
    # above the surface
    density = np.divide(psfc[time],(287.0 * temps[time]))
    #print "Rho: ", np.mean(density)
    Qra_all = QR[time]
    Qsn_all = QS[time]

    for j in range(len(Qra_all[1,:,1])):
        curcol_r = []
        curcol_s = []
        for i in range(len(Qra_all[1,1,:])):
                maxrval = np.max(Qra_all[:,j,i])
                maxsval = np.max(Qsn_all[:,j,i])
                curcol_r.append(maxrval)        
                curcol_s.append(maxsval)
        np_curcol_r = np.array(curcol_r)
        np_curcol_s = np.array(curcol_s)
        if j == 0:
            Qra = np_curcol_r
            Qsn = np_curcol_s
        else:
            Qra = np.row_stack((Qra, np_curcol_r))
            Qsn = np.row_stack((Qsn, np_curcol_s))

    # Calculate slope factor lambda
    lambr = np.divide((3.14159 * Norain * rhor), np.multiply(density, Qra))
    lambr = lambr ** 0.25

    #lambs = np.divide((3.14159 * Nosnow * rhoi), np.multiply(density, Qsn))
    #lambs = lambs ** 0.25
    lambs = np.exp(-0.0536 * (temps[time] - 273))
    
    # Calculate equivalent reflectivity factor
    Zer = (720.0 * Norain * (lambr ** -7.0)) * 1E18
    Zes = (0.224 * 720.0 * Nosnow * (lambr ** -7.0) * (rhos/rhoi) ** 2) * 1E18
    Zes_int = np.divide((lambs * Qsn * density), Nosnow)
    Zes = ((0.224 * 720 * 1E18) / (3.14159 * rhor) ** 2) * Zes_int ** 2 

    Ze = np.add(Zer, Zes)
    #Ze = Zer
    # Convert to dBZ
    dBZ = 10 * np.log10(Ze)    
    dBZ = np.nan_to_num(dBZ)
    units = 'dBZe'
    print "      MAX: ", np.max(dBZ)
    # Now plot
    
    def _generate_cmap(name, lutsize):
        """Generates the requested cmap from it's name *name*.  The lut size is
        *lutsize*."""

        spec = {
        'blue': [
            (0.0, 0.92549019607843142, 0.92549019607843142),
            (0.07142857, 0.96470588235294119, 0.96470588235294119),
            (0.14285714, 0.96470588235294119, 0.96470588235294119),
            (0.21428571, 0.0, 0.0),
            (0.28571429, 0.0, 0.0),
            (0.35714286, 0.0, 0.0),
            (0.42857143, 0.0, 0.0),
            (0.50000000, 0.0, 0.0),
            (0.57142857, 0.0, 0.0),
            (0.64285714, 0.0, 0.0),
            (0.71428571, 0.0, 0.0),
            (0.78571429, 0.0, 0.0),
            (0.85714286, 1.0, 1.0),
            (0.92857143, 0.78823529411764703, 0.78823529411764703),
            (1.0, 0.0, 0.0)],
        'green': [
            (0.0, 0.92549019607843142, 0.92549019607843142),
            (0.07142857, 0.62745098039215685, 0.62745098039215685),
            (0.14285714, 0.0, 0.0),
            (0.21428571, 1.0, 1.0),
            (0.28571429, 0.78431372549019607, 0.78431372549019607),
            (0.35714286, 0.56470588235294117, 0.56470588235294117),
            (0.42857143, 1.0, 1.0),
            (0.50000000, 0.75294117647058822, 0.75294117647058822),
            (0.57142857, 0.56470588235294117, 0.56470588235294117),
            (0.64285714, 0.0, 0.0),
            (0.71428571, 0.0, 0.0),
            (0.78571429, 0.0, 0.0),
            (0.85714286, 0.0, 0.0),
            (0.92857143, 0.33333333333333331, 0.33333333333333331),
            (1.0, 0.0, 0.0)],
        'red': [
            (0.0, 0.0, 0.0),
            (0.07142857, 0.0039215686274509803, 0.0039215686274509803),
            (0.14285714, 0.0, 0.0),
            (0.21428571, 0.0, 0.0),
            (0.28571429, 0.0, 0.0),
            (0.35714286, 0.0, 0.0),
            (0.42857143, 1.0, 1.0),
            (0.50000000, 0.90588235294117647, 0.90588235294117647),
            (0.57142857, 1.0, 1.0),
            (0.64285714, 1.0, 1.0),
            (0.71428571, 0.83921568627450982, 0.83921568627450982),
            (0.78571429, 0.75294117647058822, 0.75294117647058822),
            (0.85714286, 1.0, 1.0),
            (0.92857143, 0.59999999999999998, 0.59999999999999998),
            (1.0, 0.0, 0.0)]
        }

        # Generate the colormap object.
        if isinstance(spec, dict) and 'red' in spec.keys():
            return colors.LinearSegmentedColormap(name, spec, lutsize)
        else:
            return colors.LinearSegmentedColormap.from_list(name, spec, lutsize)
    
    LUTSIZE = mpl.rcParams['image.lut']

    cm_pyart = _generate_cmap('NWSRef', LUTSIZE)

    REF_LEVELS = range(5,90,5)
    #CREFLECT=m.contourf(x,y,dBZ,REF_LEVELS,cmap=coltbls.reflect_ncdc())
    CREFLECT=m.contourf(x,y,dBZ,REF_LEVELS,cmap=cm_pyart) #pyart NWS
    #CREFLECT=m.pcolormesh(x,y,dBZ,cmap=cm_pyart, vmin=10, vmax=70) #pyart
    #SREFLECT=plt.contourf(x,y,dBZ)

    CREFLECT.cmap.set_under((1.0, 1.0, 1.0))
    CREFLECT.cmap.set_over((23/255,12/255,30/255))

    title = 'Simulated Composite Reflectivity'
    prodid = 'cref'
    m.colorbar(CREFLECT)

    file_id = '%s_%s_f%02d' % (dom, prodid, a)

    if txtflag == 1:
        np.savetxt(file_id+'.txt',dBZ)

    drawmap(m,curtimestring,dom,a,CREFLECT, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)     


def plot_precip(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):
    print("    PRECIP")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)

    rainc =  nc.variables['RAINC']
    rainnc = nc.variables['RAINNC']

    rainc1 =  nc1.variables['RAINC']
    rainnc1 = nc1.variables['RAINNC']

    stemps = nc.variables['T2'][time]+6.5*nc.variables['HGT'][time]/1000.
    mslp = nc.variables['PSFC'][time]*np.exp(9.81/(287.0*stemps)*nc.variables['HGT'][time])*0.01 + (6.7 * nc.variables['HGT'][time] / 1000)


    # First, find out if this is first time or not
    # Based on skip.  This should be total from each output time
    if time == 0:
#        prev_total = rainc[time] + rainnc[time]
#    else:
        prev_total = rainc1[time] + rainnc1[time]
    total_accum = rainc[time] + rainnc[time]
    precip_tend = total_accum  - prev_total
    
    # Convert from mm to in
    precip_tend = precip_tend #* .0393700787
    units = 'mm/h'

   # draw filled contours.
#    clevs1 = np.array([0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750])/5
#    cmap_data = [(1.0, 1.0, 1.0),
#                 (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
#                 (0.0, 1.0, 1.0),
#                 (0.0, 0.8784313797950745, 0.501960813999176),
#                 (0.0, 0.7529411911964417, 0.0),
#                 (0.501960813999176, 0.8784313797950745, 0.0),
#                 (1.0, 1.0, 0.0),
#                 (1.0, 0.6274510025978088, 0.0),
#                 (1.0, 0.0, 0.0),
#                 (1.0, 0.125490203499794, 0.501960813999176),
#                 (0.9411764740943909, 0.250980406999588, 1.0),
#                 (0.501960813999176, 0.125490203499794, 1.0),
#                 (0.250980406999588, 0.250980406999588, 1.0),
#                 (0.125490203499794, 0.125490203499794, 0.501960813999176),
#                 (0.125490203499794, 0.125490203499794, 0.125490203499794),
#                 (0.501960813999176, 0.501960813999176, 0.501960813999176),
#                 (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
#                 (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
#                 (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
#                 (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
#                 (0.4000000059604645, 0.20000000298023224, 0.0)]
#    cmap1 = mcolors.ListedColormap(cmap_data, 'precipitation')
#    norm1 = mcolors.BoundaryNorm(clevs1, cmap1.N)

    
    def cm_precip():
        """
        Range of values:
            metric: 0 to 762 millimeters
            english: 0 to 30 inches
        """
        # The amount of precipitation in inches
#        a = [0,  .01,   .1, .25, .5,1,1.5,2, 3, 4, 6, 8,10,15,20,30]
#        a = [0,  0.01, 0.6,   1,  2,4  ,6,8,12,17,21,25,33,40,50,70]   # OK
        a = [0.1,0.25,0.5,1,1.5,2,3,4,5,6,8,10,12,16,20,24,30,36,42,60]
        
        clevs = np.array(a)
        # Normalize the bin between 0 and 1 (uneven bins are important here)
        norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
    
        # Color tuple for every bin
#        C = np.array([[255,255,255],
#                    [199,233,192],
#                    [161,217,155],
#                    [116,196,118],
#                    [49,163,83],
#                    [0,109,44],
#                    [255,250,138],
#                    [255,204,79],
#                    [254,141,60],
#                    [252,78,42],
#                    [214,26,28],
#                    [173,0,38],
#                    [112,0,38],
#                    [59,0,48],
#                    [76,0,115],
#                    [255,219,255]])  # 16

        C = np.array([[255,255,255],
                    [0,250,76],
                    [0,227,69],
                    [0,203,61],
                    [0,180,54],
                    [0,156,47],
                    [0,133,40],
                    [0,109,32],
                    [0,86,25],
                    [254,250,79],
                    [254,198,70],
                    [254,146,62],
                    [254,93,53],
                    [254,41,44],
                    [115,6,34],
                    [128,5,44],
                    [140,5,55],
                    [180,3,110],
                    [207,1,154],
                    [234,0,199]])

        print 'precip',len(a),len(C)
        # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
        COLORS = []
        for i, n in enumerate(norm):
            COLORS.append((n, np.array(C[i])/255.))
    
        # Create the colormap
        cmap = colors.LinearSegmentedColormap.from_list("precipitation", COLORS)
    
        return cmap,norm,clevs

    cmap1,norm1,clevs1 = cm_precip()

    PRECIP=m.contourf(x,y,precip_tend,clevs1,cmap=cmap1)#,extend="max")
#    PRECIP=m.contourf(x,y,precip_tend,clevs1,norm=norm1,cmap=cmap1,extend="max")
    print "      MAX: ",np.max(precip_tend)

    PRECIP.cmap.set_under((1.0, 1.0, 1.0))
    PRECIP.cmap.set_over((100/255,0/255,86/255))

    title = ' Hourly Precip (mm/h)'
    prodid = 'precip'

    m.colorbar(PRECIP)

    # Convert Surface Pressure to Mean Sea Level Pressure    
    stemps = temps[time]+6.5*nc.variables['HGT'][time]/1000.
    mslp = nc.variables['PSFC'][time]*np.exp(9.81/(287.0*stemps)*nc.variables['HGT'][time])*0.01 + (6.7 * nc.variables['HGT'][time] / 1000)

    # Contour the pressure
    P=m.contour(x,y,mslp,V=2,colors='k',linewidths=0.9)
    plt.clabel(P,inline=1,fontsize=8,fmt='%1.0f',inline_spacing=1)

    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    np.savetxt(file_id+'.txt',precip_tend)
    
    prodid = 'pmsl'
    print "      MSLP MAX: ", np.max(mslp)
    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    
    

    # Using basemap
    mode = 'wrap'
    window = 70
    local_min, local_max = extrema(mslp, mode, window)

    prmsl, lons = addcyclic(mslp, x)

    xlows = x[local_min]; xhighs = x[local_max]
    ylows = y[local_min]; yhighs = y[local_max]
    lowvals = prmsl[local_min]; highvals = prmsl[local_max]

    # plot lows as blue L's, with min pressure value underneath.
    xyplotted = []
    # don't plot if there is already a L or H within dmin meters.
    yoffset = 0.022*(m.ymax-m.ymin)
    dmin = yoffset

    for x,y,p in zip(xlows, ylows, lowvals):
        print x, m.xmax, x, m.xmin, y, m.ymax, y, m.ymin
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                plt.text(x,y,'B',fontsize=9,fontweight='bold',
                         ha='center',va='center',color='r')
                plt.text(x,y-yoffset,repr(int(p)),fontsize=5,
                         ha='center',va='top',color='r',
                         bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                xyplotted.append((x,y))
    # plot highs as red H's, with max pressure value underneath.
    xyplotted = []

    for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                plt.text(x,y,'A',fontsize=9,fontweight='bold',
                         ha='center',va='center',color='b')
                plt.text(x,y-yoffset,repr(int(p)),fontsize=5,
                         ha='center',va='top',color='b',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            xyplotted.append((x,y))

    
    
    if txtflag == 1:
        np.savetxt(file_id+'.txt',mslp)
    
    drawmap(m,curtimestring,dom,a,PRECIP, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)     


def plot_olr(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):
    print("    OLR/IRSAT")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)
    olr = nc.variables['OLR']
    
    sbc = .000000056704    
    ir_T = ((olr[time] / sbc) ** (0.25)) - 273.15

    cpt = loadCPT('IR4AVHRR6.cpt')
    # Makes a linear interpolation
    cpt_convert = LinearSegmentedColormap('cpt', cpt)

    OLR=m.pcolormesh(x,y,ir_T,cmap=cpt_convert, vmin=-103, vmax=104)
    
    title = 'TOA Inferred Temperature'
    prodid = 'olr'
    units = u"\u00B0" + "C"    

    OLR.cmap.set_under((1.0, 1.0, 1.0))
    OLR.cmap.set_over( (0.0, 0.0, 0.0))

    m.colorbar(OLR)

    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    
    if txtflag == 1:
        np.savetxt(file_id+'.txt',ir_T)

    print "      MAX: ", np.max(ir_T)
    drawmap(m,curtimestring,dom,a,OLR, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)     



def plot_pblh(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):
    print("    PBLH")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)
    pblh = nc.variables['PBLH']

    def cm_pblh():
        """
        MPH: vmin=0, vmax=140
        m/s: vmin=0, vmax=60
        """
        # The wind speed bins in miles per hour (MPH)
#        a = [0,5,10,15,20,25,30,35,40,45,50,60,70,80,100,120,140]
#        a = [0,2,5,7,10,12,15,17,20,25,30,35,40,45,50,55,60]
##        a = [0.0,0.25,0.6,0.95,1.3,1.65,2,2.35,2.7,3.05,3.4,3.75,4.1,4.45,4.8,5.15,5.5,5.85,6.2,6.55,6.9,7.25,7.6]
        a = [0  , 0.4,0.8, 1.2,  1.6,   2,3,   4,  5,   6,  8,  10, 12,  14, 16,  18, 20,  22, 24,  26, 28,  30, 32, 40]  # OK
#        clevs = np.array(a)#*1.61
        clevs = np.array(a)*100#*1.61
        # Normalize the bin between 0 and 1 (uneven bins are important here)
        norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
    
        # Color tuple for every bin
        C = np.array([[255,255,255],
                        [230,230,255],
                        [206,206,255],
                        [183,183,255],
                        [164,255,164],
                        [109,247,109],
                        [55,240,55],
                        [0,232,0],
                        [255,255,0],
                        [255,213,0],
                        [255,170,0],
                        [255,128,0],
                        [255,51,51],
                        [230,41,41],
                        [204,31,31],
                        [179,20,20],
                        [153,10,10],
                        [128,0,0],
                        [255,64,255],
                        [234,43,234],
                        [212,21,212],
                        [191,0,191],
                        [170,170,170],
                        [230,230,230]])

        print 'pblh',len(a),len(C)
        # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
        COLORS = []
        for i, n in enumerate(norm):
            COLORS.append((n, np.array(C[i])/255.))
    
        # Create the colormap29
        cmap = colors.LinearSegmentedColormap.from_list("wind", COLORS)
    
        return cmap,norm,clevs

    cmap1,norm1,clevs1 = cm_pblh()

    PBLH=plt.contourf(x,y,pblh[time],clevs1,cmap=cmap1)

    PBLH.cmap.set_under((1.0, 1.0, 1.0))
    PBLH.cmap.set_over((230/255,230/255,230/255))

    title = 'PBL Heigth'
    prodid = 'pblh'
    units = "m"    

    m.colorbar(PBLH)

    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    
    if txtflag == 1:
        np.savetxt(file_id+'.txt',pblh[time])

    print "      MAX: ", np.max(pblh[time])
    drawmap(m,curtimestring,dom,a,PBLH, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)     



def plot_surface(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):
    print("    SURFACE")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)

    # Convert Surface Pressure to Mean Sea Level Pressure    
#    stemps = temps[time]+6.5*nc.variables['HGT'][time]/1000.
#    mslp = nc.variables['PSFC'][time]*np.exp(9.81/(287.0*stemps)*nc.variables['HGT'][time])*0.01 + (6.7 * nc.variables['HGT'][time] / 1000)

    ftemps = temps[time]-273.15

    def cm_temp():
        """
        F:
            vmax=120, vmin=-60
        C:
            vmax=50, vmin=-50
        """
        # The range of temperature bins in Fahrenheit
#        a = np.arange(-60,121,5)
        #a = np.linspace(0,30,39) #np.array([0,2,4,6,8,10,12,14,16,17,18,19,19.5,20,20.5,21.0,21.5,22.0,22.5,23.0,23.5,24.0,25.0,25.5,26.0,26.5,27.0,27.5,28.0,28.5,29.0,29.7,30.7,31.5,32,33,34,35,36])
        a = np.array([0,2,4,6,8,10,12,14,16,17,18,19,19.5,20,20.5,21.0,21.5,22.0,22.5,23.0,23.5,24.0,25.0,25.5,26.0,26.5,27.0,27.5,28.0,28.5,29.0,29.7,30.7,31.5,32,33,34,35,36])
    
        # Bins normalized between 0 and 1
        norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
    
        # Color tuple for every bin
#        C = np.array([[255,255,255],
#            [220,220,255],
#            [160,140,255],
#            [112,96,220],
#            [90,70,200],
#            [55,40,165],
#            [20,0,130],
#            [20,100,210],
#            [40,130,240],
#            [80,165,245],
#            [10,145,70], #
#            [25,155,80],
#            [40,170,85],
#            [65,170,95],
#            [75,175,105],
#            [95,185,105],
#            [120,195,110],
#            [125,200,120],
#            [150,205,125],
#            [165,220,125],
#            [190,225,135], #
#            [200,230,140],
#            [220,240,150],
#            [240,240,195],
#            [240,235,140],
#            [240,215,130],
#            [245,200,90],
#            [240,175,75],
#            [230,155,60],
#            [240,135,45],
#            [225,115,0],
#            [250,80,60],
#            [240,15,105],
#            [185,0,55],
#            [100,0,5],
#            [150,0,0],
#            [190,0,0],
#            [220,0,0],
#            [250,0,0]])/255.

        C = np.array([[140,140,140],
            [188,188,188],
            [230,230,230],
            [255,255,255],
            [190,190,255],
            [160,140,255],
            [112,96,220],
            [90,70,200],
            [55,40,165],
            [20,0,130],
            [20,100,210],
            [40,130,240],
            [80,165,245],
            [10,145,70],
            [40,170,85],
            [75,175,105],
            [120,195,110],
            [150,205,125],
            [190,225,135],
            [200,230,140],
            [220,240,150],
            [240,240,195],
            [240,235,140],
            [240,215,130],
            [245,200,90],
            [240,175,75],
            [230,155,60],
            [240,135,45],
            [225,115,0],
            [250,80,60],
            [240,15,105],
#            [185,0,55],
            [140,0,0],
            [190,0,0],
            [100,0,5],
            [120,80,70],
            [140,100,90],
            [180,140,130],
            [225,190,180],
            [248,219,214]])/255.

        print 'temp',len(a),len(C)
        
        # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
        COLORS = []
        for i, n in enumerate(norm):
            COLORS.append((n, C[i]))
    
        cmap = colors.LinearSegmentedColormap.from_list("Temperature", COLORS)
    
        return cmap,a
    
    cmap1,clevs1=cm_temp()

    # Contour and fill the temperature
#    T=m.contourf(x,y,ftemps,T_LEVS,cmap=coltbls.sftemp())
    T=m.contourf(x,y,ftemps,clevs1,cmap=cmap1)

    T.cmap.set_under((1.0, 1.0, 1.0))
    T.cmap.set_over((255/255,0/255,0/255))

    # Contour the pressure
#    P=m.contour(x,y,mslp,V=2,colors='w',linewidths=1.5)
#    plt.clabel(P,inline=1,fontsize=8,fmt='%1.0f',inline_spacing=1)

    #plt.clabel(T,inline=1,fontsize=10)

#    # Convert winds from m/s to kts and then draw barbs    
#    u_wind_kts = u_wind_ms[time] * 3.6
#    v_wind_kts = v_wind_ms[time] * 3.6
#    plt.barbs(x_th,y_th,u_wind_kts[::thin,::thin],\
#        v_wind_kts[::thin,::thin], length=5,\
#        sizes={'spacing':0.2},pivot='middle')

    m.colorbar(T)

    title = '2m Temperature'
#    prodid = 'pmsl'
    units = u"\u00B0" + "C"    

#    file_id = '%s_%s_f%02d' % (dom, prodid, a)
#    np.savetxt(file_id+'.txt',mslp)

    prodid = 'temp'
    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    
    if txtflag == 1:
        np.savetxt(file_id+'.txt',ftemps)

    print "TEMP MIN", np.max(ftemps),"TEMP MAX", np.max(ftemps)
    drawmap(m,curtimestring,dom,a,T, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)

def plot_sfwind(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):
    print("    10M WIND")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)

    # Convert winds from m/s to kts and then draw barbs    
    u_wind_kts = u_wind_ms[time] * 1.9438
    v_wind_kts = v_wind_ms[time] * 1.9438
    windmag = np.power(np.power(u_wind_kts,2)+np.power(v_wind_kts,2), 0.5)
#    WIND_LEVS = range(10,70,2)

    windcolors = ((235,235,235),(215,225,255),(181,201,255),(142,178,255),(127,150,255),(99,112,248),(0,99,255),(0,100,210),
    (0,150,150),(0,160,70),(0,198,51),(50,225,25),(99,235,0),(140,255,0),(198,255,51),(230,255,0),(255,245,0),(255,220,0),
    (255,188,0),(255,125,0),(255,85,0),(255,0,0),(215,0,0),(170,0,0),(105,0,70),(170,0,100),(240,0,130),(240,0,160),
    (245,120,190),(250,190,230),(255,230,235),(255,251,253))
    clevs1 = np.array([0.0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150])
    cmap1 = LinearSegmentedColormap.from_list("wind", cnv_to_rgb(windcolors), N=len(windcolors), gamma=1.0)

#    def cm_wind():
#        """
#        F:
#            vmax=120, vmin=-60
#        C:
#            vmax=50, vmin=-50
#        """
#        # The range of temperature bins in Fahrenheit
#        a = [0,1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82]  # OK
#
#        # Bins normalized between 0 and 1
#        norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
#    
#        # Color tuple for every bin
#        C = np.array([[255,255,255],
#                    [128,255,255],
#                    [112,238,241],
#                    [96,222,228],
#                    [80,205,214],
#                    [64,189,200],
#                    [48,172,186],
#                    [32,156,173],
#                    [16,139,159],
#                    [0,180,50],
#                    [51,195,66],
#                    [102,210,81],
#                    [153,225,97],
#                    [204,240,112],
#                    [255,255,128],
#                    [255,221,82],
#                    [255,166,62],
#                    [255,110,41],
#                    [255,55,20],
#                    [255,0,0],
#                    [215,0,0],
#                    [174,0,0],
#                    [132,0,0],
#                    [91,0,0],
#                    [170,0,255],
#                    [184,35,255],
#                    [198,70,255],
#                    [213,106,255],
#                    [227,141,255],
#                    [241,176,255],
#                    [255,211,255],
#                    [255,190,224],
#                    [255,176,205],
#                    [255,162,186],
#                    [255,147,167],
#                    [255,133,148],
#                    [255,119,129],
#                    [238,92,102],
#                    [220,79,95],
#                    [203,66,88],
#                    [186,53,80],
#                    [168,40,72],
#                    [151,27,65]])/255.
#    
#        print 'wind',len(a),len(C)
#        
#        # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
#        COLORS = []
#        for i, n in enumerate(norm):
#            COLORS.append((n, C[i]))
#    
#        cmap = colors.LinearSegmentedColormap.from_list("Wind Speed", COLORS)
#    
#        return cmap,a
#    
#    cmap1,clevs1=cm_wind()

    W=m.contourf(x,y,windmag,clevs1,cmap=cmap1)

    W.cmap.set_under((1.0, 1.0, 1.0))
    W.cmap.set_over((151/255,27/255,65/255))

    plt.barbs(x_th,y_th,u_wind_kts[::thin,::thin],\
        v_wind_kts[::thin,::thin], length=5,\
        sizes={'spacing':0.2},pivot='middle')

    # Convert Surface Pressure to Mean Sea Level Pressure    

#    stemps = temps[time]+6.5*nc.variables['HGT'][time]/1000.
#    mslp = nc.variables['PSFC'][time]*np.exp(9.81/(287.0*stemps)*nc.variables['HGT'][time])*0.01 + (6.7 * nc.variables['HGT'][time] / 1000)

    #Contour the pressure
#    PLEVS = range(900,1050,5)
#    P=m.contour(x,y,mslp,PLEVS,V=2,colors='r',linewidths=1.5)
#    plt.clabel(P,inline=1,fontsize=8,fmt='%1.0f',inline_spacing=1)

    print "      WIND MAX: ", np.max(windmag)

    title = '10m Wind Speed and Direction'
    prodid = 'wind'
    units = "kt"

    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    
    if txtflag == 1:
        np.savetxt(file_id+'.txt',windmag)

    m.colorbar(W)
    drawmap(m,curtimestring,dom,a,W, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)


def plot_mslp(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):

    print("    PRECIP")
    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)

    stemps = nc.variables['T2'][time]+6.5*nc.variables['HGT'][time]/1000.
    mslp = nc.variables['PSFC'][time]*np.exp(9.81/(287.0*stemps)*nc.variables['HGT'][time])*0.01 + (6.7 * nc.variables['HGT'][time] / 1000)

#    stemps1 = nc1.variables['T2'][time]+6.5*nc1.variables['HGT'][time]/1000.
#    mslp1 = nc1.variables['PSFC'][time]*np.exp(9.81/(287.0*stemps1)*nc1.variables['HGT'][time])*0.01 + (6.7 * nc1.variables['HGT'][time] / 1000)
#
#    # First, find out if this is first time or not
#    # Based on skip.  This should be total from each output time
#    if time == 0:
##        prev_total = rainc[time] + rainnc[time]
##    else:
#        prev_total = mslp1
#    total_accum = mslp
#    precip_tend = total_accum  - prev_total
#
#    # Activar aqui grafico de tendencia de la presion

    units = 'hPa'

    def cm_precip():
        """
        Range of values:
            metric: 0 to 762 millimeters
            english: 0 to 30 inches
        """
        # The amount of precipitation in inches
#        a = [0,  .01,   .1, .25, .5,1,1.5,2, 3, 4, 6, 8,10,15,20,30]
#        a = [0,  0.01, 0.6,   1,  2,4  ,6,8,12,17,21,25,33,40,50,70]   # OK
        a = [800,850,900,950,970,990,1000,1002,1004,1006,1008,1010,1012,1013,1014,1015,1016,1017,1018,1019,1020,1022,1025,1028,1030,1035,1040,1050]

        clevs = np.array(a)
        # Normalize the bin between 0 and 1 (uneven bins are important here)
        norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]

        C = np.array([[100,0,0],
                        [170,0,0],
                        [215,0,0],
                        [255,0,0],
                        [255,30,30],
                        [255,70,70],
                        [255,90,90],
                        [255,110,110],
                        [255,135,135],
                        [255,175,175],
                        [255,190,190],
                        [255,210,210],
                        [255,230,230],
                        [255,240,240],
                        [255,255,255],
                        [240,240,255],
                        [230,230,255],
                        [210,210,255],
                        [190,190,255],
                        [175,175,255],
                        [135,135,255],
                        [110,110,255],
                        [90,90,255],
                        [70,70,255],
                        [30,30,255],
                        [0,0,255],
                        [0,0,215],
                        [0,0,170],
                        [0,0,100]])

        print 'mslp',len(a),len(C)
        # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
        COLORS = []
        for i, n in enumerate(norm):
            COLORS.append((n, np.array(C[i])/255.))
    
        # Create the colormap
        cmap = colors.LinearSegmentedColormap.from_list("precipitation", COLORS)
    
        return cmap,norm,clevs

    cmap1,norm1,clevs1 = cm_precip()

    PRECIP=m.contourf(x,y,mslp,clevs1,cmap=cmap1)#,extend="max")
#    PRECIP=m.contourf(x,y,precip_tend,clevs1,norm=norm1,cmap=cmap1,extend="max")
    print "      MAX: ",np.max(mslp)

    PRECIP.cmap.set_under((100/255,0/255,0/255))
    PRECIP.cmap.set_over((0/255,0/255,100/255))

    title = ' MSLP (hPa)'
    prodid = 'mslp'

    m.colorbar(PRECIP)

    # Contour the pressure
    P=m.contour(x,y,mslp,clevs1,colors='k',linewidths=0.9)
    plt.clabel(P,inline=1,fontsize=8,fmt='%1.0f',inline_spacing=1)

    # using cartopy
    # Use definition to plot H/L symbols
    #    plot_maxmin_points(lons, lats, mslp, 'max', 50, symbol='H', color='b',  transform=dataproj)
    #    plot_maxmin_points(lons, lats, mslp, 'min', 25, symbol='L', color='r', transform=dataproj)

    # Using basemap
    mode = 'wrap'
    window = 70
    local_min, local_max = extrema(mslp, mode, window)

    prmsl, lons = addcyclic(mslp, x)

    xlows = x[local_min]; xhighs = x[local_max]
    ylows = y[local_min]; yhighs = y[local_max]
    lowvals = prmsl[local_min]; highvals = prmsl[local_max]

    # plot lows as blue L's, with min pressure value underneath.
    xyplotted = []
    # don't plot if there is already a L or H within dmin meters.
    yoffset = 0.022*(m.ymax-m.ymin)
    dmin = yoffset

    for x,y,p in zip(xlows, ylows, lowvals):
        print x, m.xmax, x, m.xmin, y, m.ymax, y, m.ymin
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                plt.text(x,y,'B',fontsize=9,fontweight='bold',
                         ha='center',va='center',color='r')
                plt.text(x,y-yoffset,repr(int(p)),fontsize=5,
                         ha='center',va='top',color='r',
                         bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
                xyplotted.append((x,y))
    # plot highs as red H's, with max pressure value underneath.
    xyplotted = []

    for x,y,p in zip(xhighs, yhighs, highvals):
        if x < m.xmax and x > m.xmin and y < m.ymax and y > m.ymin:
            dist = [np.sqrt((x-x0)**2+(y-y0)**2) for x0,y0 in xyplotted]
            if not dist or min(dist) > dmin:
                plt.text(x,y,'A',fontsize=9,fontweight='bold',
                         ha='center',va='center',color='b')
                plt.text(x,y-yoffset,repr(int(p)),fontsize=5,
                         ha='center',va='top',color='b',
                    bbox = dict(boxstyle="square",ec='None',fc=(1,1,1,0.5)))
            xyplotted.append((x,y))


    print "      MSLP MAX: ", np.max(mslp)
    file_id = '%s_%s_f%02d' % (dom, prodid, a)
    
    if txtflag == 1:
        np.savetxt(file_id+'.txt',mslp)
    
    drawmap(m,curtimestring,dom,a,PRECIP, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)     



def plot_htwind(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag):

    print("    Geo Pot Levels WIND")

    # Set Figure Size (1000 x 800)
    plt.figure(figsize=(width,height),frameon=False,dpi=300)

    # Extract the pressure, geopotential height, and wind variables
    p = getvar(nc, "pressure")
    z = getvar(nc, "z", units="dm")
    ua = getvar(nc, "ua", units="kt")
    va = getvar(nc, "va", units="kt")
    wspd = getvar(nc, "wspd_wdir", units="kts")[0,:]
    
    # Interpolate geopotential height, u, and v winds to 500 hPa
    hlevels = [925,850,700,500,200]
    
    for hl in hlevels:
        ht_500 = to_np(interplevel(z, p, hl))
        u_500 = to_np(interplevel(ua, p, hl))
        v_500 = to_np(interplevel(va, p, hl))
        wspd_hgt = to_np(interplevel(wspd, p, hl))

        def cm_wind():
            """
            F:
                vmax=120, vmin=-60
            C:
                vmax=50, vmin=-50
            """
            # The range of temperature bins in Fahrenheit
            a = [0,1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82]  # OK
    
            # Bins normalized between 0 and 1
            norm = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
        
            # Color tuple for every bin
            C = np.array([[255,255,255],
                        [128,255,255],
                        [112,238,241],
                        [96,222,228],
                        [80,205,214],
                        [64,189,200],
                        [48,172,186],
                        [32,156,173],
                        [16,139,159],
                        [0,180,50],
                        [51,195,66],
                        [102,210,81],
                        [153,225,97],
                        [204,240,112],
                        [255,255,128],
                        [255,221,82],
                        [255,166,62],
                        [255,110,41],
                        [255,55,20],
                        [255,0,0],
                        [215,0,0],
                        [174,0,0],
                        [132,0,0],
                        [91,0,0],
                        [170,0,255],
                        [184,35,255],
                        [198,70,255],
                        [213,106,255],
                        [227,141,255],
                        [241,176,255],
                        [255,211,255],
                        [255,190,224],
                        [255,176,205],
                        [255,162,186],
                        [255,147,167],
                        [255,133,148],
                        [255,119,129],
                        [238,92,102],
                        [220,79,95],
                        [203,66,88],
                        [186,53,80],
                        [168,40,72],
                        [151,27,65]])/255.
        
            print 'wind',len(a),len(C)
            
            # Create a tuple for every color indicating the normalized position on the colormap and the assigned color.
            COLORS = []
            for i, n in enumerate(norm):
                COLORS.append((n, C[i]))
        
            cmap = colors.LinearSegmentedColormap.from_list("Wind Speed", COLORS)
        
            return cmap,a
        
        cmap1,clevs1=cm_wind()
    
#        ht_500 = to_np(interplevel(z, p, hl))
#        u_500 = to_np(interplevel(ua, p, hl))
#        v_500 = to_np(interplevel(va, p, hl))
#        wspd_hgt = to_np(interplevel(wspd, p, hl))
    
        W=m.contourf(x,y,wspd_hgt,clevs1,cmap=cmap1)
    
        W.cmap.set_under((1.0, 1.0, 1.0))
        W.cmap.set_over((151/255,27/255,65/255))
    
        plt.barbs(x_th,y_th,u_500[::thin,::thin],\
            v_500[::thin,::thin], length=5,\
            sizes={'spacing':0.2},pivot='middle')
    
        # Contour the pressure
        P=m.contour(x,y,ht_500,V=2,colors='w',linewidths=1.5)
        plt.clabel(P,inline=1,fontsize=8,fmt='%1.0f',inline_spacing=1)
    
        print "      WIND HGT "+str(hl)+" MAX: ", np.max(wspd_hgt)
        print "      GEOP HGT "+str(hl)+" MAX: ", np.max(ht_500)
    
        title = str(hl)+'mb Wind Speed and Direction'
        prodid = str(hl)+'mb_wind'
        units = "kt"
    
        file_id = '%s_%s_f%02d' % (dom, prodid, a)
        
        if txtflag == 1:
            np.savetxt(file_id+'.txt',wspd_hgt)
    
        m.colorbar(W)
        drawmap(m,curtimestring,dom,a,W, title, prodid, units, llcrnrlon, llcrnrlat, ftimes)


def cnv_to_rgb(clist):

    newcolors = []
    for i in range(len(clist)):
        newcolors.append((float(clist[i][0])/255,float(clist[i][1])/255,float(clist[i][2])/255))

    return newcolors



#*********************************************************************************************************

def dataproc(lista,a,outdir,skip,restart_time,dom,var,export_flag,filename):

    ftimes = len(lista)

    nc = Dataset(lista[a], "r")
    if a != 0:
        nc1 = Dataset(lista[a-1], "r")
    else:
        nc1 = Dataset(lista[a], "r")

    # Grab these variables for now
    temps =  nc.variables['T2']
    u_wind_ms = nc.variables['U10']
    v_wind_ms = nc.variables['V10']
    psfc = nc.variables['PSFC']
    T = nc.variables['T']
    times = nc.variables['Times']

    # Thin factor is used for thinning out wind barbs
    thin = 10

    # BEGIN ACTUAL PROCESSING HERE
    # x_dim and y_dim are the x and y dimensions of the model
    # domain in gridpoints
    x_dim = len(nc.dimensions['west_east'])
    y_dim = len(nc.dimensions['south_north'])

    # Get the grid spacing
    dx = float(nc.DX)
    dy = float(nc.DY)

    width_meters = dx * (x_dim - 1)
    height_meters = dy * (y_dim - 1)

    cen_lat = float(nc.CEN_LAT)
    cen_lon = float(nc.CEN_LON)
    truelat1 = float(nc.TRUELAT1)
    truelat2 = float(nc.TRUELAT2)
    standlon = float(nc.STAND_LON)
    truelat1 = float(60)
    truelat2 = float(10)

    # Draw the base map behind it with the lats and
    # lons calculated earlier
#    m = Basemap(resolution='h',projection='lcc',\
#        width=width_meters,height=height_meters,\
#        lat_0=cen_lat,lon_0=cen_lon,lat_1=truelat1,\
#        lat_2=truelat2)

    m = Basemap(llcrnrlon=np.min(nc.variables['XLONG'][0]),llcrnrlat=np.min(nc.variables['XLAT'][0]),urcrnrlon=np.max(nc.variables['XLONG'][0])+0.25,urcrnrlat=np.max(nc.variables['XLAT'][0]),
             resolution='h', projection='cyl', lat_0=cen_lat,lon_0=cen_lon)

    # This sets the standard grid point structure at full resolution
    x,y = m(nc.variables['XLONG'][0],nc.variables['XLAT'][0])
    xlat,ylons = m(cen_lon,cen_lat)  # Para hacer marca en algun punto de interes

    # This sets a thinn-ed out grid point structure for plotting
    # wind barbs at the interval specified in "thin"
    x_th,y_th = m(nc.variables['XLONG'][0,::thin,::thin],\
        nc.variables['XLAT'][0,::thin,::thin])

    # Set universal figure margins
    width = 10
    height = 8

    plt.figure(figsize=(width,height),dpi=300)
#    plt.tight_layout()
    plt.rc("figure.subplot", left = .001)
    plt.rc("figure.subplot", right = .999)
    plt.rc("figure.subplot", bottom = .001)
    plt.rc("figure.subplot", top = .999)


    # Check to see if we are exporting
    if export_flag == 1:
        dom = 'wrf'

    time = 0
    curtimestring = timestring(''.join(times[0]),a)

    llcrnrlon=np.min(nc.variables['XLONG'][0])
    llcrnrlat=np.min(nc.variables['XLAT'][0])

    txtflag = 0  # Para activar escritura de ficheros de textos activar aqui poniendo =1

    if var == 'temp':
        plot_surface(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    elif var == 'precip':
        plot_precip(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    elif var == 'olr':
        plot_olr(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    elif var == 'pblh':
        plot_pblh(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    elif var == 'cref':
        plot_comp_reflect(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    elif var == 'wind':
        plot_sfwind(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    elif var == 'hgtwind':
        plot_htwind(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
    else:
        plot_surface(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
        plot_precip(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
#        plot_comp_reflect(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
        plot_sfwind(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
        plot_olr(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
#        plot_pblh(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
#        plot_htwind(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)
        plot_mslp(width,height,nc,nc1,temps,u_wind_ms,v_wind_ms,psfc,T,times,thin,m,x,y,xlat,ylons,x_th,y_th,time,dom,curtimestring,a,llcrnrlon,llcrnrlat,ftimes,txtflag)

    if export_flag == 1:
        os.system('mv *.gif %s' % outdir)


def plot_maxmin_points(lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=24,
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                transform=transform)
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, size=12, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=transform)


def extrema(mat,mode,window):

    """find the indices of local extrema (min and max)
    in the input array."""

    mn = minimum_filter(mat, size=window, mode=mode)
    mx = maximum_filter(mat, size=window, mode=mode)

    # (mat == mx) true if pixel is equal to the local max
    # (mat == mn) true if pixel is equal to the local in
    # Return the indices of the maxima, minima

    return np.nonzero(mat == mn), np.nonzero(mat == mx)

def main(PROCESS_LIMIT):

    # Set the default domain to be d02
    dom = 'd02'
    var = 'all'
    export_flag = 0
    filename = '../wrfout_' + dom
    
    restart_time = 0
    
    # Set up a command-line argument structure to allow
    # for command-line changes of variables.
    # f --> the name of the domain we want to use
    (opts,args)=getopt.getopt(sys.argv[1:],'f:v:r:e')
    for o,a in opts:
        if o=="-f":
            filename = a
        if o=="-v":
            var = str(a)
        if o=="-e":
            export_flag = 1    
        if o=="-r":
            restart_time = int(a)
    
    # Skip is the length between outputs
    skip =0.5 
    
    # Directory to move images to (if requested)
    outdir = './images'
    
    lista = glob.glob("wrfout_d02*:00")
    lista.sort()
    
    # paralelizar aqui
    for a in range(len(lista)):
        
#        process=multiprocessing.Process(target=dataproc,args=(lista,a,outdir,skip,restart_time,dom,var,export_flag,filename))
#        while(len(multiprocessing.active_children()) == PROCESS_LIMIT):
#            time.sleep(1)
#        process.start()
        
        dataproc(lista,a,outdir,skip,restart_time,dom,var,export_flag,filename)

if __name__ == '__main__':
    main(PROCESS_LIMIT)

# separar producto de lluvia
# qpf 24 horass
# mslp con lineas discontinuas de espesor 500 y 1000mb
# rh y wind levels
# vort 500
# w wind levels
# cape y K
# Td

# productos de anomalias
# productos de ligthnings
# productos de temp max y min

# producto de asentamiento por estaciones

# revisar tropical tirbbits y nhc, productos para el tropico
# logo CFA/INSMET

# ver elier pa preparar la 232 para poner estos scripts y parar el archivado de descargas en la 232

