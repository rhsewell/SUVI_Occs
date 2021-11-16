"""
SUVI Occultations Retreival

Science Developement Code

Procedure:
    suvi_occ_times.py

Purpose:
    Find occulations windows for dusk and dawn occulations, find SUVI channel
    images in given windows, create ER data from channel data, create column
    densities using all channels in a given window

Requirements:
    - Downloaded EXIS level 1b daily data
    - Package dependancies below
    
Calling Sequence:
    Just run it
    
Author:
    Robert Sewell
"""

import netCDF4 as nc
import numpy as np
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
import SUVI_ER
from astropy.table import Table
import astropy.io.fits as fits
import gzip
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.io
from time import sleep
from scipy import interpolate

#Set up lists for occ windows
dusk_starts=[]
dusk_ends=[]
dawn_starts=[]
dawn_ends=[]

#Channels we can use
cs_wv=np.array(['Fe171','Fe195','Fe284','He304'])

#Interp for O2 data
#---Need effective cross section values for O2!-------
cross_sec=scipy.io.readsav(r'C:\Users\Robert\Documents\SUVI_Occs\photon_cross_sections.sav')
o2=cross_sec['photo'].O2[0].XSECTION[0][:,0]
wave=cross_sec['photo'].ANGSTROMS[0]
f_o2=interpolate.interp1d(wave,o2,kind='linear')

#Ed Thiemann effective cross section values
cs_o=np.array([3.3553367992432838e-018,4.1293458431227207e-018,
               7.6977052093263391e-018,8.0240236916888457e-018])
cs_n2=np.array([4.3016756367776931e-018,5.6801732421666216e-018,
                1.1664286662545586e-017,1.1769712733339984e-017])
cs_o2=f_o2([171,195,284,304])

#Exis time epoch
epoch = datetime(2000, 1, 1, 12, 0, 0)

#Loop over downloaded EXIS data find eclipse flag to define occ windows 
#reference
for file in os.listdir("C:\\Users\\Robert\\Documents\\SUVI_Occs\\EXIS_data\\"):
    file='C:\\Users\\Robert\\Documents\\SUVI_Occs\\EXIS_data\\'+file
    ds = nc.Dataset(file)
    
    dusk_inds=np.where(ds['SC_eclipse_flag'][:].data == 1)
    dawn_inds=np.where(ds['SC_eclipse_flag'][:].data == 3)
    
    if (dusk_inds[0].size > 0) & (dawn_inds[0].size > 0):
        seconds_dusk_start = float(ds['time'][dusk_inds[0][0]].data)
        dusk_start = epoch + timedelta(seconds=seconds_dusk_start)
        
        seconds_dusk_end = float(ds['time'][dusk_inds[0][-1]].data)
        dusk_end = epoch + timedelta(seconds=seconds_dusk_end)
        
        seconds_dawn_start = float(ds['time'][dawn_inds[0][0]].data)
        dawn_start = epoch + timedelta(seconds=seconds_dawn_start)
        
        seconds_dawn_end = float(ds['time'][dawn_inds[0][-1]].data)
        dawn_end = epoch + timedelta(seconds=seconds_dawn_end)
        
        dusk_ends.append(dusk_end)
        dawn_starts.append(dawn_start)

#Order of pulling SUVI channel data
bands=['Fe171','He304','Fe284','Fe195']
dates=[]

#Loop over all dusk occs to find channel data
#-----Migrate to function for both dusk and dawn-------
for i in range(0,len(dusk_ends)-1):
    #Set up lists needed for column density file
    avg_er_list=[]
    alt_list=[]
    channels=[]
    obsgeo_lat=[]
    obsgeo_lon=[]
    obsgeo_height=[]    
    atmosgeo_lat=[]
    atmosgeo_lon=[]
    atmosgeo_height=[]
    time=[]
    
    #Urls for SUVI channel data for each occultation day 
    fe171_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe171/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    he304_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-he304/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    fe284_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe284/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    fe195_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe195/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    urls=[fe171_url,he304_url,fe284_url,fe195_url]
    
    #Loop over Urls
    for j,url in enumerate(urls):    
        topside=None
        #Handle timeouts when requesting page
        for x in range(0, 10):  # try 10 times
            try:
                page = requests.get(url)
                str_error = None
            except Exception as str_error:
                sleep(10)  # wait for 10 seconds before trying to fetch the data again
            else:
                break
            
        #Parse SUVI channel urls for files that occur in occ window
        soup = BeautifulSoup(page.content, 'html.parser')
        for link in soup.find_all('a'):
            current_link = link.get('href')
            if current_link.endswith('.fits.gz'):
                res=re.search('.+s([0-9]+)_e([0-9]+)_c.+',current_link)
                if res:
                    obs_start=datetime.strptime(res.group(1), "%Y%j%H%M%S%f")
                    obs_end=datetime.strptime(res.group(2), "%Y%j%H%M%S%f")
                    #Find topside (non-occulted) image
                    if (obs_start<=(dusk_ends[i]-timedelta(minutes=6))) & (obs_start>=(dusk_ends[i]-timedelta(minutes=15))) & ((obs_end-obs_start).seconds == 1):
                        topside=current_link
                    #Find occultation image
                    if (obs_start>=dusk_ends[i]-timedelta(seconds=35)) & (obs_end<=(dusk_ends[i]+timedelta(minutes=1,seconds=15))) & ((obs_end-obs_start).seconds == 1) & (topside is not None):
                        occultation=current_link   
                        
                        #Set up ER fits file
                        hdr = fits.Header()
                        hdr['OccFile'] = occultation
                        hdr['Channel'] = bands[j]
                        hdr['OccType'] = 'Dusk'
                        
                        #Make ER profile and get all location info
                        er, top, occ, alt_map, avg_er, median_er, freq_er, avg_er2, median_er2, freq_er2, alt, alt2, hdr['DATE-OBS'], hdr['OBSECEF-X'], hdr['OBSECEF-Y'], hdr['OBSECEF-Z'], occ_gse, occ_geo, observer_atmos_gse,observer_atmos_ecef,observer_atmos_geo=SUVI_ER.makeER(url+occultation,url+topside,'dusk',bands[j])
                        
                        #Make fits (------Migrate to Function!-------)
                        hdr.comments['DATE-OBS']='sun observation start time on sat'
                        hdr.comments['OBSECEF-X']='[m] observing platform ECEF X coordinate'
                        hdr.comments['OBSECEF-Y']='[m] observing platform ECEF Y coordinate'
                        hdr.comments['OBSECEF-Z']='[m] observing platform ECEF Z coordinate'
                        hdr['OBSGSE-X'] = occ_gse.x.value
                        hdr['OBSGSE-Y'] = occ_gse.y.value
                        hdr['OBSGSE-Z'] = occ_gse.z.value
                        hdr.comments['OBSGSE-X']='[m] observing platform GSE X coordinate'
                        hdr.comments['OBSGSE-Y']='[m] observing platform GSE Y coordinate'
                        hdr.comments['OBSGSE-Z']='[m] observing platform GSE Z coordinate'
                        hdr['OBSGEO-Lat'] = occ_geo.lat.value
                        hdr['OBSGEO-Lon'] = occ_geo.lon.value
                        hdr['OBSGEO-H'] = occ_geo.height.value
                        hdr.comments['OBSGEO-Lat']='[deg] observing platform GEO Lat coordinate'
                        hdr.comments['OBSGEO-Lon']='[deg] observing platform GEO Lon coordinate'
                        hdr.comments['OBSGEO-H']='[m] observing platform GEO Height coordinate'
                        
                        hdr['ATECEF-X'] = observer_atmos_ecef.x.value
                        hdr['ATECEF-Y'] = observer_atmos_ecef.y.value
                        hdr['ATECEF-Z'] = observer_atmos_ecef.z.value
                        hdr.comments['ATECEF-X']='[m] center pixel location in atmopshere ECEF X coordinate'
                        hdr.comments['ATECEF-Y']='[m] center pixel location in atmopshere ECEF Y coordinate'
                        hdr.comments['ATECEF-Z']='[m] center pixel location in atmopshere ECEF Z coordinate'
                        hdr['ATGSE-X'] = observer_atmos_gse.x.value
                        hdr['ATGSE-Y'] = observer_atmos_gse.y.value
                        hdr['ATGSE-Z'] = observer_atmos_gse.z.value
                        hdr.comments['ATGSE-X']='[m] center pixel location in atmopshere GSE X coordinate'
                        hdr.comments['ATGSE-Y']='[m] center pixel location in atmopshere GSE Y coordinate'
                        hdr.comments['ATGSE-Z']='[m] center pixel location in atmopshere GSE Z coordinate'
                        hdr['AT-Lat'] = observer_atmos_geo.lat.value
                        hdr['AT-Lon'] = observer_atmos_geo.lon.value
                        hdr['AT-H'] = observer_atmos_geo.height.value
                        hdr.comments['AT-Lat']='[deg] center pixel location in atmopshere GEO Lat coordinate (WGS84)'
                        hdr.comments['AT-Lon']='[deg] center pixel location in atmopshere GEO Lon coordinate (WGS84)'
                        hdr.comments['AT-H']='[m] center pixel location in atmopshere GEO Height coordinate (WGS84)'
                        
                        #Add loc and time info to be used in col density file
                        obsgeo_lat.append(occ_geo.lat.value)
                        obsgeo_lon.append(occ_geo.lon.value)
                        obsgeo_height.append(occ_geo.height.value)
                        atmosgeo_lat.append(observer_atmos_geo.lat.value)
                        atmosgeo_lon.append(observer_atmos_geo.lon.value)
                        atmosgeo_height.append(observer_atmos_geo.height.value)
                        time.append(np.mean([float(res.group(1)),float(res.group(2))]))
                        
                        #Make fits with ER image, topside image, occ image, ER
                        #profile tables (from both ER profile methods)
                        #------Migrate to Function!------
                        primary_hdu=fits.PrimaryHDU(er,header=hdr)
                        
                        hdr_top = fits.Header()
                        hdr_top['TopFile'] = topside
                        top_hdu=fits.ImageHDU(top,header=hdr_top)
                        
                        hdr_occ = fits.Header()
                        hdr_occ['OccFile'] = occultation
                        occ_hdu=fits.ImageHDU(occ,header=hdr_occ)
                        
                        hdr_alt_map = fits.Header()
                        hdr_occ['Desc.'] = '[km] Alt Map of Occ'
                        alt_map_hdu=fits.ImageHDU(alt_map,header=hdr_alt_map)
                        
                        hdr_er_avg=fits.Header()
                        hdr_er_avg['AvgType']='Alt avg method: ER Image Processing'
                        c1 = fits.Column(name='Avg ER', array=avg_er, format='D')
                        c2 = fits.Column(name='Median ER', array=median_er, format='D')
                        c3 = fits.Column(name='Max Hist Freq ER', array=freq_er, format='D')
                        c4 = fits.Column(name='Altitude', array=alt, format='D')
                        avg_table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
                        
                        hdr_er_avg2=fits.Header()
                        hdr_er_avg2['AvgType']='Alt avg method: Occ/Top Seperate Image Processing'
                        c1 = fits.Column(name='Avg ER', array=avg_er2, format='D')
                        c2 = fits.Column(name='Median ER', array=median_er2, format='D')
                        c3 = fits.Column(name='Max Hist Freq ER', array=freq_er2, format='D')
                        c4 = fits.Column(name='Altitude', array=alt2, format='D')
                        avg_table_hdu2 = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
                        
                        hdul = fits.HDUList([primary_hdu, top_hdu, occ_hdu, alt_map_hdu, avg_table_hdu, avg_table_hdu2])
                        
                        #Zip fits
                        zipfitsfile=gzip.open('C:\\Users\\Robert\\Documents\\SUVI_Occs\\ER_data\\'+res.group(1)+'_'+bands[j]+'.fits.gz','wb')
                        hdul.writeto(zipfitsfile,overwrite=True)
                        
                        #Add to list what channels are in occ window for col 
                        #Density call
                        avg_er_list.append(avg_er)
                        alt_list.append(alt)
                        channels.append(bands[j])
                        dates.append(dusk_ends[i])
                        
    #Make col density with all avaliable channels
    ref_alt,col_density=SUVI_ER.make_col_density(obs_start,'Dusk',np.array(channels),np.array(avg_er_list),np.array(alt_list),cs_wv,cs_o,cs_n2,cs_o2)
    
    #Write col density data to txt file
    with open(r'C:\\Users\\Robert\\Documents\\SUVI_Occs\\ColDen_data\\'+str(round(np.mean(time)))+'_Dusk_ColDen.txt',"w") as file_out:
        file_out.write("Avg Occ Measurement Time: "+str(np.mean(time))+"\n")
        file_out.write("Avg GOES ECEF-X: "+str(np.mean(obsgeo_lat))+"\n")
        file_out.write("Avg GOES ECEF-Y: "+str(np.mean(obsgeo_lon))+"\n")
        file_out.write("Avg GOES ECEF-Z: "+str(np.mean(obsgeo_height))+"\n")
        file_out.write("Avg Atmosphere ECEF-X: "+str(np.mean(atmosgeo_lat))+"\n")
        file_out.write("Avg Atmosphere ECEF-Y: "+str(np.mean(atmosgeo_lon))+"\n")
        file_out.write("Avg Atmosphere ECEF-Z: "+str(np.mean(atmosgeo_height))+"\n")
        file_out.write("Channels Used: "+', '.join(channels)+"\n")
        file_out.write("Alt [km]      Col Den O [cm^-3]      Col Den N2 [cm^-3]\n")
        for j in range(0,len(ref_alt)-1):
            file_out.write(str(ref_alt[j])+'      '+str(col_density[0,j].T)+'      '+str(col_density[1,j].T)+'\n')
        file_out.close()
    
#Repeat above but for dawn side occ                  
for i in range(0,len(dawn_starts)-1):
    avg_er_list=[]
    alt_list=[]
    channels=[]
    obsgeo_lat=[]
    obsgeo_lon=[]
    obsgeo_height=[]    
    atmosgeo_lat=[]
    atmosgeo_lon=[]
    atmosgeo_height=[]
    time=[]
    fe171_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe171/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    he304_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-he304/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    fe284_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe284/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    fe195_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe195/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    urls=[fe171_url,he304_url,fe284_url,fe195_url]
    for j,url in enumerate(urls):
        topside=None
        for x in range(0, 10):  # try 10 times
            try:
                page = requests.get(url)
                str_error = None
            except Exception as str_error:
                pass
            if str_error:
                sleep(10)  # wait for 10 seconds before trying to fetch the data again
            else:
                break
        soup = BeautifulSoup(page.content, 'html.parser')
        links=soup.find_all('a')
        links.reverse()
        for link in links:
            current_link = link.get('href')
            if current_link.endswith('.fits.gz'):
                res=re.search('.+s([0-9]+)_e([0-9]+)_c.+',current_link)
                if res:
                    obs_start=datetime.strptime(res.group(1), "%Y%j%H%M%S%f")
                    obs_end=datetime.strptime(res.group(2), "%Y%j%H%M%S%f")
                    #Note: window is reversed from above for topside and occ 
                    #as topside image is taken after occ image in this case 
                    if (obs_start>=(dawn_starts[i]+timedelta(minutes=6))) & (obs_start<=(dawn_starts[i]+timedelta(minutes=15))) & ((obs_end-obs_start).seconds == 1):
                        topside=current_link
                    if (obs_start>=(dawn_starts[i]-timedelta(minutes=1,seconds=15))) & (obs_end<=dawn_starts[i]+timedelta(seconds=35)) & ((obs_end-obs_start).seconds == 1) & (topside is not None):
                        occultation=current_link
                        hdr = fits.Header()
                        hdr['TopFile'] = topside
                        hdr['OccFile'] = occultation
                        hdr['Channel'] = bands[j]
                        hdr['OccType'] = 'Dawn'
                        er, top, occ, alt_map, avg_er, median_er, freq_er, avg_er2, median_er2, freq_er2, alt, alt2, hdr['DATE-OBS'], hdr['OBSECEF-X'], hdr['OBSECEF-Y'], hdr['OBSECEF-Z'], occ_gse,occ_geo,observer_atmos_gse,observer_atmos_ecef,observer_atmos_geo=SUVI_ER.makeER(url+occultation,url+topside,'dawn',bands[j])
                        hdr.comments['DATE-OBS']='sun observation start time on sat'
                        hdr.comments['OBSECEF-X']='[m] observing platform ECEF X coordinate'
                        hdr.comments['OBSECEF-Y']='[m] observing platform ECEF Y coordinate'
                        hdr.comments['OBSECEF-Z']='[m] observing platform ECEF Z coordinate'
                        hdr['OBSGSE-X'] = occ_gse.x.value
                        hdr['OBSGSE-Y'] = occ_gse.y.value
                        hdr['OBSGSE-Z'] = occ_gse.z.value
                        hdr.comments['OBSGSE-X']='[m] observing platform GSE X coordinate'
                        hdr.comments['OBSGSE-Y']='[m] observing platform GSE Y coordinate'
                        hdr.comments['OBSGSE-Z']='[m] observing platform GSE Z coordinate'
                        hdr['OBSGEO-Lat'] = occ_geo.lat.value
                        hdr['OBSGEO-Lon'] = occ_geo.lon.value
                        hdr['OBSGEO-H'] = occ_geo.height.value
                        hdr.comments['OBSGEO-Lat']='[deg] observing platform GEO Lat coordinate'
                        hdr.comments['OBSGEO-Lon']='[deg] observing platform GEO Lon coordinate'
                        hdr.comments['OBSGEO-H']='[m] observing platform GEO Height coordinate'
                        
                        hdr['ATECEF-X'] = observer_atmos_ecef.x.value
                        hdr['ATECEF-Y'] = observer_atmos_ecef.y.value
                        hdr['ATECEF-Z'] = observer_atmos_ecef.z.value
                        hdr.comments['ATECEF-X']='[m] center pixel location in atmopshere ECEF X coordinate'
                        hdr.comments['ATECEF-Y']='[m] center pixel location in atmopshere ECEF Y coordinate'
                        hdr.comments['ATECEF-Z']='[m] center pixel location in atmopshere ECEF Z coordinate'
                        hdr['ATGSE-X'] = observer_atmos_gse.x.value
                        hdr['ATGSE-Y'] = observer_atmos_gse.y.value
                        hdr['ATGSE-Z'] = observer_atmos_gse.z.value
                        hdr.comments['ATGSE-X']='[m] center pixel location in atmopshere GSE X coordinate'
                        hdr.comments['ATGSE-Y']='[m] center pixel location in atmopshere GSE Y coordinate'
                        hdr.comments['ATGSE-Z']='[m] center pixel location in atmopshere GSE Z coordinate'
                        hdr['AT-Lat'] = observer_atmos_geo.lat.value
                        hdr['AT-Lon'] = observer_atmos_geo.lon.value
                        hdr['AT-H'] = observer_atmos_geo.height.value
                        hdr.comments['AT-Lat']='[deg] center pixel location in atmopshere GEO Lat coordinate (WGS84)'
                        hdr.comments['AT-Lon']='[deg] center pixel location in atmopshere GEO Lon coordinate (WGS84)'
                        hdr.comments['AT-H']='[m] center pixel location in atmopshere GEO Height coordinate (WGS84)'
                        
                        obsgeo_lat.append(occ_geo.lat.value)
                        obsgeo_lon.append(occ_geo.lon.value)
                        obsgeo_height.append(occ_geo.height.value)
                        atmosgeo_lat.append(observer_atmos_geo.lat.value)
                        atmosgeo_lon.append(observer_atmos_geo.lon.value)
                        atmosgeo_height.append(observer_atmos_geo.height.value)
                        time.append(np.mean([float(res.group(1)),float(res.group(2))]))
                        
                        primary_hdu=fits.PrimaryHDU(er,header=hdr)
                        
                        hdr_top = fits.Header()
                        hdr_top['TopFile'] = topside
                        top_hdu=fits.ImageHDU(top,header=hdr_top)
                        
                        hdr_occ = fits.Header()
                        hdr_occ['OccFile'] = occultation
                        occ_hdu=fits.ImageHDU(occ,header=hdr_occ)
                        
                        hdr_alt_map = fits.Header()
                        hdr_occ['Desc.'] = '[km] Alt Map of Occ'
                        alt_map_hdu=fits.ImageHDU(alt_map,header=hdr_alt_map)
                        
                        hdr_er_avg=fits.Header()
                        hdr_er_avg['AvgType']='Alt avg method: ER Image Processing'
                        c1 = fits.Column(name='Avg ER', array=avg_er, format='D')
                        c2 = fits.Column(name='Median ER', array=median_er, format='D')
                        c3 = fits.Column(name='Max Hist Freq ER', array=freq_er, format='D')
                        c4 = fits.Column(name='Altitude', array=alt, format='D')
                        avg_table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
                        
                        hdr_er_avg2=fits.Header()
                        hdr_er_avg2['AvgType']='Alt avg method: Occ/Top Seperate Image Processing'
                        c1 = fits.Column(name='Avg ER', array=avg_er2, format='D')
                        c2 = fits.Column(name='Median ER', array=median_er2, format='D')
                        c3 = fits.Column(name='Max Hist Freq ER', array=freq_er2, format='D')
                        c4 = fits.Column(name='Altitude', array=alt2, format='D')
                        avg_table_hdu2 = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
                        
                        hdul = fits.HDUList([primary_hdu, top_hdu, occ_hdu, alt_map_hdu, avg_table_hdu, avg_table_hdu2])
                        
                        zipfitsfile=gzip.open('C:\\Users\\Robert\\Documents\\SUVI_Occs\\ER_data\\'+res.group(1)+'_'+bands[j]+'.fits.gz','wb')
                        hdul.writeto(zipfitsfile,overwrite=True)
                        #plt.plot(avg_occ,alt,'blue')
                        avg_er_list.append(avg_er)
                        alt_list.append(alt)
                        channels.append(bands[j])
                        dates.append(dawn_starts[i])
                        
    ref_alt,col_density=SUVI_ER.make_col_density(obs_start,'Dawn',np.array(channels),np.array(avg_er_list),np.array(alt_list),cs_wv,cs_o,cs_n2,cs_o2)
    with open(r'C:\\Users\\Robert\\Documents\\SUVI_Occs\\ColDen_data\\'+str(round(np.mean(time)))+'_Dawn_ColDen.txt',"w") as file_out:
        file_out.write("Avg Occ Measurement Time: "+str(np.mean(time))+"\n")
        file_out.write("Avg GOES GEO-Lat: "+str(np.mean(obsgeo_lat))+"\n")
        file_out.write("Avg GOES GEO-Lon: "+str(np.mean(obsgeo_lon))+"\n")
        file_out.write("Avg GOES GEO-Height: "+str(np.mean(obsgeo_height))+"\n")
        file_out.write("Avg Atmosphere GEO-Lat: "+str(np.mean(atmosgeo_lat))+"\n")
        file_out.write("Avg Atmosphere GEO-Lon: "+str(np.mean(atmosgeo_lon))+"\n")
        file_out.write("Avg Atmosphere GEO-Height: "+str(np.mean(atmosgeo_height))+"\n")
        file_out.write("Channels Used: "+', '.join(channels)+"\n")
        file_out.write("Alt [km]      Col Den O [cm^-3]      Col Den N2 [cm^-3]\n")
        for j in range(0,len(ref_alt)-1):
            file_out.write(str(ref_alt[j])+'      '+str(col_density[0,j].T)+'      '+str(col_density[1,j].T)+'\n')
        file_out.close()
    
