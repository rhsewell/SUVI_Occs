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

start_time=datetime.now()
dusk_starts=[]
dusk_ends=[]
dawn_starts=[]
dawn_ends=[]
cross_sec=scipy.io.readsav(r'C:\Users\Robert\Documents\SUVI_Occs\photon_cross_sections.sav')
n2=cross_sec['photo'].N2[0].XSECTION[0]
o=cross_sec['photo'].O3P[0].XSECTION[0]
wave=cross_sec['photo'].ANGSTROMS[0]
for file in os.listdir("C:\\Users\\Robert\\Documents\\SUVI_Occs\\EXIS_data\\"):
    file='C:\\Users\\Robert\\Documents\\SUVI_Occs\\EXIS_data\\'+file
    ds = nc.Dataset(file)
    
    dusk_inds=np.where(ds['SC_eclipse_flag'][:].data == 1 )
    dawn_inds=np.where(ds['SC_eclipse_flag'][:].data == 3)
    
    epoch = datetime(2000, 1, 1, 12, 0, 0)
    seconds_dusk_start = float(ds['time'][dusk_inds[0][0]].data)
    dusk_start = epoch + timedelta(seconds=seconds_dusk_start)
    
    seconds_dusk_end = float(ds['time'][dusk_inds[0][-1]].data)
    dusk_end = epoch + timedelta(seconds=seconds_dusk_end)
    
    seconds_dawn_start = float(ds['time'][dawn_inds[0][0]].data)
    dawn_start = epoch + timedelta(seconds=seconds_dawn_start)
    
    seconds_dawn_end = float(ds['time'][dawn_inds[0][-1]].data)
    dawn_end = epoch + timedelta(seconds=seconds_dawn_end)
    
    #dusk_starts.append(dusk_start)
    dusk_ends.append(dusk_end)
    dawn_starts.append(dawn_start)
    #dawn_ends.append(dawn_end)
    
    #print('Dusk: ',dusk_start.strftime('%Y%j%H%M%S%f')[:-5],dusk_end.strftime('%Y%j%H%M%S%f')[:-5])
    #print('Dawn: ',dawn_start.strftime('%Y%j%H%M%S%f')[:-5],dawn_end.strftime('%Y%j%H%M%S%f')[:-5],'\n')

custom_lines = [Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='red', lw=4),
                Line2D([0], [0], color='purple', lw=4),
                Line2D([0], [0], color='orange', lw=4)]

fig, ax = plt.subplots()
outlier=[]
outlier_alt=[]
outlier_type=[]
bands=['Fe171','He304','Fe284','Fe195']
for i in range(0,len(dusk_ends)-1):
    avg_er_list=[]
    alt_list=[]
    channels=[]
    fe171_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe171/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    he304_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-he304/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    fe284_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe284/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    fe195_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe195/'+dusk_ends[i].strftime('%Y')+'/'+dusk_ends[i].strftime('%m')+'/'+dusk_ends[i].strftime('%d')+'/'
    urls=[fe171_url,he304_url,fe284_url,fe195_url]
    for j,url in enumerate(urls):    
        topside=None
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        for link in soup.find_all('a'):
            current_link = link.get('href')
            if current_link.endswith('.fits.gz'):
                res=re.search('.+s([0-9]+)_e([0-9]+)_c.+',current_link)
                if res:
                    obs_start=datetime.strptime(res.group(1), "%Y%j%H%M%S%f")
                    obs_end=datetime.strptime(res.group(2), "%Y%j%H%M%S%f")
                    if (obs_start<=(dusk_ends[i]-timedelta(minutes=6))) & (obs_start>=(dusk_ends[i]-timedelta(minutes=15))) & ((obs_end-obs_start).seconds == 1):
                        topside=current_link
                    if (obs_start>=dusk_ends[i]-timedelta(seconds=35)) & (obs_end<=(dusk_ends[i]+timedelta(minutes=1,seconds=15))) & ((obs_end-obs_start).seconds == 1) & (topside is not None):
                        occultation=current_link     
                        hdr = fits.Header()
                        hdr['TopFile'] = topside
                        hdr['OccFile'] = occultation
                        hdr['Channel'] = bands[j]
                        hdr['OccType'] = 'Dusk'
                        er, avg_er, median_er, freq_er, alt, hdr['DATE-OBS'], hdr['OBSGEO-X'], hdr['OBSGEO-Y'], hdr['OBSGEO-Z'], occ_gse=SUVI_ER.makeER(url+occultation,url+topside,'dusk')
                        hdr.comments['DATE-OBS']='sun observation start time on sat'
                        hdr.comments['OBSGEO-X']='[m] observing platform ECEF X coordinate'
                        hdr.comments['OBSGEO-Y']='[m] observing platform ECEF Y coordinate'
                        hdr.comments['OBSGEO-Z']='[m] observing platform ECEF Z coordinate'
                        hdr['OBSGSE-X'] = occ_gse.x.value
                        hdr['OBSGSE-Y'] = occ_gse.y.value
                        hdr['OBSGSE-Z'] = occ_gse.z.value
                        hdr.comments['OBSGSE-X']='[m] observing platform GSE X coordinate'
                        hdr.comments['OBSGSE-Y']='[m] observing platform GSE Y coordinate'
                        hdr.comments['OBSGSE-Z']='[m] observing platform GSE Z coordinate'
                        primary_hdu=fits.PrimaryHDU(er,header=hdr)
                        c1 = fits.Column(name='Avg ER', array=avg_er, format='D')
                        c2 = fits.Column(name='Median ER', array=median_er, format='D')
                        c3 = fits.Column(name='Max Hist Freq ER', array=freq_er, format='D')
                        c4 = fits.Column(name='Altitude', array=alt, format='D')
                        table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
                        hdul = fits.HDUList([primary_hdu, table_hdu])
                        #zipfitsfile=gzip.open('C:\\Users\\Robert\\Documents\\SUVI_Occs\\ER_data\\'+res.group(1)+'_'+bands[j]+'.fits.gz','wb')
                        #hdul.writeto(zipfitsfile,overwrite=True)
                        #plt.plot(avg_occ,alt,'red')
                        avg_er_list.append(avg_er)
                        alt_list.append(alt)
                        channels.append(bands[j])
                        if avg_er[np.abs(alt - 100.).argmin()]>0.01:
                            outlier.append(occultation)
                            outlier_alt.append(alt[np.abs(alt - 100.).argmin()])
                            outlier_type.append('Dusk')
    ref_alt,col_density=SUVI_ER.make_col_density(np.array(channels),np.array(avg_er_list),np.array(alt_list),o,n2,wave)
    plt.plot(ref_alt,col_density[0].T,'blue')
    plt.plot(ref_alt,col_density[1].T,'red')
                    
for i in range(0,len(dawn_starts)-1):
    avg_er_list=[]
    alt_list=[]
    channels=[]
    fe171_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe171/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    he304_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-he304/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    fe284_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe284/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    fe195_url='https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/suvi-l1b-fe195/'+dawn_starts[i].strftime('%Y')+'/'+dawn_starts[i].strftime('%m')+'/'+dawn_starts[i].strftime('%d')+'/'
    urls=[fe171_url,he304_url,fe284_url,fe195_url]
    for j,url in enumerate(urls):
        topside=None
        page = requests.get(url)
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
                    if (obs_start>=(dawn_starts[i]+timedelta(minutes=6))) & (obs_start<=(dawn_starts[i]+timedelta(minutes=15))) & ((obs_end-obs_start).seconds == 1):
                        topside=current_link
                    if (obs_start>=(dawn_starts[i]-timedelta(minutes=1,seconds=15))) & (obs_end<=dawn_starts[i]+timedelta(seconds=35)) & ((obs_end-obs_start).seconds == 1) & (topside is not None):
                        occultation=current_link
                        hdr = fits.Header()
                        hdr['TopFile'] = topside
                        hdr['OccFile'] = occultation
                        hdr['Channel'] = bands[j]
                        hdr['OccType'] = 'Dawn'
                        er, avg_er, median_er, freq_er, alt, hdr['DATE-OBS'], hdr['OBSGEO-X'], hdr['OBSGEO-Y'], hdr['OBSGEO-Z'], occ_gse=SUVI_ER.makeER(url+occultation,url+topside,'dawn')
                        hdr.comments['DATE-OBS']='sun observation start time on sat'
                        hdr.comments['OBSGEO-X']='[m] observing platform ECEF X coordinate'
                        hdr.comments['OBSGEO-Y']='[m] observing platform ECEF Y coordinate'
                        hdr.comments['OBSGEO-Z']='[m] observing platform ECEF Z coordinate'
                        hdr['OBSGSE-X'] = occ_gse.x.value
                        hdr['OBSGSE-Y'] = occ_gse.y.value
                        hdr['OBSGSE-Z'] = occ_gse.z.value
                        hdr.comments['OBSGSE-X']='[m] observing platform GSE X coordinate'
                        hdr.comments['OBSGSE-Y']='[m] observing platform GSE Y coordinate'
                        hdr.comments['OBSGSE-Z']='[m] observing platform GSE Z coordinate'
                        primary_hdu=fits.PrimaryHDU(er,header=hdr)
                        c1 = fits.Column(name='Avg ER', array=avg_er, format='D')
                        c2 = fits.Column(name='Median ER', array=median_er, format='D')
                        c3 = fits.Column(name='Max Hist Freq ER', array=freq_er, format='D')
                        c4 = fits.Column(name='Altitude', array=alt, format='D')
                        table_hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
                        hdul = fits.HDUList([primary_hdu, table_hdu])
                        #zipfitsfile=gzip.open('C:\\Users\\Robert\\Documents\\SUVI_Occs\\ER_data\\'+res.group(1)+'_'+bands[j]+'.fits.gz','wb')
                        #hdul.writeto(zipfitsfile,overwrite=True)
                        #plt.plot(avg_occ,alt,'blue')
                        avg_er_list.append(avg_er)
                        alt_list.append(alt)
                        channels.append(bands[j])
                        if avg_er[np.abs(alt - 100.).argmin()]>0.01:
                            outlier.append(occultation)
                            outlier_alt.append(alt[np.abs(alt - 100.).argmin()])
                            outlier_type.append('Dawn')
    ref_alt,col_density=SUVI_ER.make_col_density(np.array(channels),np.array(avg_er_list),np.array(alt_list),o,n2,wave)
    plt.plot(ref_alt,col_density[0].T,'purple')
    plt.plot(ref_alt,col_density[1].T,'orange')

ax.legend(custom_lines, ['Dusk O', 'Dusk N2','Dawn O', 'Dawn N2'])
print(datetime.now()-start_time)
print(outlier)
