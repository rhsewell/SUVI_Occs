"""
SUVI Occultations Retreival

Science Developement Code

Procedure:
    SUVI_ER.py

Purpose:
    Includes functions required for creating ER profile and deriving column 
    densities

Requirements:
    - Package dependancies below
    
Calling Sequence:
    Functions called by suvi_occ_times.py
    
Author:
    Robert Sewell
"""

import astropy.io.fits as fits
import skimage
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from astropy import units as u
import sunpy.map
import sunpy.image.coalignment
from suvi_fix_L1b_header import fix_suvi_l1b_header
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from mathutils.geometry import intersect_point_line
import scipy.io
import scipy.optimize
import cv2
from skimage.registration import phase_cross_correlation
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
import astropy.coordinates
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV


def get_coords_from_fits(fits_file):
    #Uses sunpy SUVI map to get image coordinates for each channel fits file
    hdu=fits.open(fits_file)
    data=hdu[0].data
    header=fix_suvi_l1b_header(hdu[0].header)
    map_image=sunpy.map.sources.SUVIMap(data, header)
    return data, header, sunpy.map.all_coordinates_from_map(map_image)

def get_terminator_coords(sat_map,rot_mat):
    #Determines the coordinates of the terminator plane using SUVI location and
    #line of sight to the sun
    sat_loc_gse=SkyCoord(sat_map.observer).transform_to(frames.GeocentricSolarEcliptic)
    sun_loc_gse=SkyCoord(0*u.m, 0*u.m, 0*u.m, obstime=sat_map.obstime,frame=frames.Heliocentric,\
                      observer='sun').transform_to(frames.GeocentricSolarEcliptic)
    sat_sun_line=((sat_loc_gse.cartesian.x.value,sat_loc_gse.cartesian.y.value,sat_loc_gse.cartesian.z.value), \
                (sun_loc_gse.cartesian.x.value,sun_loc_gse.cartesian.y.value,sun_loc_gse.cartesian.z.value))
    observer_atmos = intersect_point_line((0.0,0.0,0.0), sat_sun_line[0], sat_sun_line[1])
    observer_atmos = SkyCoord(0*u.m,sat_loc_gse.cartesian.y,sat_loc_gse.cartesian.z,\
                          obstime=sat_loc_gse.obstime,frame=frames.GeocentricSolarEcliptic,representation_type='cartesian')
    observer_atmos.observer=sat_map.observer
    sat_terminator_distance=np.sqrt((sat_loc_gse.cartesian.x-observer_atmos.cartesian.x)**2+\
                          (sat_loc_gse.cartesian.y-observer_atmos.cartesian.y)**2+\
                              (sat_loc_gse.cartesian.z-observer_atmos.cartesian.z)**2)
    hcc_atmos=observer_atmos.transform_to(frames.Heliocentric)
    hcc_sat=SkyCoord(sat_map.observer,observer=sat_map.observer).transform_to(frames.Heliocentric)
    Tx=cv2.warpAffine(np.tan(sat_map.Tx)*sat_terminator_distance, rot_mat, sat_map.Tx.shape[1::-1])
    Ty=cv2.warpAffine(np.tan(sat_map.Ty)*sat_terminator_distance, rot_mat, sat_map.Tx.shape[1::-1])
    zpix=(hcc_atmos.cartesian.x.value*Tx*(hcc_sat.cartesian.x.value**2-hcc_sat.cartesian.x.value*hcc_atmos.cartesian.x.value)+hcc_atmos.cartesian.y.value*Ty*(hcc_sat.cartesian.y.value**2-hcc_sat.cartesian.y.value*hcc_atmos.cartesian.y.value))/(hcc_atmos.cartesian.z.value*(hcc_sat.cartesian.z.value**2-hcc_sat.cartesian.z.value*hcc_atmos.cartesian.z.value))+hcc_atmos.cartesian.z.value
    return observer_atmos,sat_terminator_distance,SkyCoord(Tx*u.m,Ty*u.m,\
                   zpix*u.m,obstime=sat_loc_gse.obstime,\
                       frame=frames.Heliocentric,representation_type='cartesian',observer=observer_atmos.observer).transform_to(frames.GeocentricSolarEcliptic)

def pixels_of_sun(er,radius, x0, y0,radius_multi):
    #Use image sun center and radius in pixels to determine pixels w/ sun
    x_ = np.arange(0, 1280, dtype=int)
    y_ = np.arange(0, 1280, dtype=int)
    pixels_out=np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 > (radius_multi*radius)**2)
    pixels_in=np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= (radius_multi*radius)**2)

    return pixels_out,pixels_in


def horizontalize(img,thresh,typeOcc):
    X = np.array(np.where(thresh > 0)).T
    # Perform a PCA and compute the angle of the first principal axes
    pca = PCA(n_components=2).fit(X)
    angle = np.mod(np.arctan2(*pca.components_[0])/np.pi*180,180)
    print(angle)
    rot_mat = cv2.getRotationMatrix2D((img.shape[0]/2.,img.shape[1]/2.), angle, 1.0)
    # Rotate the image by the computed angle:
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1])
    if np.nanmean(rotated_img[int(img.shape[0]/2.-img.shape[0]/4),int(img.shape[0]/2.-10):int(img.shape[0]/2.+10)]) > np.nanmean(rotated_img[int(img.shape[0]/2.+img.shape[0]/4),int(img.shape[0]/2.-10):int(img.shape[0]/2.+10)]):
        rot_mat = cv2.getRotationMatrix2D((img.shape[0]/2.,img.shape[1]/2.), 180, 1.0)
        rotated_img = cv2.warpAffine(rotated_img, rot_mat, img.shape[1::-1])
    return rotated_img

def make_alt_map(er_image,terminator_coords,sat_terminator_distance):
    #Used instead of sunpy map SUVI
    #Makes altitude map using 2.5 arc sec look angle of each pixel
    pix=list(range(0,er_image.shape[0]))

    d_pix=sat_terminator_distance.value/1000.*np.tan(2.5*u.arcsec).value

    alt_center=np.array([terminator_coords.height.value/1000.+(i-(er_image.shape[0]/2.-.5))*d_pix for i in pix])
    horizontal_dist=np.array([np.abs(i-(er_image.shape[0]/2.-.5))*d_pix for i in pix])
    xv,yv=np.meshgrid(horizontal_dist,alt_center)
    alt_map=np.sqrt(xv**2+yv**2)
    alt_center=alt_center
    return alt_map,alt_center

def make_avg_er(er_image,alt_map,alt_center):
    #Average the ER image over the alt map along the center line of image
    avg_er=np.empty(er_image.shape[0]-1)
    median_er=np.empty(er_image.shape[0]-1)
    freq_er=np.empty(er_image.shape[0]-1)
    alt=np.empty(er_image.shape[0]-1)
    for i in range(1,er_image.shape[0]):
        tmp_er=er_image[(alt_map >= alt_center[i-1]) & (alt_map < alt_center[i])]
        alt[i-1]=np.mean(alt_map[(alt_map >= alt_center[i-1]) & (alt_map < alt_center[i])])
        avg_er[i-1]=np.nanmean(tmp_er)
        median_er[i-1]=np.nanmedian(tmp_er)
        tmp_hist=np.histogram(tmp_er[~np.isnan(tmp_er) & ~np.isinf(tmp_er)],50)
        freq_er[i-1]=tmp_hist[1][np.argmax(tmp_hist[0])]
    return avg_er,median_er,freq_er,alt

def makeER(occultation_fits,topside_fits,typeOcc,ch):
    #Process the occulation image to get data, coords and header
    occ_data, occ_header, coords_occ=get_coords_from_fits(occultation_fits)
    image_center_occ = tuple(np.array([occ_header['CRPIX2'],occ_header['CRPIX1']]))
    
    #Rotate so solar north is up
    rot_mat_occ = cv2.getRotationMatrix2D(image_center_occ, -occ_header['CROTA'], 1.0)
    result_occ=cv2.warpAffine(occ_data, rot_mat_occ, occ_data.shape[1::-1])#, flags=cv2.INTER_LINEAR)

    #Get terminator coords and distance from sat
    observer_atmos,sat_terminator_distance,terminator_coords_occ=get_terminator_coords(coords_occ,rot_mat_occ)
    observer_atmos_geo=SkyCoord(observer_atmos).transform_to(astropy.coordinates.ITRS).earth_location.to_geodetic('WGS84')
    
    #Process topside image for data, coords and header
    top_data, top_header, coords_topside=get_coords_from_fits(topside_fits)
    image_center_top = tuple(np.array([top_header['CRPIX2'],top_header['CRPIX1']]))
    
    #Rotate so solar north is up
    rot_mat_top = cv2.getRotationMatrix2D(image_center_top, -top_header['CROTA'], 1.0)
    result_top=cv2.warpAffine(top_data, rot_mat_top, top_data.shape[1::-1])#, flags=cv2.INTER_LINEAR)
    
    #Find solar pixels for oc and top image using center and solar diameter in 
    #fits header
    pixels_out_top,pixels_in_top=pixels_of_sun(result_top,top_header['DIAM_SUN']/2.,top_header['CRPIX2'],top_header['CRPIX1'],1.1)
    pixels_out_occ,pixels_in_occ=pixels_of_sun(result_occ,occ_header['DIAM_SUN']/2.,occ_header['CRPIX2'],occ_header['CRPIX1'],1.1)

    #Set the non-solar pixels to zero for cross correlation 
    result_occ[np.isnan(result_occ)]=0
    result_top[np.isnan(result_top)]=0
    result_top_tmp=result_top.copy()
    result_occ_tmp=result_occ.copy()
    
    #Take the phase correlation to line up topside and occ image
    shift, error, diffphase = phase_cross_correlation(result_top_tmp,result_occ_tmp)
    
    result_occ_tmp[pixels_out_occ]=np.nan
    result_top_tmp[pixels_out_top]=np.nan
    
    result_occ=scipy.ndimage.shift(result_occ, shift)
    
    #Take phase correlated top and occ image to make ER image
    er=result_occ/result_top

    #Center all images so sun center is at the image center
    translation_matrix = np.float32([ [1,0,result_occ.shape[1]/2.-.5-top_header['CRPIX2']], [0,1,result_occ.shape[0]/2.-.5-top_header['CRPIX1']]])
    er_centered = cv2.warpAffine(er, translation_matrix, er.shape[1::-1])
    result_occ_centered=cv2.warpAffine(result_occ, translation_matrix, er.shape[1::-1])
    result_top_centered=cv2.warpAffine(result_top, translation_matrix, er.shape[1::-1])
    
    pixels_out,pixels_in=pixels_of_sun(er_centered,top_header['DIAM_SUN']/2.,er_centered.shape[1]/2-.5,er_centered.shape[0]/2-.5,.95)
    er_centered[pixels_out]=np.nan
    
    ret, threshold = cv2.threshold(er_centered,0.3,1,cv2.THRESH_BINARY)

    #Use PCA to get the image horizontal to be the smae direction as Earth's horizon
    er=horizontalize(er_centered,threshold,typeOcc)
    occ=horizontalize(result_occ_centered, threshold, typeOcc)
    top=horizontalize(result_top_centered, threshold, typeOcc)
# =============================================================================
#     er[np.isnan(er) | np.isinf(er)]=0
#     res=np.fft.fft2(er)
#     row,col=res.shape
#     res[int(row*.01):int(row*(1-.01))] = 0
#     res[:, int(col*.01):int(col*(1-.01))] = 0
#     er=np.real(np.fft.ifft2(res))
#     occ[pixels_out]=np.nan
#     top[pixels_out]=np.nan
#     er[pixels_out]=np.nan
# =============================================================================
    
    #Make the altitude map given sat position and distance to terminator plane
    alt_map,alt_center=make_alt_map(er,observer_atmos_geo,sat_terminator_distance)
    
    #Make average ER profile over similar alts
    avg_er,median_er,freq_er,alt=make_avg_er(er,alt_map,alt_center)
    occ_gse=SkyCoord(coords_occ.observer).transform_to(frames.GeocentricSolarEcliptic).cartesian

    #Make averages of processed occ and topside  images
    avg_occ,median_occ,freq_occ,alt_tmp=make_avg_er(occ, alt_map, alt_center)
    avg_top,median_top,freq_top,atl_tmp=make_avg_er(top, alt_map, alt_center)
    
    #Make average, median and mode ER from top and occ averages
    avg_er2=avg_occ/avg_top
    median_er2=median_occ/median_top
    freq_er2=freq_occ/freq_top
    
    #Truncate to be only sun values (non-nan)
    alt2=alt[(~np.isinf(avg_er2) & ~np.isnan(avg_er2))]
    median_er2=median_er2[(~np.isinf(avg_er2) & ~np.isnan(avg_er2))]
    freq_er2=freq_er2[(~np.isinf(avg_er2) & ~np.isnan(avg_er2))]
    alt=alt[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    median_er=median_er[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    freq_er=freq_er[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    avg_er=avg_er[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    avg_er2=avg_er2[(~np.isinf(avg_er2) & ~np.isnan(avg_er2))]
    
    sat_geo=SkyCoord(coords_occ.observer).transform_to(astropy.coordinates.ITRS).earth_location.to_geodetic('WGS84')
    return er, top, occ, alt_map, avg_er, median_er, freq_er, avg_er2, median_er2, freq_er2, alt, alt2, occ_header['DATE-OBS'], occ_header['OBSGEO-X'], occ_header['OBSGEO-Y'], occ_header['OBSGEO-Z'], occ_gse,sat_geo,observer_atmos,observer_atmos.transform_to(astropy.coordinates.ITRS(obstime=coords_occ.observer.obstime)),observer_atmos_geo

def make_col_density(date,typeocc,channels,avg_er_list,alt_list,cs_wv,cs_o,cs_n2,cs_o2):
    alts=[]
    ers=[]
    os=[]
    n2s=[]
    o2s=[]
    ch=[]
    
    #Determine what channels are in occ window and add their channel's effective
    #cross section to the x-matrix
    if 'Fe171' in channels:
        alt_fe171=np.concatenate(alt_list[np.where(channels=='Fe171')])
        avg_er_fe171=np.concatenate(avg_er_list[np.where(channels=='Fe171')])
        avg_er_fe171=avg_er_fe171[np.argsort(alt_fe171)]
        alt_fe171=alt_fe171[np.argsort(alt_fe171)]
        alt_fe171=alt_fe171[~np.isnan(avg_er_fe171) & ~np.isinf(avg_er_fe171) & (avg_er_fe171 > 0)]
        avg_er_fe171=avg_er_fe171[~np.isnan(avg_er_fe171) & ~np.isinf(avg_er_fe171) & (avg_er_fe171 > 0)]    
        alts.append(alt_fe171)
        ers.append(avg_er_fe171)
        os.append(cs_o[cs_wv=='Fe171'][0])
        n2s.append(cs_n2[cs_wv=='Fe171'][0])
        o2s.append(cs_o2[cs_wv=='Fe171'][0])
        ch.append('Fe171')
    if 'He304' in channels:
        alt_he304=np.concatenate(alt_list[np.where(channels=='He304')])
        avg_er_he304=np.concatenate(avg_er_list[np.where(channels=='He304')])
        avg_er_he304=avg_er_he304[np.argsort(alt_he304)]
        alt_he304=alt_he304[np.argsort(alt_he304)]
        alt_he304=alt_he304[~np.isnan(avg_er_he304) & ~np.isinf(avg_er_he304) & (avg_er_he304 > 0)]
        avg_er_he304=avg_er_he304[~np.isnan(avg_er_he304) & ~np.isinf(avg_er_he304) & (avg_er_he304 > 0)]  
        alts.append(alt_he304)
        ers.append(avg_er_he304)
        os.append(cs_o[cs_wv=='He304'][0])
        n2s.append(cs_n2[cs_wv=='He304'][0])
        o2s.append(cs_o2[cs_wv=='He304'][0])
        ch.append('He304')
    if 'Fe284' in channels:
        alt_fe284=np.concatenate(alt_list[np.where(channels=='Fe284')])
        avg_er_fe284=np.concatenate(avg_er_list[np.where(channels=='Fe284')])
        avg_er_fe284=avg_er_fe284[np.argsort(alt_fe284)]
        alt_fe284=alt_fe284[np.argsort(alt_fe284)]
        alt_fe284=alt_fe284[~np.isnan(avg_er_fe284) & ~np.isinf(avg_er_fe284) & (avg_er_fe284 > 0)]
        avg_er_fe284=avg_er_fe284[~np.isnan(avg_er_fe284) & ~np.isinf(avg_er_fe284) & (avg_er_fe284 > 0)]  
        alts.append(alt_fe284)
        ers.append(avg_er_fe284)
        os.append(cs_o[cs_wv=='Fe284'][0])
        n2s.append(cs_n2[cs_wv=='Fe284'][0])
        o2s.append(cs_o2[cs_wv=='Fe284'][0])
        ch.append('Fe284')
    if 'Fe195' in channels:
        alt_fe195=np.concatenate(alt_list[np.where(channels=='Fe195')])
        avg_er_fe195=np.concatenate(avg_er_list[np.where(channels=='Fe195')])
        avg_er_fe195=avg_er_fe195[np.argsort(alt_fe195)]
        alt_fe195=alt_fe195[np.argsort(alt_fe195)]
        alt_fe195=alt_fe195[~np.isnan(avg_er_fe195) & ~np.isinf(avg_er_fe195) & (avg_er_fe195 > 0)]
        avg_er_fe195=avg_er_fe195[~np.isnan(avg_er_fe195) & ~np.isinf(avg_er_fe195) & (avg_er_fe195 > 0)]  
        alts.append(alt_fe195)
        ers.append(avg_er_fe195)
        os.append(cs_o[cs_wv=='Fe195'][0])
        n2s.append(cs_n2[cs_wv=='Fe195'][0])
        o2s.append(cs_o2[cs_wv=='Fe195'][0])
        ch.append('Fe195')
        
    #Find the alt range that we have data on all available channels
    min_all=np.max([np.min(alt) for alt in alts])
    max_all=np.min([np.max(alt) for alt in alts])
    interp_ers=[]
    
    #Interpolate all channels to the same grid
    for i in range(len(ers)):
        if i==0:
            inds_all=np.where((alts[i][:]>min_all) & (alts[i][:]<max_all))
            ers[i]=ers[i][inds_all[0][0]:inds_all[0][-1]]
            alts[i]=alts[i][inds_all[0][0]:inds_all[0][-1]]
            print(min(alts[i]),max(alts[i]))
            print(len(ers[i]))
            f_interp=interpolate.interp1d(alts[i],ers[i],kind='linear')
            smooth_er = savgol_filter(f_interp(alts[0]),53,3)
            interp_ers.append(smooth_er)
            y=np.matrix( ((np.log(interp_ers[i]))) )
            if len(ch)>20:
                H=np.matrix( [[-1*os[i],-1*n2s[i],-1*o2s[i]]] )
            else:
                H=np.matrix( [[-1*os[i],-1*n2s[i]]] )
        else:
            f_interp=interpolate.interp1d(alts[i],ers[i],kind='linear')
            smooth_er = savgol_filter(f_interp(alts[0]),53,3)
            interp_ers.append(smooth_er)
            print(len(interp_ers[i]))
            y=np.append([np.log(interp_ers[i])],y,axis=0)
            if len(ch)>20:
                H=np.append( [[-1*os[i],-1*n2s[i],-1*o2s[i]]],H,axis=0 )
            else:
                H=np.append( [[-1*os[i],-1*n2s[i]]],H,axis=0 )

    #Solve the least squares problem for all available channels to get N_O 
    #and N_N2            
    x=np.linalg.inv(H.transpose()*H)*H.transpose()*y
    
    return alts[0],x
