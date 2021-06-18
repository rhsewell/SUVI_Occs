import numpy as np
import re
import scipy.io

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from astropy import units as u
import sunpy.map
from suvi_fix_L1b_header import fix_suvi_l1b_header
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from mathutils.geometry import intersect_point_line
from scipy.interpolate import LinearNDInterpolator,griddata
import scipy.io
import scipy.optimize
import cv2
import csv
from astropy import coordinates as coords
#from suvi.py import get_coords_from_fits

def ClosestPointOnLine(a, b, p):
    ap = p-a
    ab = b-a
    result = a + np.dot(ap,ab)/np.dot(ab,ab) * ab
    return result

def get_coords_from_fits(fits_file):
    hdu=fits.open(fits_file)
    data=hdu[0].data
    header=fix_suvi_l1b_header(hdu[0].header)
    map_image=sunpy.map.sources.SUVIMap(data, header)
    #fig,ax = plt.subplots(1)
    #ax.set_aspect('equal')
    #ax.imshow(hdu[0].data,cmap='gray')
    #circ=Circle((hdu[0].header['CRPIX1'],hdu[0].header['CRPIX2']),15)
    #print(hdu[0].header['CRPIX1'],hdu[0].header['CRPIX2'])
    #Circle.set_color(circ,'red')
    #ax.add_patch(circ)
    #plt.show()
    return data, header, sunpy.map.all_coordinates_from_map(map_image)


def get_terminator_coords(sat_map):
    sat_loc_gse=SkyCoord(sat_map.observer).transform_to(frames.GeocentricSolarEcliptic)
    sun_loc_gse=SkyCoord(0*u.m, 0*u.m, 0*u.m, obstime=sat_map.obstime,frame=frames.Heliocentric,\
                      observer='sun').transform_to(frames.GeocentricSolarEcliptic)
    sat_sun_line=((sat_loc_gse.cartesian.x.value,sat_loc_gse.cartesian.y.value,sat_loc_gse.cartesian.z.value), \
                (sun_loc_gse.cartesian.x.value,sun_loc_gse.cartesian.y.value,sun_loc_gse.cartesian.z.value))
    observer_atmos = intersect_point_line((0.0,0.0,0.0), sat_sun_line[0], sat_sun_line[1])
    observer_atmos = SkyCoord(observer_atmos[0].x*u.m,observer_atmos[0].y*u.m,observer_atmos[0].z*u.m,\
                          obstime=sat_loc_gse.obstime,frame=frames.GeocentricSolarEcliptic,representation_type='cartesian')
    observer_atmos.observer=observer_atmos
    sat_terminator_distance=np.sqrt((sat_loc_gse.cartesian.x-observer_atmos.cartesian.x)**2+\
                          (sat_loc_gse.cartesian.y-observer_atmos.cartesian.y)**2+\
                              (sat_loc_gse.cartesian.z-observer_atmos.cartesian.z)**2)
    print(np.sqrt(observer_atmos.cartesian.x**2+observer_atmos.cartesian.y**2+observer_atmos.cartesian.z**2))
    hcc_atmos=observer_atmos.transform_to(frames.Heliocentric)
    hcc_sat=SkyCoord(sat_map.observer,observer=sat_map.observer).transform_to(frames.Heliocentric)
    zpix=(hcc_atmos.cartesian.x.value*Tx*(hcc_sat.cartesian.x.value**2-hcc_sat.cartesian.x.value*hcc_atmos.cartesian.x.value)+hcc_atmos.cartesian.y.value*Ty*(hcc_sat.cartesian.y.value**2-hcc_sat.cartesian.y.value*hcc_atmos.cartesian.y.value))/(hcc_atmos.cartesian.z.value*(hcc_sat.cartesian.z.value**2-hcc_sat.cartesian.z.value*hcc_atmos.cartesian.z.value))+hcc_atmos.cartesian.z.value
    return SkyCoord(np.tan(sat_map.Tx)*sat_terminator_distance,np.tan(sat_map.Ty)*sat_terminator_distance,\
                   observer_atmos.transform_to(frames.Heliocentric).z,obstime=sat_loc_gse.obstime,\
                       frame=frames.Heliocentric,representation_type='cartesian',observer=observer_atmos).transform_to(frames.GeocentricSolarEcliptic)


def get_terminator_coords_rot(sat_map,rot_mat):
    sat_loc_gse=SkyCoord(sat_map.observer).transform_to(frames.GeocentricSolarEcliptic)
    sun_loc_gse=SkyCoord(0*u.m, 0*u.m, 0*u.m, obstime=sat_map.obstime,frame=frames.Heliocentric,\
                      observer='sun').transform_to(frames.GeocentricSolarEcliptic)
    sat_sun_line=((sat_loc_gse.cartesian.x.value,sat_loc_gse.cartesian.y.value,sat_loc_gse.cartesian.z.value), \
                (sun_loc_gse.cartesian.x.value,sun_loc_gse.cartesian.y.value,sun_loc_gse.cartesian.z.value))
    observer_atmos = intersect_point_line((0.0,0.0,0.0), sat_sun_line[0], sat_sun_line[1])
    observer_atmos = SkyCoord(observer_atmos[0].x*u.m,observer_atmos[0].y*u.m,observer_atmos[0].z*u.m,\
                          obstime=sat_loc_gse.obstime,frame=frames.GeocentricSolarEcliptic,representation_type='cartesian')
    observer_atmos.observer=sat_map.observer
    sat_terminator_distance=np.sqrt((sat_loc_gse.cartesian.x-observer_atmos.cartesian.x)**2+\
                          (sat_loc_gse.cartesian.y-observer_atmos.cartesian.y)**2+\
                              (sat_loc_gse.cartesian.z-observer_atmos.cartesian.z)**2)
    print(np.sqrt(observer_atmos.cartesian.x**2+observer_atmos.cartesian.y**2+observer_atmos.cartesian.z**2))
    hcc_atmos=observer_atmos.transform_to(frames.Heliocentric)
    hcc_sat=SkyCoord(sat_map.observer,observer=sat_map.observer).transform_to(frames.Heliocentric)
    Tx=cv2.warpAffine(np.tan(sat_map.Tx)*sat_terminator_distance, rot_mat, sat_map.Tx.shape[1::-1])
    Ty=cv2.warpAffine(np.tan(sat_map.Ty)*sat_terminator_distance, rot_mat, sat_map.Tx.shape[1::-1])
    zpix=(hcc_atmos.cartesian.x.value*Tx*(hcc_sat.cartesian.x.value**2-hcc_sat.cartesian.x.value*hcc_atmos.cartesian.x.value)+hcc_atmos.cartesian.y.value*Ty*(hcc_sat.cartesian.y.value**2-hcc_sat.cartesian.y.value*hcc_atmos.cartesian.y.value))/(hcc_atmos.cartesian.z.value*(hcc_sat.cartesian.z.value**2-hcc_sat.cartesian.z.value*hcc_atmos.cartesian.z.value))+hcc_atmos.cartesian.z.value
    return SkyCoord(Tx*u.m,Ty*u.m,\
                   zpix*u.m,obstime=sat_loc_gse.obstime,\
                       frame=frames.Heliocentric,representation_type='cartesian',observer=observer_atmos.observer).transform_to(frames.GeocentricSolarEcliptic)
        
def get_obs_sun_coords(sat_map):
    return SkyCoord(np.tan(sat_map.Tx)*sat_map.observer.radius,np.tan(sat_map.Ty)*sat_map.observer.radius,0.0*u.m,frame=frames.Heliocentric) 

def pixels_of_sun(er,radius, x0, y0,radius_multi):
    x_ = np.arange(0, 1280, dtype=int)
    y_ = np.arange(0, 1280, dtype=int)
    pixels_out=np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 > (radius_multi*radius)**2)
    pixels_in=np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= (radius_multi*radius)**2)
    #er[np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 >= (radius_multi*radius)**2)]=np.nan
    return pixels_out,pixels_in

alt_msis=[]
o_msis=[]
n2_msis=[]
o2_msis=[]
with open(r'C:\Users\Robert\Documents\SUVI_Occs\msise00_2019_09_15_04_22.txt') as f:
    for line in f:
        data_line=re.match(r"\ *(\d+\.\d)\ +(\d\.\d+E[\+|-]\d+)\ +(\d\.\d+E[\+|-]\d+)\ +(\d\.\d+E[\+|-]\d+).*" ,line)
        if data_line:
            alt_msis.append(float(data_line.group(1)))
            o_msis.append(float(data_line.group(2)))
            n2_msis.append(float(data_line.group(3)))
            o2_msis.append(float(data_line.group(4)))
            
alt_msis=np.array(alt_msis)
o_msis=np.array(o_msis)
n2_msis=np.array(n2_msis)
o2_msis=np.array(o2_msis)

dr=alt_msis[1]-alt_msis[0]
N_o=[]
N_n2=[]
N_o2=[]
#A=[]
for i,h_t in enumerate(alt_msis,start=0):
    tmp_o=0
    tmp_n2=0
    tmp_o2=0
    #a=0
    for j,r in enumerate(alt_msis[i:-1],start=i+1):
        if j<len(alt_msis):
            tmp_o=tmp_o+o_msis[j-1]*(np.sqrt((alt_msis[j]*1.05E5)**2-(h_t*1.05E5)**2)-np.sqrt((alt_msis[j-1]*1.05E5)**2-(h_t*1.05E5)**2))
            tmp_n2=tmp_n2+n2_msis[j-1]*(np.sqrt((alt_msis[j]*1.05E5)**2-(h_t*1.05E5)**2)-np.sqrt((alt_msis[j-1]*1.05E5)**2-(h_t*1.05E5)**2))
            tmp_o2=tmp_o2+o2_msis[j-1]*(np.sqrt((alt_msis[j]*1.05E5)**2-(h_t*1.05E5)**2)-np.sqrt((alt_msis[j-1]*1.05E5)**2-(h_t*1.05E5)**2))
            #tmp_o=tmp_o+(alt_msis[j]*1.0E5-alt_msis[j-1]*1.0E5)*(integrand(alt_msis[j],o_msis[j],h_t)+integrand(alt_msis[j-1],o_msis[j-1],h_t))/2
            #tmp_n2=tmp_n2+(alt_msis[j]*1.0E5-alt_msis[j-1]*1.0E5)*(integrand(alt_msis[j],n2_msis[j],h_t)+integrand(alt_msis[j-1],n2_msis[j-1],h_t))/2
            #tmp_o2=tmp_o2+(alt_msis[j]*1.0E5-alt_msis[j-1]*1.0E5)*(integrand(alt_msis[j],o2_msis[j],h_t)+integrand(alt_msis[j-1],o2_msis[j-1],h_t))/2
    #A.append(a)
    N_o.append(tmp_o)
    N_n2.append(tmp_n2)
    N_o2.append(tmp_o2)
    
#A=np.array(A)
N_o=np.array(N_o)
N_n2=np.array(N_n2)
N_o2=np.array(N_o2)
    
cross_sec=scipy.io.readsav(r'C:\Users\Robert\Documents\SUVI_Occs\photon_cross_sections.sav')

o2=cross_sec['photo'].O2[0].XSECTION[0]
n2=cross_sec['photo'].N2[0].XSECTION[0]
o=cross_sec['photo'].O3P[0].XSECTION[0]
wave=cross_sec['photo'].ANGSTROMS[0]

sigmao_171=o[(np.abs(wave - 171).argmin()),0]
sigman2_171=n2[(np.abs(wave - 171).argmin()),0]
sigmao2_171=n2[(np.abs(wave - 171).argmin()),0]


sigmao_195=o[(np.abs(wave - 195).argmin()),0]
sigman2_195=n2[(np.abs(wave - 195).argmin()),0]
sigmao2_195=n2[(np.abs(wave - 195).argmin()),0]


sigmao_303=o[(np.abs(wave - 303).argmin()),0]
sigman2_303=n2[(np.abs(wave - 303).argmin()),0]
sigmao2_303=n2[(np.abs(wave - 303).argmin()),0]

sigmao_284=o[(np.abs(wave - 284).argmin()),0]
sigman2_284=n2[(np.abs(wave - 284).argmin()),0]
sigmao2_284=n2[(np.abs(wave - 284).argmin()),0]

ER_171=np.exp(-(N_o*sigmao_171+N_n2*sigman2_171+N_o2*sigmao2_171))
ER_195=np.exp(-(N_o*sigmao_195+N_n2*sigman2_195+N_o2*sigmao2_195))
ER_303=np.exp(-(N_o*sigmao_303+N_n2*sigman2_303+N_o2*sigmao2_303))
ER_284=np.exp(-(N_o*sigmao_284+N_n2*sigman2_284+N_o2*sigmao2_284))

pd.DataFrame(ER_171).to_csv(r'C:\Users\Robert\Documents\SUVI_Occs\MSIS_ER_171.csv')
    
top_data_171, top_header_171, coords_topside_171=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe171_G16_s20192580414067_e20192580414077_c20192580414287.fits")
occ_data_171, occ_header_171, coords_occ_171=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe171_G16_s20192580422067_e20192580422077_c20192580422293.fits")
image_center_occ_171 = tuple(np.array([top_header_171['CRPIX1'],top_header_171['CRPIX2']]))
rot_mat_occ_171 = cv2.getRotationMatrix2D(image_center_occ_171, -top_header_171['CROTA'], 1.0)
terminator_coords_occ_171=get_terminator_coords_rot(coords_occ_171,rot_mat_occ_171)
terminator_coords_occ_171=terminator_coords_occ_171.transform_to(coords.ITRS).spherical
new_data_171=top_data_171.copy()
new_alt=(terminator_coords_occ_171.distance.value-6.371E6)/1000.#((terminator_coords_top_171.distance.value-6.371E6)/1000.-np.min((terminator_coords_top_171.distance.value-6.371E6)/1000.))
new_data_171[new_alt<alt_msis[0]]=0.0
for i,alt in enumerate(alt_msis,start=0):
    if i<len(alt_msis)-1:
        inds=(new_alt >= alt_msis[i]) & (new_alt < alt_msis[i+1])
        if len(new_data_171[inds]) > 0:
            #print(new_data[inds])
            new_data_171[inds]=[x * ER_171[i] for x in top_data_171[inds]]
            #new_data[inds]=ER_171[i]
            #print(new_data[inds])
           # print(len(new_data[inds]))
           
top_data_303, top_header_303, coords_topside_303=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-He303_G16_s20192580418567_e20192580418577_c20192580419195.fits")
occ_data_303, occ_header_303, coords_occ_303=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-He303_G16_s20192580422567_e20192580422577_c20192580423198.fits")
image_center_occ_303 = tuple(np.array([top_header_303['CRPIX1'],top_header_303['CRPIX2']]))
rot_mat_occ_303 = cv2.getRotationMatrix2D(image_center_occ_303, -top_header_303['CROTA'], 1.0)
terminator_coords_occ_303=get_terminator_coords_rot(coords_occ_303,rot_mat_occ_303)
terminator_coords_occ_303=terminator_coords_occ_303.transform_to(coords.ITRS).spherical
new_data_303=top_data_303.copy()
new_alt=(terminator_coords_occ_303.distance.value-6.371E6)/1000.#((terminator_coords_top_303.distance.value-6.371E6)/1000.-np.min((terminator_coords_top_303.distance.value-6.371E6)/1000.))
new_data_303[new_alt<alt_msis[0]]=0.0
for i,alt in enumerate(alt_msis,start=0):
    if i<len(alt_msis)-1:
        inds=(new_alt >= alt_msis[i]) & (new_alt < alt_msis[i+1])
        if len(new_data_303[inds]) > 0:
            new_data_303[inds]=[x * ER_303[i] for x in top_data_303[inds]]

            
top_data_195, top_header_195, coords_topside_195=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe195_G16_s20192580419167_e20192580419177_c20192580419386.fits")
occ_data_195, occ_header_195, coords_occ_195=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe195_G16_s20192580422267_e20192580422277_c20192580422496.fits")
image_center_occ_195 = tuple(np.array([top_header_195['CRPIX1'],top_header_195['CRPIX2']]))
rot_mat_occ_195 = cv2.getRotationMatrix2D(image_center_occ_195, -top_header_195['CROTA'], 1.0)
terminator_coords_occ_195=get_terminator_coords_rot(coords_occ_195,rot_mat_occ_195)
terminator_coords_occ_195=terminator_coords_occ_195.transform_to(coords.ITRS).spherical
new_data_195=top_data_195.copy()
new_alt=(terminator_coords_occ_195.distance.value-6.371E6)/1000.#((terminator_coords_top_195.distance.value-6.371E6)/1000.-np.min((terminator_coords_top_195.distance.value-6.371E6)/1000.))
new_data_195[new_alt<alt_msis[0]]=0.0
for i,alt in enumerate(alt_msis,start=0):
    if i<len(alt_msis)-1:
        inds=(new_alt >= alt_msis[i]) & (new_alt < alt_msis[i+1])
        if len(new_data_195[inds]) > 0:
            new_data_195[inds]=[x * ER_195[i] for x in top_data_195[inds]]
            
top_data_284, top_header_284, coords_topside_284=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe284_G16_s20192580418367_e20192580418377_c20192580418593.fits")
occ_data_284, occ_header_284, coords_occ_284=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe284_G16_s20192580422367_e20192580422377_c20192580422593.fits")
image_center_occ_284 = tuple(np.array([top_header_284['CRPIX1'],top_header_284['CRPIX2']]))
rot_mat_occ_284 = cv2.getRotationMatrix2D(image_center_occ_284, -top_header_284['CROTA'], 1.0)
terminator_coords_occ_284=get_terminator_coords_rot(coords_occ_284,rot_mat_occ_284)
terminator_coords_occ_284=terminator_coords_occ_284.transform_to(coords.ITRS).spherical
new_data_284=top_data_284.copy()
new_alt=(terminator_coords_occ_284.distance.value-6.371E6)/1000.#((terminator_coords_top_284.distance.value-6.371E6)/1000.-np.min((terminator_coords_top_284.distance.value-6.371E6)/1000.))
new_data_284[new_alt<alt_msis[0]]=0.0
for i,alt in enumerate(alt_msis,start=0):
    if i<len(alt_msis)-1:
        inds=(new_alt >= alt_msis[i]) & (new_alt < alt_msis[i+1])
        if len(new_data_284[inds]) > 0:
            new_data_284[inds]=[x * ER_284[i] for x in top_data_284[inds]]
            

image_center_occ_171 = tuple(np.array([top_header_171['CRPIX1'],top_header_171['CRPIX2']]))
rot_mat_occ_171 = cv2.getRotationMatrix2D(image_center_occ_171, -top_header_171['CROTA'], 1.0)
terminator_coords_occ_171=get_terminator_coords_rot(coords_occ_171,rot_mat_occ_171)
terminator_coords_occ_171=terminator_coords_occ_171.transform_to(coords.ITRS).spherical
#sun_coords_occ_171=get_obs_sun_coords(coords_occ_171)
result_occ_171=cv2.warpAffine(new_data_171, rot_mat_occ_171, occ_data_171.shape[1::-1])#, flags=cv2.INTER_LINEAR)

image_center_top_171 = tuple(np.array([top_header_171['CRPIX1'],top_header_171['CRPIX2']]))
rot_mat_top_171 = cv2.getRotationMatrix2D(image_center_top_171, -top_header_171['CROTA'], 1.0)
result_top_171=cv2.warpAffine(top_data_171, rot_mat_top_171, top_data_171.shape[1::-1])#, flags=cv2.INTER_LINEAR)

num_rows, num_cols = result_occ_171.shape[:2]
# Creating a translation matrix
translation_matrix = np.float32([ [1,0,top_header_171['CRPIX1']-top_header_171['CRPIX1']], [0,1,top_header_171['CRPIX2']-top_header_171['CRPIX2']] ])
# Image translation
result_top_171 = cv2.warpAffine(result_top_171, translation_matrix, (num_cols,num_rows))
er_171=result_occ_171/result_top_171
pixels_out_171,pixels_in_171=pixels_of_sun(er_171,top_header_171['DIAM_SUN']/2.,top_header_171['CRPIX1'],top_header_171['CRPIX2'],0.95)
er_171[pixels_out_171]=np.nan      
        

image_center_occ_303 = tuple(np.array([top_header_303['CRPIX1'],top_header_303['CRPIX2']]))
rot_mat_occ_303 = cv2.getRotationMatrix2D(image_center_occ_303, -top_header_303['CROTA'], 1.0)
terminator_coords_occ_303=get_terminator_coords_rot(coords_occ_303,rot_mat_occ_303)
terminator_coords_occ_303=terminator_coords_occ_303.transform_to(coords.ITRS).spherical
#sun_coords_occ_303=get_obs_sun_coords(coords_occ_303)
result_occ_303=cv2.warpAffine(new_data_303, rot_mat_occ_303, occ_data_303.shape[1::-1], flags=cv2.INTER_LINEAR)

sun_coords_topside_303=get_obs_sun_coords(coords_topside_303)
image_center_top_303 = tuple(np.array([top_header_303['CRPIX1'],top_header_303['CRPIX2']]))
rot_mat_top_303 = cv2.getRotationMatrix2D(image_center_top_303, -top_header_303['CROTA'], 1.0)
result_top_303=cv2.warpAffine(top_data_303, rot_mat_top_303, top_data_303.shape[1::-1], flags=cv2.INTER_LINEAR)

num_rows, num_cols = result_occ_303.shape[:2]
# Creating a translation matrix
translation_matrix = np.float32([ [1,0,top_header_303['CRPIX1']-top_header_303['CRPIX1']], [0,1,top_header_303['CRPIX2']-top_header_303['CRPIX2']] ])
# Image translation
result_top_303 = cv2.warpAffine(result_top_303, translation_matrix, (num_cols,num_rows))
#interpolator_303=LinearNDInterpolator((sun_coords_topside_303.x.value.flatten(),sun_coords_topside_303.y.value.flatten()),top_data_303.flatten())
#coords_topside_303=interpolator_303((sun_coords_occ_303.x.value.flatten(),sun_coords_occ_303.y.value.flatten()))
#er_303=occ_data_303/coords_topside_303.reshape((1280,1280))
er_303=result_occ_303/result_top_303
pixels_out_303,pixels_in_303=pixels_of_sun(er_303,top_header_303['DIAM_SUN']/2.,top_header_303['CRPIX1'],top_header_303['CRPIX2'],0.95)
er_303[pixels_out_303]=np.nan



all_rad_grid=np.arange(np.min([np.min(terminator_coords_occ_171.distance[~np.isnan(er_171)].value),np.min(terminator_coords_occ_303.distance[~np.isnan(er_303)].value)])+1000./2,np.max([np.max(terminator_coords_occ_171.distance[~np.isnan(er_171)].value),np.max(terminator_coords_occ_303.distance[~np.isnan(er_303)].value)])+999.99,1000)
all_theta_grid=np.arange(np.min([np.min(terminator_coords_occ_171.lat[~np.isnan(er_171)].value),np.min(terminator_coords_occ_303.lat[~np.isnan(er_303)].value)])+1./60./2.,np.max([np.max(terminator_coords_occ_171.lat[~np.isnan(er_171)].value),np.max(terminator_coords_occ_303.lat[~np.isnan(er_303)].value)])+1./60.*.99999,1./60.)
#171 regrid
#theta_grid_171=np.arange(np.min(terminator_coords_occ_171.lat[~np.isnan(er_171)].value)+1./60./2.,np.max(terminator_coords_occ_171.lat[~np.isnan(er_171)].value),1./60.)
#rad_grid_171=np.arange(np.min(terminator_coords_occ_171.distance[~np.isnan(er_171)].value)+1000./2,np.max(terminator_coords_occ_171[~np.isnan(er_171)].distance.value),1000)
rad_grid_171=all_rad_grid[(np.abs(all_rad_grid-np.min(terminator_coords_occ_171.distance[~np.isnan(er_171)].value))).argmin():(np.abs(all_rad_grid-np.max(terminator_coords_occ_171.distance[~np.isnan(er_171)].value))).argmin()+1]
theta_grid_171=all_theta_grid[(np.abs(all_theta_grid-np.min(terminator_coords_occ_171.lat[~np.isnan(er_171)].value))).argmin():(np.abs(all_theta_grid-np.max(terminator_coords_occ_171.lat[~np.isnan(er_171)].value))).argmin()+1]
#rad_171,theta_171=np.meshgrid(rad_grid_171,theta_grid_171)
#er_coarse_171=griddata((terminator_coords_occ_171.distance[~np.isnan(er_171)].value.ravel(),terminator_coords_occ_171.lat[~np.isnan(er_171)].value.ravel()), er_171[~np.isnan(er_171)].ravel(), (rad_171, theta_171), method='linear')
rad=cv2.warpAffine(terminator_coords_occ_171.distance, rot_mat_occ_171, occ_data_171.shape[1::-1])
theta=cv2.warpAffine(terminator_coords_occ_171.lat, rot_mat_occ_171, occ_data_171.shape[1::-1])
#er_coarse_171=griddata((rad[~np.isnan(er_171)].ravel(),theta[~np.isnan(er_171)].ravel()), er_171[~np.isnan(er_171)].ravel(), (rad_171, theta_171), method='linear')

avg_er_171=[]
for i,alt in enumerate(rad_grid_171,start=0):
    avg_er_171.append(np.nanmean(er_171[((rad < rad_grid_171[i]+500.) & (rad >= rad_grid_171[i]-500.))]))

#303 regrid
#theta_grid_303=np.arange(np.min(terminator_coords_occ_303.lat[~np.isnan(er_303)].value)+1./60./2.,np.max(terminator_coords_occ_303.lat[~np.isnan(er_303)].value),1./60.)
#rad_grid_303=np.arange(np.min(terminator_coords_occ_303.distance[~np.isnan(er_303)].value)+1000./2,np.max(terminator_coords_occ_303[~np.isnan(er_303)].distance.value),1000)
rad_grid_303=all_rad_grid[(np.abs(all_rad_grid-np.min(terminator_coords_occ_303.distance[~np.isnan(er_303)].value))).argmin():(np.abs(all_rad_grid-np.max(terminator_coords_occ_303.distance[~np.isnan(er_303)].value))).argmin()+1]
theta_grid_303=all_theta_grid[(np.abs(all_theta_grid-np.min(terminator_coords_occ_303.lat[~np.isnan(er_303)].value))).argmin():(np.abs(all_theta_grid-np.max(terminator_coords_occ_303.lat[~np.isnan(er_303)].value))).argmin()+1]
#rad_303,theta_303=np.meshgrid(rad_grid_303,theta_grid_303)
#er_coarse_303=griddata((terminator_coords_occ_303.distance[~np.isnan(er_303)].value.ravel(),terminator_coords_occ_303.lat[~np.isnan(er_303)].value.ravel()), er_303[~np.isnan(er_303)].ravel(), (rad_303, theta_303), method='linear')
#er_coarse_radial_average_303=np.nanmean(er_coarse_303,0)

rad=cv2.warpAffine(terminator_coords_occ_303.distance, rot_mat_occ_303, occ_data_303.shape[1::-1])
theta=cv2.warpAffine(terminator_coords_occ_303.lat, rot_mat_occ_303, occ_data_303.shape[1::-1])
#avg_er_303=[[]]
avg_er_303=[]
for i,alt in enumerate(rad_grid_303,start=0):
    #avg_er_theta_303=[]
    #for j,lat in enumerate(theta_grid_303,start=0):
        #avg_er_theta_303.append(np.nanmean(er_303[(rad<rad_grid_303[i]+500.)&(rad >= rad_grid_303[i]-500.)&(theta < theta_grid_303[j]+1./60./2.)&(theta >= theta_grid_303[j]-1./60./2.)]))
    #avg_er_303.append(avg_er_theta_303)
    #print(avg_er_theta_303)
    avg_er_303.append(np.nanmean(er_303[((rad < rad_grid_303[i]+500.) & (rad >= rad_grid_303[i]-500.))]))

tmp=np.intersect1d(rad_grid_303,rad_grid_171)

avg_er_171=np.array(avg_er_171)
avg_er_303=np.array(avg_er_303)
er_171_intersect=avg_er_171[np.in1d(rad_grid_171,tmp)]
er_303_intersect=avg_er_303[np.in1d(rad_grid_303,tmp)]

cross_sec=scipy.io.readsav(r'C:\Users\Robert\Documents\SUVI_Occs\photon_cross_sections.sav')

n2=cross_sec['photo'].N2[0].XSECTION[0]
o=cross_sec['photo'].O3P[0].XSECTION[0]
wave=cross_sec['photo'].ANGSTROMS[0]

y=np.matrix( ((np.log(er_171_intersect)),(np.log(er_303_intersect))) )
H=-1*np.matrix( ((o[(np.abs(wave - 171).argmin()),0],n2[(np.abs(wave - 171).argmin()),0]),(o[(np.abs(wave - 303).argmin()),0],n2[(np.abs(wave - 303).argmin()),0])) )
x=np.linalg.inv(H.transpose()*H)*H.transpose()*y