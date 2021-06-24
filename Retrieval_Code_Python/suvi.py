# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 12:24:48 2021

@author: Robert
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
from skimage.registration._phase_cross_correlation import _upsampled_dft
from sklearn.decomposition import PCA

def get_coords_from_fits(fits_file):
    hdu=fits.open(fits_file)
    data=hdu[0].data
    header=fix_suvi_l1b_header(hdu[0].header)
    map_image=sunpy.map.sources.SUVIMap(data, header)
    #fig,ax = plt.subplots(1)
    #ax.set_aspect('equal')
    #ax.imshow(hdu[0].data,cmap='gray')
    #circ=Circle((hdu[0].header['CRPIX1'],hdu[0].header['CRPIX2']),15)
    print(hdu[0].header['CRPIX1'],hdu[0].header['CRPIX2'])
    #Circle.set_color(circ,'red')
    #ax.add_patch(circ)
    #plt.show()
    return data, header, sunpy.map.all_coordinates_from_map(map_image)

def get_terminator_coords(sat_map,rot_mat):
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

def get_terminator_distance(sat_map):
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
    return sat_terminator_distance.value/1000.

def coalign_images(image1,image2,xrange_image1,yrange_image1,xrange_image2,yrange_image2):
    shifts = sunpy.image.coalignment.calculate_shift(image2[xrange_image2[0]:xrange_image2[1], yrange_image2[0]:yrange_image2[1]], image1[xrange_image1[0]:xrange_image1[1], yrange_image1[0]:yrange_image1[1]])
    translation_matrix = np.float32([[1, 0, shifts[0].value],
          [0, 1, shifts[1].value]])
    print('***************************************', shifts)
    image2_shifted = cv2.warpAffine(image2, translation_matrix, image2.shape[1::-1])
    return image2_shifted

def horizontalize(img,thresh):
    X = np.array(np.where(thresh > 0)).T
    # Perform a PCA and compute the angle of the first principal axes
    pca = PCA(n_components=2).fit(X)
    angle = np.mod(np.arctan2(*pca.components_[0])/np.pi*180,180)
    print(angle)
    rot_mat = cv2.getRotationMatrix2D((img.shape[0]/2.,img.shape[1]/2.), angle, 1.0)
    # Rotate the image by the computed angle:
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1])
    return rotated_img

#def main():
#171
occ_data_171, occ_header_171, coords_occ_171=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe171_G16_s20192580422067_e20192580422077_c20192580422293.fits")
image_center_occ_171 = tuple(np.array([occ_header_171['CRPIX2'],occ_header_171['CRPIX1']]))
rot_mat_occ_171 = cv2.getRotationMatrix2D(image_center_occ_171, -occ_header_171['CROTA'], 1.0)
terminator_coords_occ_171=get_terminator_coords(coords_occ_171,rot_mat_occ_171)
#sun_coords_occ_171=get_obs_sun_coords(coords_occ_171)
result_occ_171=cv2.warpAffine(occ_data_171, rot_mat_occ_171, occ_data_171.shape[1::-1])#, flags=cv2.INTER_LINEAR)

top_data_171, top_header_171, coords_topside_171=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe171_G16_s20192580414067_e20192580414077_c20192580414287.fits")
#terminator_coords_topside_171=get_terminator_coords(coords_topside_171)er
#sun_coords_topside_171=get_obs_sun_coords(coords_topside_171)
image_center_top_171 = tuple(np.array([top_header_171['CRPIX2'],top_header_171['CRPIX1']]))
rot_mat_top_171 = cv2.getRotationMatrix2D(image_center_top_171, -top_header_171['CROTA'], 1.0)
result_top_171=cv2.warpAffine(top_data_171, rot_mat_top_171, top_data_171.shape[1::-1])#, flags=cv2.INTER_LINEAR)

result_occ_171[np.isnan(result_occ_171)]=0
result_top_171[np.isnan(result_top_171)]=0
shift, error, diffphase = phase_cross_correlation(result_top_171,result_occ_171)
result_occ_171=scipy.ndimage.shift(result_occ_171, shift)


fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(result_occ_171/result_top_171,origin='lower',vmin=0,vmax=1.05)




#num_rows, num_cols = result_occ_171.shape[:2]
# Creating a translation matrix
#translation_matrix = np.float32([ [1,0,occ_header_171['CRPIX1']-top_header_171['CRPIX1']], [0,1,occ_header_171['CRPIX2']-top_header_171['CRPIX2']] ])
# Image translation
#result_top_171 = cv2.warpAffine(result_top_171, translation_matrix, (num_cols,num_rows))
#result_top_171=coalign_images(result_occ_171,result_top_171,[800,880],[400,480],[800,900],[400,500])
#fig,ax = plt.subplots(1)
#ax.set_aspect('equal')
#plt.imshow(result_occ_171/result_top_171, origin='lower', vmin =0, vmax = 2)
#interpolator_171=LinearNDInterpolator((sun_coords_topside_171.x.value.flatten(),sun_coords_topside_171.y.value.flatten()),top_data_171.flatten())
#coords_topside_171=interpolator_171((sun_coords_occ_171.x.value.flatten(),sun_coords_occ_171.y.value.flatten()))
#er_171=occ_data_171/coords_topside_171.reshape((1280,1280))
er_171=result_occ_171/result_top_171
pixels_out_171,pixels_in_171=pixels_of_sun(er_171,top_header_171['DIAM_SUN']/2.,top_header_171['CRPIX2'],top_header_171['CRPIX1'],1.0)
er_171[pixels_out_171]=np.nan
translation_matrix = np.float32([ [1,0,640-top_header_171['CRPIX2']], [0,1,640-top_header_171['CRPIX1']]])
er_171_centered = cv2.warpAffine(er_171, translation_matrix, er_171.shape[1::-1])
ret, threshold = cv2.threshold(er_171_centered,.1,1,cv2.THRESH_BINARY)
er_171=horizontalize(er_171_centered,threshold)
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(er_171,origin='lower',vmin=0,vmax=1.05)


# =============================================================================
##Too dim
# occ_data_93, occ_header_93, coords_occ_93=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe093_G16_s20192580421367_e20192580421377_c20192580421592.fits")
# terminator_coords_occ_93=get_terminator_coords(coords_occ_93)
# sun_coords_occ_93=get_obs_sun_coords(coords_occ_93)
# top_data_93, top_header_93, coords_topside_93=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe093_G16_s20192580417367_e20192580417377_c20192580417587.fits")
# terminator_coords_topside_93=get_terminator_coords(coords_topside_93)
# sun_coords_topside_93=get_obs_sun_coords(coords_topside_93)
# 
# interpolator_93=LinearNDInterpolator((sun_coords_topside_93.x.value.flatten(),sun_coords_topside_93.y.value.flatten()),top_data_93.flatten())
# coords_topside_93=interpolator_93((sun_coords_occ_93.x.value.flatten(),sun_coords_occ_93.y.value.flatten()))
# er_93=occ_data_93/coords_topside_93.reshape((1280,1280))
# pixels_out_93,pixels_in_93=pixels_of_sun(er_93,occ_header_93['DIAM_SUN']/2.,occ_header_93['CRPIX1'],occ_header_93['CRPIX2'],0.95)
# er_93[pixels_out_93]=np.nan
# theta_grid_93=np.arange(np.min(terminator_coords_occ_93.lat[~np.isnan(er_93)].value)+1./60./2.,np.max(terminator_coords_occ_93.lat[~np.isnan(er_93)].value),1./60.)
# rad_grid_93=np.arange(np.min(terminator_coords_occ_93.distance[~np.isnan(er_93)].value)+1000./2,np.max(terminator_coords_occ_93[~np.isnan(er_93)].distance.value),1000)
# rad_93,theta_93=np.meshgrid(rad_grid_93,theta_grid_93)
# er_coarse_93=griddata((terminator_coords_occ_93.distance[~np.isnan(er_93)].value.ravel(),terminator_coords_occ_93.lat[~np.isnan(er_93)].value.ravel()), er_93[~np.isnan(er_93)].ravel(), (rad_93, theta_93), method='linear')
# er_coarse_radial_average_93=np.nanmean(er_coarse_93,0)
# =============================================================================



occ_data_303, occ_header_303, coords_occ_303=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-He303_G16_s20192580422567_e20192580422577_c20192580423198.fits")
image_center_occ_303 = tuple(np.array([occ_header_303['CRPIX2'],occ_header_303['CRPIX1']]))
rot_mat_occ_303 = cv2.getRotationMatrix2D(image_center_occ_303, -occ_header_303['CROTA'], 1.0)
terminator_coords_occ_303=get_terminator_coords(coords_occ_303,rot_mat_occ_303)
#sun_coords_occ_303=get_obs_sun_coords(coords_occ_303)
result_occ_303=cv2.warpAffine(occ_data_303, rot_mat_occ_303, occ_data_303.shape[1::-1], flags=cv2.INTER_LINEAR)

top_data_303, top_header_303, coords_topside_303=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-He303_G16_s20192580418567_e20192580418577_c20192580419195.fits")
#terminator_coords_topside_303=get_terminator_coords(coords_topside_303)
sun_coords_topside_303=get_obs_sun_coords(coords_topside_303)
image_center_top_303 = tuple(np.array([top_header_303['CRPIX2'],top_header_303['CRPIX1']]))
rot_mat_top_303 = cv2.getRotationMatrix2D(image_center_top_303, -top_header_303['CROTA'], 1.0)
result_top_303=cv2.warpAffine(top_data_303, rot_mat_top_303, top_data_303.shape[1::-1], flags=cv2.INTER_LINEAR)

# =============================================================================
# num_rows, num_cols = result_occ_303.shape[:2]
# # Creating a translation matrix
# translation_matrix = np.float32([ [1,0,occ_header_303['CRPIX1']-top_header_303['CRPIX1']], [0,1,occ_header_303['CRPIX2']-top_header_303['CRPIX2']] ])
# # Image translation
# result_top_303 = cv2.warpAffine(result_top_303, translation_matrix, (num_cols,num_rows))
# 
# =============================================================================
result_occ_303[np.isnan(result_occ_303)]=0
result_top_303[np.isnan(result_top_303)]=0
shift, error, diffphase = phase_cross_correlation(result_top_303,result_occ_303)
result_occ_303=scipy.ndimage.shift(result_occ_303, shift)

# =============================================================================
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(result_occ_303,origin='lower')
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(result_top_303,origin='lower')
# 
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(top_shift,origin='lower')
# =============================================================================
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(result_occ_303/result_top_303,origin='lower',vmin=0,vmax=1.05)

#interpolator_303=LinearNDInterpolator((sun_coords_topside_303.x.value.flatten(),sun_coords_topside_303.y.value.flatten()),top_data_303.flatten())
#coords_topside_303=interpolator_303((sun_coords_occ_303.x.value.flatten(),sun_coords_occ_303.y.value.flatten()))
#er_303=occ_data_303/coords_topside_303.reshape((1280,1280))

er_303=result_occ_303/result_top_303
pixels_out_303,pixels_in_303=pixels_of_sun(er_303,top_header_303['DIAM_SUN']/2.,top_header_303['CRPIX2'],top_header_303['CRPIX1'],1.0)
er_303[pixels_out_303]=np.nan
translation_matrix = np.float32([ [1,0,640-top_header_303['CRPIX2']], [0,1,640-top_header_303['CRPIX1']]])
er_303_centered = cv2.warpAffine(er_303, translation_matrix, er_303.shape[1::-1])
ret, threshold = cv2.threshold(er_303_centered,.1,1,cv2.THRESH_BINARY)
er_303=horizontalize(er_303_centered,threshold)
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(er_303,origin='lower',vmin=0,vmax=1.05)



# =============================================================================
# #Do Not use
# occ_data_303, occ_header_303, coords_occ_303=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-He303_G16_s20192580420367_e20192580420377_c20192580420586.fits")
# terminator_coords_occ_303=get_terminator_coords(coords_occ_303)
# sun_coords_occ_303=get_obs_sun_coords(coords_occ_303)
# top_data_303, top_header_303, coords_topside_303=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-He303_G16_s20192580418567_e20192580418577_c20192580419195.fits")
# terminator_coords_topside_303=get_terminator_coords(coords_topside_303)
# sun_coords_topside_303=get_obs_sun_coords(coords_topside_303)
# 
# interpolator=LinearNDInterpolator((sun_coords_topside_303.x.value.flatten(),sun_coords_topside_303.y.value.flatten()),top_data_303.flatten())
# coords_topside_303=interpolator((sun_coords_occ_303.x.value.flatten(),sun_coords_occ_303.y.value.flatten()))
# er=occ_data_303/coords_topside_303.reshape((1280,1280))
# pixels_out,pixels_in=pixels_of_sun(er,occ_header_303['DIAM_SUN']/2.,occ_header_303['CRPIX1'],occ_header_303['CRPIX2'],0.95)
# er[pixels_out]=np.nan
# theta_grid=np.arange(np.min(terminator_coords_occ_303.lat[~np.isnan(er)].value)+1./60./2.,np.max(terminator_coords_occ_303.lat[~np.isnan(er)].value),1./60.)
# rad_grid=np.arange(np.min(terminator_coords_occ_303.distance[~np.isnan(er)].value)+1000./2,np.max(terminator_coords_occ_303[~np.isnan(er)].distance.value),1000)
# rad,theta=np.meshgrid(rad_grid,theta_grid)
# er_coarse_303=griddata((terminator_coords_occ_303.distance[~np.isnan(er)].value.ravel(),terminator_coords_occ_303.lat[~np.isnan(er)].value.ravel()), er[~np.isnan(er)].ravel(), (rad, theta), method='linear')
# er_coarse_radial_average_303=np.nanmean(er_coarse_303,0)
# =============================================================================

#195 1
# =============================================================================
# occ_data_195, occ_header_195, coords_occ_195=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe195_G16_s20192580423167_e20192580423177_c20192580423388.fits")
# terminator_coords_occ_195=get_terminator_coords(coords_occ_195)
# sun_coords_occ_195=get_obs_sun_coords(coords_occ_195)
# top_data_195, top_header_195, coords_topside_195=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe195_G16_s20192580419167_e20192580419177_c20192580419386.fits")
# terminator_coords_topside_195=get_terminator_coords(coords_topside_195)
# sun_coords_topside_195=get_obs_sun_coords(coords_topside_195)
# 
# interpolator_195=LinearNDInterpolator((sun_coords_topside_195.x.value.flatten(),sun_coords_topside_195.y.value.flatten()),top_data_195.flatten())
# coords_topside_195=interpolator_195((sun_coords_occ_195.x.value.flatten(),sun_coords_occ_195.y.value.flatten()))
# er_195=occ_data_195/coords_topside_195.reshape((1280,1280))
# pixels_out_195,pixels_in_195=pixels_of_sun(er_195,occ_header_195['DIAM_SUN']/2.,occ_header_195['CRPIX1'],occ_header_195['CRPIX2'],0.95)
# er_195[pixels_out_195]=np.nan
# theta_grid_195=np.arange(np.min(terminator_coords_occ_195.lat[~np.isnan(er_195)].value)+1./60./2.,np.max(terminator_coords_occ_195.lat[~np.isnan(er_195)].value),1./60.)
# rad_grid_195=np.arange(np.min(terminator_coords_occ_195.distance[~np.isnan(er_195)].value)+1000./2,np.max(terminator_coords_occ_195[~np.isnan(er_195)].distance.value),1000)
# rad_195,theta_195=np.meshgrid(rad_grid_195,theta_grid_195)
# er_coarse_195=griddata((terminator_coords_occ_195.distance[~np.isnan(er_195)].value.ravel(),terminator_coords_occ_195.lat[~np.isnan(er_195)].value.ravel()), er_195[~np.isnan(er_195)].ravel(), (rad_195, theta_195), method='linear')
# er_coarse_radial_average_195=np.nanmean(er_coarse_195,0)
# =============================================================================

#195 2
occ_data_195_2, occ_header_195_2, coords_occ_195_2=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe195_G16_s20192580422267_e20192580422277_c20192580422496.fits")
image_center_occ_195_2 = tuple(np.array([occ_header_195_2['CRPIX2'],occ_header_195_2['CRPIX1']]))
rot_mat_occ_195_2 = cv2.getRotationMatrix2D(image_center_occ_195_2, -occ_header_195_2['CROTA'], 1.0)
terminator_coords_occ_195_2=get_terminator_coords(coords_occ_195_2,rot_mat_occ_195_2)
#sun_coords_occ_195_2=get_obs_sun_coords(coords_occ_195_2)
result_occ_195_2=cv2.warpAffine(occ_data_195_2, rot_mat_occ_195_2, occ_data_195_2.shape[1::-1], flags=cv2.INTER_LINEAR)

top_data_195_2, top_header_195_2, coords_topside_195_2=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe195_G16_s20192580419167_e20192580419177_c20192580419386.fits")
#terminator_coords_topside_195_2=get_terminator_coords(coords_topside_195_2)
#sun_coords_topside_195_2=get_obs_sun_coords(coords_topside_195_2)
image_center_top_195_2 = tuple(np.array([top_header_195_2['CRPIX2'],top_header_195_2['CRPIX1']]))
rot_mat_top_195_2 = cv2.getRotationMatrix2D(image_center_top_195_2, -top_header_195_2['CROTA'], 1.0)
result_top_195_2=cv2.warpAffine(top_data_195_2, rot_mat_top_195_2, top_data_195_2.shape[1::-1], flags=cv2.INTER_LINEAR)

# =============================================================================
# num_rows, num_cols = result_occ_195_2.shape[:2]
# # Creating a translation matrix
# translation_matrix = np.float32([ [1,0,occ_header_195_2['CRPIX1']-top_header_195_2['CRPIX1']], [0,1,occ_header_195_2['CRPIX2']-top_header_195_2['CRPIX2']] ])
# # Image translation
# result_top_195_2 = cv2.warpAffine(result_top_195_2, translation_matrix, (num_cols,num_rows))
# 
# =============================================================================
result_occ_195_2[np.isnan(result_occ_195_2)]=0
result_top_195_2[np.isnan(result_top_195_2)]=0
shift, error, diffphase = phase_cross_correlation(result_top_195_2,result_occ_195_2)
result_occ_195_2=scipy.ndimage.shift(result_occ_195_2, shift)

# =============================================================================
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(result_occ_195_2,origin='lower')
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(result_top_195_2,origin='lower')
# 
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(top_shift,origin='lower')
# =============================================================================
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(result_occ_195_2/result_top_195_2,origin='lower',vmin=0,vmax=1.05)

#interpolator_195_2=LinearNDInterpolator((sun_coords_topside_195_2.x.value.flatten(),sun_coords_topside_195_2.y.value.flatten()),top_data_195_2.flatten())
#coords_topside_195_2=interpolator_195_2((sun_coords_occ_195_2.x.value.flatten(),sun_coords_occ_195_2.y.value.flatten()))
#er_195_2=occ_data_195_2/coords_topside_195_2.reshape((1280,1280))
er_195_2=result_occ_195_2/result_top_195_2
pixels_out_195_2,pixels_in_195_2=pixels_of_sun(er_195_2,top_header_195_2['DIAM_SUN']/2.,top_header_195_2['CRPIX2'],top_header_195_2['CRPIX1'],1.0)
er_195_2[pixels_out_195_2]=np.nan
translation_matrix = np.float32([ [1,0,640-occ_header_195_2['CRPIX2']], [0,1,640-occ_header_195_2['CRPIX1']]])
er_195_2_centered = cv2.warpAffine(er_195_2, translation_matrix, er_195_2.shape[1::-1])
ret, threshold = cv2.threshold(er_195_2_centered,.1,1,cv2.THRESH_BINARY)
er_195_2=horizontalize(er_195_2_centered,threshold)
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(er_195_2,origin='lower',vmin=0,vmax=1.05)

#284
occ_data_284, occ_header_284, coords_occ_284=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe284_G16_s20192580422367_e20192580422377_c20192580422593.fits")
image_center_occ_284 = tuple(np.array([occ_header_284['CRPIX2'],occ_header_284['CRPIX1']]))
rot_mat_occ_284 = cv2.getRotationMatrix2D(image_center_occ_284, -occ_header_284['CROTA'], 1.0)
terminator_coords_occ_284=get_terminator_coords(coords_occ_284,rot_mat_occ_284)
#sun_coords_occ_284=get_obs_sun_coords(coords_occ_284)
result_occ_284=cv2.warpAffine(occ_data_284, rot_mat_occ_284, occ_data_284.shape[1::-1], flags=cv2.INTER_LINEAR)

top_data_284, top_header_284, coords_topside_284=get_coords_from_fits(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe284_G16_s20192580418367_e20192580418377_c20192580418593.fits")
#terminator_coords_topside_284=get_terminator_coords(coords_topside_284)
#sun_coords_topside_284=get_obs_sun_coords(coords_topside_284)
image_center_top_284 = tuple(np.array([top_header_284['CRPIX2'],top_header_284['CRPIX1']]))
rot_mat_top_284 = cv2.getRotationMatrix2D(image_center_top_284, -top_header_284['CROTA'], 1.0)
result_top_284=cv2.warpAffine(top_data_284, rot_mat_top_284, top_data_284.shape[1::-1], flags=cv2.INTER_LINEAR)

result_occ_284[np.isnan(result_occ_284)]=0
result_top_284[np.isnan(result_top_284)]=0
shift, error, diffphase = phase_cross_correlation(result_top_284,result_occ_284)
result_occ_284=scipy.ndimage.shift(result_occ_284, shift)

# =============================================================================
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(result_occ_284,origin='lower')
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(result_top_284,origin='lower')
# 
# fig,ax=plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(top_shift,origin='lower')
# =============================================================================
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(result_occ_284/result_top_284,origin='lower',vmin=0,vmax=1.05)

# =============================================================================
# num_rows, num_cols = result_occ_284.shape[:2]
# # Creating a translation matrix
# translation_matrix = np.float32([ [1,0,occ_header_284['CRPIX1']-top_header_284['CRPIX1']], [0,1,occ_header_284['CRPIX2']-top_header_284['CRPIX2']] ])
# # Image translation
# result_top_284 = cv2.warpAffine(result_top_284, translation_matrix, (num_cols,num_rows))
# =============================================================================
#interpolator_284=LinearNDInterpolator((sun_coords_topside_284.x.value.flatten(),sun_coords_topside_284.y.value.flatten()),top_data_284.flatten())
#coords_topside_284=interpolator_284((sun_coords_occ_284.x.value.flatten(),sun_coords_occ_284.y.value.flatten()))
#er_284=occ_data_284/coords_topside_284.reshape((1280,1280))

er_284=result_occ_284/result_top_284
pixels_out_284,pixels_in_284=pixels_of_sun(er_284,top_header_284['DIAM_SUN']/2.,top_header_284['CRPIX2'],top_header_284['CRPIX1'],1.0)
er_284[pixels_out_284]=np.nan
translation_matrix = np.float32([ [1,0,640-top_header_284['CRPIX2']], [0,1,640-top_header_284['CRPIX1']]])
er_284_centered = cv2.warpAffine(er_284, translation_matrix, er_284.shape[1::-1])
ret, threshold = cv2.threshold(er_284_centered,.1,1,cv2.THRESH_BINARY)
er_284=horizontalize(er_284_centered,threshold)
fig,ax=plt.subplots(1)
ax.set_aspect('equal')
ax.imshow(er_284,origin='lower',vmin=0,vmax=1.05)

pix=list(range(0,er_171.shape[0]))
occ_gse_171=SkyCoord(coords_occ_171.observer).transform_to(frames.GeocentricSolarEcliptic).cartesian
occ_dist_171=get_terminator_distance(coords_occ_171)
alt_171=[((np.sqrt(occ_gse_171.y.value**2+occ_gse_171.z.value**2))-6.371E6)/1000.+(i-1280./2.)*occ_dist_171*np.tan(2.5*u.arcsec) for i in pix]

occ_gse_303=SkyCoord(coords_occ_303.observer).transform_to(frames.GeocentricSolarEcliptic).cartesian
occ_dist_303=get_terminator_distance(coords_occ_303)
alt_303=[((np.sqrt(occ_gse_303.y.value**2+occ_gse_303.z.value**2))-6.371E6)/1000.+(i-1280./2.)*occ_dist_303*np.tan(2.5*u.arcsec) for i in pix]

occ_gse_195_2=SkyCoord(coords_occ_195_2.observer).transform_to(frames.GeocentricSolarEcliptic).cartesian
occ_dist_195_2=get_terminator_distance(coords_occ_195_2)
alt_195_2=[((np.sqrt(occ_gse_195_2.y.value**2+occ_gse_195_2.z.value**2))-6.371E6)/1000.+(i-1280./2.)*occ_dist_195_2*np.tan(2.5*u.arcsec) for i in pix]

occ_gse_284=SkyCoord(coords_occ_284.observer).transform_to(frames.GeocentricSolarEcliptic).cartesian
occ_dist_284=get_terminator_distance(coords_occ_284)
alt_284=[((np.sqrt(occ_gse_284.y.value**2+occ_gse_284.z.value**2))-6.371E6)/1000.+(i-1280./2.)*occ_dist_284*np.tan(2.5*u.arcsec) for i in pix]

fig,ax = plt.subplots(1)
plt.plot(alt_171,np.nanmean(er_171,1))
plt.plot(alt_303,np.nanmean(er_303,1))
plt.plot(alt_195_2,np.nanmean(er_195_2,1))
plt.plot(alt_284,np.nanmean(er_284,1))
leg=plt.legend(['171','303','195','284'])
plt.ylabel('ER')
plt.xlabel('Alt (km)')


#grid over all images in occultation
all_rad_grid=np.arange(np.min([np.min(terminator_coords_occ_171.distance[~np.isnan(er_171)].value),np.min(terminator_coords_occ_303.distance[~np.isnan(er_303)].value),np.min(terminator_coords_occ_195_2.distance[~np.isnan(er_195_2)].value),np.min(terminator_coords_occ_284.distance[~np.isnan(er_284)].value)])+1000./2,np.max([np.max(terminator_coords_occ_171.distance[~np.isnan(er_171)].value),np.max(terminator_coords_occ_303.distance[~np.isnan(er_303)].value),np.max(terminator_coords_occ_195_2.distance[~np.isnan(er_195_2)].value),np.max(terminator_coords_occ_284.distance[~np.isnan(er_284)].value)])+999.99,1000)
all_theta_grid=np.arange(np.min([np.min(terminator_coords_occ_171.lat[~np.isnan(er_171)].value),np.min(terminator_coords_occ_303.lat[~np.isnan(er_303)].value),np.min(terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value),np.min(terminator_coords_occ_284.lat[~np.isnan(er_284)].value)])+1./60./2.,np.max([np.max(terminator_coords_occ_171.lat[~np.isnan(er_171)].value),np.max(terminator_coords_occ_303.lat[~np.isnan(er_303)].value),np.max(terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value),np.max(terminator_coords_occ_284.lat[~np.isnan(er_284)].value)])+1./60.*.99999,1./60.)

#171 regrid
#theta_grid_171=np.arange(np.min(terminator_coords_occ_171.lat[~np.isnan(er_171)].value)+1./60./2.,np.max(terminator_coords_occ_171.lat[~np.isnan(er_171)].value),1./60.)
#rad_grid_171=np.arange(np.min(terminator_coords_occ_171.distance[~np.isnan(er_171)].value)+1000./2,np.max(terminator_coords_occ_171[~np.isnan(er_171)].distance.value),1000)
rad_grid_171=all_rad_grid[(np.abs(all_rad_grid-np.min(terminator_coords_occ_171.distance[~np.isnan(er_171)].value))).argmin():(np.abs(all_rad_grid-np.max(terminator_coords_occ_171.distance[~np.isnan(er_171)].value))).argmin()+1]
theta_grid_171=all_theta_grid[(np.abs(all_theta_grid-np.min(terminator_coords_occ_171.lat[~np.isnan(er_171)].value))).argmin():(np.abs(all_theta_grid-np.max(terminator_coords_occ_171.lat[~np.isnan(er_171)].value))).argmin()+1]
#rad_171,theta_171=np.meshgrid(rad_grid_171,theta_grid_171)
#er_coarse_171=griddata((terminator_coords_occ_171.distance[~np.isnan(er_171)].value.ravel(),terminator_coords_occ_171.lat[~np.isnan(er_171)].value.ravel()), er_171[~np.isnan(er_171)].ravel(), (rad_171, theta_171), method='linear')
rad=terminator_coords_occ_171.distance.value#cv2.warpAffine(terminator_coords_occ_171.distance, rot_mat_occ_171, occ_data_171.shape[1::-1])
theta=terminator_coords_occ_171.lat.value#cv2.warpAffine(terminator_coords_occ_171.lat, rot_mat_occ_171, occ_data_171.shape[1::-1])
#er_coarse_171=griddata((rad[~np.isnan(er_171)].ravel(),theta[~np.isnan(er_171)].ravel()), er_171[~np.isnan(er_171)].ravel(), (rad_171, theta_171), method='linear')

#er_coarse_radial_average_171=np.nanmean(er_coarse_171,0)
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

rad=terminator_coords_occ_303.distance.value#cv2.warpAffine(terminator_coords_occ_303.distance, rot_mat_occ_303, occ_data_303.shape[1::-1])
theta=terminator_coords_occ_303.lat#cv2.warpAffine(terminator_coords_occ_303.lat, rot_mat_occ_303, occ_data_303.shape[1::-1])
#avg_er_303=[[]]
avg_er_303=[]
for i,alt in enumerate(rad_grid_303,start=0):
    #avg_er_theta_303=[]
    #for j,lat in enumerate(theta_grid_303,start=0):
        #avg_er_theta_303.append(np.nanmean(er_303[(rad<rad_grid_303[i]+500.)&(rad >= rad_grid_303[i]-500.)&(theta < theta_grid_303[j]+1./60./2.)&(theta >= theta_grid_303[j]-1./60./2.)]))
    #avg_er_303.append(avg_er_theta_303)
    #print(avg_er_theta_303)
    avg_er_303.append(np.nanmean(er_303[((rad < rad_grid_303[i]+500.) & (rad >= rad_grid_303[i]-500.))]))

        
    
#195 regrid
#theta_grid_195_2=np.arange(np.min(terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value)+1./60./2.,np.max(terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value),1./60.)
#rad_grid_195_2=np.arange(np.min(terminator_coords_occ_195_2.distance[~np.isnan(er_195_2)].value)+1000./2,np.max(terminator_coords_occ_195_2[~np.isnan(er_195_2)].distance.value),1000)
rad_grid_195_2=all_rad_grid[(np.abs(all_rad_grid-np.min(terminator_coords_occ_195_2.distance[~np.isnan(er_195_2)].value))).argmin():(np.abs(all_rad_grid-np.max(terminator_coords_occ_195_2.distance[~np.isnan(er_195_2)].value))).argmin()+1]
theta_grid_195_2=all_theta_grid[(np.abs(all_theta_grid-np.min(terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value))).argmin():(np.abs(all_theta_grid-np.max(terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value))).argmin()+1]
#rad_195_2,theta_195_2=np.meshgrid(rad_grid_195_2,theta_grid_195_2)
#er_coarse_195_2=griddata((terminator_coords_occ_195_2.distance[~np.isnan(er_195_2)].value.ravel(),terminator_coords_occ_195_2.lat[~np.isnan(er_195_2)].value.ravel()), er_195_2[~np.isnan(er_195_2)].ravel(), (rad_195_2, theta_195_2), method='linear')
#er_coarse_radial_average_195_2=np.nanmean(er_coarse_195_2,0)
rad=terminator_coords_occ_195_2.distance.value#cv2.warpAffine(terminator_coords_occ_195_2.distance, rot_mat_occ_195_2, occ_data_195_2.shape[1::-1])
theta=terminator_coords_occ_195_2.lat.value#cv2.warpAffine(terminator_coords_occ_195_2.lat, rot_mat_occ_195_2, occ_data_195_2.shape[1::-1])
avg_er_195_2=[]
for i,alt in enumerate(rad_grid_195_2,start=0):
    avg_er_195_2.append(np.nanmean(er_195_2[((rad < rad_grid_195_2[i]+500.) & (rad >= rad_grid_195_2[i]-500.))]))

#284 regrid
#theta_grid_284=np.arange(np.min(terminator_coords_occ_284.lat[~np.isnan(er_284)].value)+1./60./2.,np.max(terminator_coords_occ_284.lat[~np.isnan(er_284)].value),1./60.)
#rad_grid_284=np.arange(np.min(terminator_coords_occ_284.distance[~np.isnan(er_284)].value)+1000./2,np.max(terminator_coords_occ_284[~np.isnan(er_284)].distance.value),1000)
rad_grid_284=all_rad_grid[(np.abs(all_rad_grid-np.min(terminator_coords_occ_284.distance[~np.isnan(er_284)].value))).argmin():(np.abs(all_rad_grid-np.max(terminator_coords_occ_284.distance[~np.isnan(er_284)].value))).argmin()+1]
theta_grid_284=all_theta_grid[(np.abs(all_theta_grid-np.min(terminator_coords_occ_284.lat[~np.isnan(er_284)].value))).argmin():(np.abs(all_theta_grid-np.max(terminator_coords_occ_284.lat[~np.isnan(er_284)].value))).argmin()+1]
#rad_284,theta_284=np.meshgrid(rad_grid_284,theta_grid_284)
#er_coarse_284=griddata((terminator_coords_occ_284.distance[~np.isnan(er_284)].value.ravel(),terminator_coords_occ_284.lat[~np.isnan(er_284)].value.ravel()), er_284[~np.isnan(er_284)].ravel(), (rad_284, theta_284), method='linear')
#er_coarse_radial_average_284=np.nanmean(er_coarse_284,0)
rad=terminator_coords_occ_284.distance.value#cv2.warpAffine(terminator_coords_occ_284.distance, rot_mat_occ_284, occ_data_284.shape[1::-1])
theta=terminator_coords_occ_284.lat.value#cv2.warpAffine(terminator_coords_occ_284.lat, rot_mat_occ_284, occ_data_284.shape[1::-1])
avg_er_284=[]
for i,alt in enumerate(rad_grid_284,start=0):
    avg_er_284.append(np.nanmean(er_284[((rad < rad_grid_284[i]+500.) & (rad >= rad_grid_284[i]-500.))]))
    
tmp=np.intersect1d(rad_grid_303,rad_grid_284)
tmp=np.intersect1d(tmp,rad_grid_171)
tmp=np.intersect1d(tmp,rad_grid_195_2)

avg_er_171=np.array(avg_er_171)
avg_er_303=np.array(avg_er_303)
avg_er_195_2=np.array(avg_er_195_2)
avg_er_284=np.array(avg_er_284)
er_171_intersect=avg_er_171[np.in1d(rad_grid_171,tmp)]
er_303_intersect=avg_er_303[np.in1d(rad_grid_303,tmp)]
er_195_2_intersect=avg_er_195_2[np.in1d(rad_grid_195_2,tmp)]
er_284_intersect=avg_er_284[np.in1d(rad_grid_284,tmp)]

cross_sec=scipy.io.readsav(r'C:\Users\Robert\Documents\SUVI_Occs\photon_cross_sections.sav')

n2=cross_sec['photo'].N2[0].XSECTION[0]
o=cross_sec['photo'].O3P[0].XSECTION[0]
wave=cross_sec['photo'].ANGSTROMS[0]

y=np.matrix( ((np.log(er_171_intersect)),(np.log(er_303_intersect)),(np.log(er_195_2_intersect)),(np.log(er_284_intersect))) )
H=-1*np.matrix( ((o[(np.abs(wave - 171).argmin()),0],n2[(np.abs(wave - 171).argmin()),0]),(o[(np.abs(wave - 303).argmin()),0],n2[(np.abs(wave - 303).argmin()),0]),(o[(np.abs(wave - 195).argmin()),0],n2[(np.abs(wave - 195).argmin()),0]),(o[(np.abs(wave - 284).argmin()),0],n2[(np.abs(wave - 284).argmin()),0])) )
x=np.linalg.inv(H.transpose()*H)*H.transpose()*y

o_col=[]
n_col=[]
tmp_rad=[]
for i in range(len(er_171_intersect)-1):
    y=np.array([np.log(er_171_intersect[i]),np.log(er_303_intersect[i]),np.log(er_195_2_intersect[i]),np.log(er_284_intersect[i])])
    if np.array_equal(np.isnan(y) ,np.array([False,False,False,False])):
        res=scipy.optimize.nnls(H,y[~np.isnan(y)])
        o_col.append(res[0][0])
        n_col.append(res[0][1])
        tmp_rad.append(tmp[i])
        
sigma_o_171=o[(np.abs(wave - 171).argmin()),0]
sigma_o_303=o[(np.abs(wave - 303).argmin()),0]
sigma_o_195=o[(np.abs(wave - 195).argmin()),0]
sigma_o_284=o[(np.abs(wave - 284).argmin()),0]
sigma_n2_171=n2[(np.abs(wave - 171).argmin()),0]
sigma_n2_303=n2[(np.abs(wave - 303).argmin()),0]
sigma_n2_195=n2[(np.abs(wave - 195).argmin()),0]
sigma_n2_284=n2[(np.abs(wave - 284).argmin()),0]

ln_er_171=np.log(er_171_intersect)
ln_er_303=np.log(er_303_intersect)
ln_er_195=np.log(er_195_2_intersect)
ln_er_284=np.log(er_284_intersect)

N_n2_171303=(ln_er_303*sigma_o_171-ln_er_171*sigma_o_303)/(-sigma_n2_303*sigma_o_171+sigma_n2_171*sigma_o_303)
N_o_171303=(ln_er_303*sigma_n2_171-ln_er_171*sigma_n2_303)/(sigma_n2_303*sigma_o_171-sigma_n2_171*sigma_o_303)

N_n2_171195=(ln_er_195*sigma_o_171-ln_er_171*sigma_o_195)/(-sigma_n2_195*sigma_o_171+sigma_n2_171*sigma_o_195)
N_o_171195=(ln_er_195*sigma_n2_171-ln_er_171*sigma_n2_195)/(sigma_n2_195*sigma_o_171-sigma_n2_171*sigma_o_195)

N_n2_171284=(ln_er_284*sigma_o_171-ln_er_171*sigma_o_284)/(-sigma_n2_284*sigma_o_171+sigma_n2_171*sigma_o_284)
N_o_171284=(ln_er_284*sigma_n2_171-ln_er_171*sigma_n2_284)/(sigma_n2_284*sigma_o_171-sigma_n2_171*sigma_o_284)

N_n2_303195=(ln_er_195*sigma_o_303-ln_er_303*sigma_o_195)/(-sigma_n2_195*sigma_o_303+sigma_n2_303*sigma_o_195)
N_o_303195=(ln_er_195*sigma_n2_303-ln_er_303*sigma_n2_195)/(sigma_n2_195*sigma_o_303-sigma_n2_303*sigma_o_195)

N_n2_303284=(ln_er_284*sigma_o_303-ln_er_303*sigma_o_284)/(-sigma_n2_284*sigma_o_303+sigma_n2_303*sigma_o_284)
N_o_303284=(ln_er_284*sigma_n2_303-ln_er_303*sigma_n2_284)/(sigma_n2_284*sigma_o_303-sigma_n2_303*sigma_o_284)

N_n2_195284=(ln_er_284*sigma_o_195-ln_er_195*sigma_o_284)/(-sigma_n2_284*sigma_o_195+sigma_n2_195*sigma_o_284)
N_o_195284=(ln_er_284*sigma_n2_195-ln_er_195*sigma_n2_284)/(sigma_n2_284*sigma_o_195-sigma_n2_195*sigma_o_284)

# =============================================================================
# suvi_hdu=fits.open(r"C:\Users\Robert\Documents\SUVI_Occs\test_fits_files\OR_SUVI-L1b-Fe171_G16_s20192581318080_e20192581318090_c20192581318300.fits")
# #suvi_hdu.verify('silentfix')
# header=fix_suvi_l1b_header(suvi_hdu[0].header)
# data=suvi_hdu[0].data
# suvimap=sunpy.map.sources.SUVIMap(data,header)
# suvi_coords=sunpy.map.all_coordinates_from_map(suvimap)
# 
# gse_suvi=SkyCoord(suvi_coords.observer).transform_to(frames.GeocentricSolarEcliptic)
# sun_center = SkyCoord(0*u.m, 0*u.m, 0*u.m, obstime=gse_suvi.obstime,frame=frames.Heliocentric,\
#                       observer='sun').transform_to(frames.GeocentricSolarEcliptic)
# 
# suvi_sun_line=((gse_suvi.cartesian.x.value,gse_suvi.cartesian.y.value,gse_suvi.cartesian.z.value), \
#                 (sun_center.cartesian.x.value,sun_center.cartesian.y.value,sun_center.cartesian.z.value))
# observer_atmos = intersect_point_line((0.0,0.0,0.0), suvi_sun_line[0], suvi_sun_line[1])
# observer_atmos = SkyCoord(observer_atmos[0].x*u.m,observer_atmos[0].y*u.m,observer_atmos[0].z*u.m,\
#                           obstime=gse_suvi.obstime,frame=frames.GeocentricSolarEcliptic,representation_type='cartesian')
# observer_atmos.observer=observer_atmos
# suvi_obs_distance=np.sqrt((gse_suvi.cartesian.x-observer_atmos.cartesian.x)**2+\
#                           (gse_suvi.cartesian.y-observer_atmos.cartesian.y)**2+\
#                               (gse_suvi.cartesian.z-observer_atmos.cartesian.z)**2)
# 
# atmos_map=SkyCoord(np.tan(suvi_coords.Tx)*suvi_obs_distance,np.tan(suvi_coords.Ty)*suvi_obs_distance,\
#                    observer_atmos.transform_to(frames.Heliocentric).z,obstime=gse_suvi.obstime,\
#                        frame=frames.Heliocentric,representation_type='cartesian',observer=observer_atmos).transform_to(frames.GeocentricSolarEcliptic)
# 
# sunmap=SkyCoord(np.tan(suvi_coords.Tx)*suvi_coords.observer.radius,np.tan(suvi_coords.Ty)*suvi_coords.observer.radius,0.0*u.m,frame=frames.Heliocentric)
# 
# terminator_up=[observer_atmos.x.value,observer_atmos.y.value,observer_atmos.z.value]
# terminator_up=terminator_up/np.linalg.norm(terminator_up)
# terminator_sun=[sun_center.cartesian.x.value-observer_atmos.x.value,sun_center.cartesian.y.value-observer_atmos.y.value,\
#                sun_center.cartesian.z.value-observer_atmos.z.value]
# terminator_sun=terminator_sun/np.linalg.norm(terminator_sun)
# terminator_across=np.cross(terminator_up,terminator_sun)
# 
# fig,ax = plt.subplots(1)
# ax.set_aspect('equal')
# ax.imshow(suvimap.data,cmap='gray')
# circ=Circle((header['CRPIX1'],header['CRPIX2']),15)
# Circle.set_color(circ,'red')
# ax.add_patch(circ)
# plt.show()
# 
# =============================================================================


