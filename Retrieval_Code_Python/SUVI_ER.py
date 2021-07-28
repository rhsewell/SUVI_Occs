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
from scipy import interpolate

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

def horizontalize(img,thresh,typeOcc):
    X = np.array(np.where(thresh > 0)).T
    # Perform a PCA and compute the angle of the first principal axes
    pca = PCA(n_components=2).fit(X)
    if typeOcc == 'dusk':
        angle = np.mod(np.arctan2(*pca.components_[0])/np.pi*180,180)
    else:
        angle = np.mod(np.arctan2(*pca.components_[0])/np.pi*180,180)+180
    print(angle)
    rot_mat = cv2.getRotationMatrix2D((img.shape[0]/2.,img.shape[1]/2.), angle, 1.0)
    # Rotate the image by the computed angle:
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1])
    return rotated_img

def makeER(occultation_fits,topside_fits,typeOcc):
    occ_data, occ_header, coords_occ=get_coords_from_fits(occultation_fits)
    image_center_occ = tuple(np.array([occ_header['CRPIX2'],occ_header['CRPIX1']]))
    rot_mat_occ = cv2.getRotationMatrix2D(image_center_occ, -occ_header['CROTA'], 1.0)
    terminator_coords_occ=get_terminator_coords(coords_occ,rot_mat_occ)
    result_occ=cv2.warpAffine(occ_data, rot_mat_occ, occ_data.shape[1::-1])#, flags=cv2.INTER_LINEAR)
    
    top_data, top_header, coords_topside=get_coords_from_fits(topside_fits)
    image_center_top = tuple(np.array([top_header['CRPIX2'],top_header['CRPIX1']]))
    rot_mat_top = cv2.getRotationMatrix2D(image_center_top, -top_header['CROTA'], 1.0)
    result_top=cv2.warpAffine(top_data, rot_mat_top, top_data.shape[1::-1])#, flags=cv2.INTER_LINEAR)
    
    result_occ[np.isnan(result_occ)]=0
    result_top[np.isnan(result_top)]=0
    shift, error, diffphase = phase_cross_correlation(result_top,result_occ)
    result_occ=scipy.ndimage.shift(result_occ, shift)
    
    er=result_occ/result_top
    pixels_out,pixels_in=pixels_of_sun(er,top_header['DIAM_SUN']/2.,top_header['CRPIX2'],top_header['CRPIX1'],1.0)
    er[pixels_out]=np.nan
    result_occ[pixels_out]=np.nan
    translation_matrix = np.float32([ [1,0,639-top_header['CRPIX2']], [0,1,639-top_header['CRPIX1']]])
    er_centered = cv2.warpAffine(er, translation_matrix, er.shape[1::-1])
    result_occ_centered=cv2.warpAffine(result_occ, translation_matrix, er.shape[1::-1])
    
    ret, threshold = cv2.threshold(er_centered,0.5,1,cv2.THRESH_BINARY)
    
    #dx,dy=np.gradient(threshold)
    #average_gradient = np.arctan2(-np.mean(dy[dy >0]), np.mean(dx[dx>0]))
    #average_gradient=np.mean(average_gradient[average_gradient>0])
    #px = [639+np.cos(average_gradient) * 100, 639]
    #py = [639-np.cos(average_gradient) * 100,639]
    #plt.plot(px,py,color="white", linewidth=3)
    
    #rot_mat_grad = cv2.getRotationMatrix2D(tuple(np.array([639,639])), np.cos(-np.sin(average_gradient)/np.cos(average_gradient))*180./np.pi, 1.0)
    #rot_er=cv2.warpAffine(er_centered, rot_mat_grad, er.shape[1::-1])
    
    er=horizontalize(er_centered,threshold,typeOcc)
    occ=horizontalize(result_occ_centered, threshold, typeOcc)
    #fig,ax=plt.subplots(1)
    #ax.set_aspect('equal')
    #ax.imshow(er,origin='lower',vmin=0,vmax=1.05)
    
    #plt.title(occultation_fits+'\n'+topside_fits)
    pixels_out,pixels_in=pixels_of_sun(er,top_header['DIAM_SUN']/2.,639,639,1.0)
    er[pixels_out]=np.nan
    occ[pixels_out]=np.nan
    
    pix=list(range(0,er.shape[0]))
    occ_gse=SkyCoord(coords_occ.observer).transform_to(frames.GeocentricSolarEcliptic).cartesian
    occ_dist=get_terminator_distance(coords_occ)
    alt=np.array([((np.sqrt(occ_gse.y.value**2+occ_gse.z.value**2))-6.371E6)/1000.+(i-1280./2.)*occ_dist*np.tan(2.5*u.arcsec).value for i in pix])
    avg_er=np.nanmean(er,1)
    median_er=np.nanmedian(er,1)
    freq_er_50=[]
    freq_er=[]
    for i,row in enumerate(er[:][0]-1):
        res=np.histogram(er[i][:][~np.isnan(er[i][:]) & ~np.isinf(er[i][:])],50)
        freq_er_50.append(res[1][np.argmax(res[0])])
    freq_er_50=np.array(freq_er_50)
    for i,row in enumerate(er[:][0]-1):
        res=np.histogram(er[i][:][~np.isnan(er[i][:]) & ~np.isinf(er[i][:])],500)
        freq_er.append(res[1][np.argmax(res[0])])
    freq_er=np.array(freq_er)
    
    #avg_occ=np.nanmedian(occ,1)
    alt=alt[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    median_er=median_er[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    freq_er=freq_er[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    freq_er_50=freq_er_50[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    fig,ax=plt.subplots(1)
    plt.plot(alt,freq_er_50,'red')
    plt.plot(alt,freq_er,'blue')
    ax.legend(['Bins=50', 'Bins=500'])
    #avg_occ=avg_occ[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    avg_er=avg_er[(~np.isinf(avg_er) & ~np.isnan(avg_er))]
    
    return er, avg_er, median_er, freq_er, alt, occ_header['DATE-OBS'], occ_header['OBSGEO-X'], occ_header['OBSGEO-Y'], occ_header['OBSGEO-Z'], occ_gse

def make_col_density(channels,avg_er_list,alt_list,o,n2,wave):
    alts=[]
    ers=[]
    os=[]
    n2s=[]
    if 'Fe171' in channels:
        alt_fe171=np.concatenate(alt_list[np.where(channels=='Fe171')])
        avg_er_fe171=np.concatenate(avg_er_list[np.where(channels=='Fe171')])
        avg_er_fe171=avg_er_fe171[np.argsort(alt_fe171)]
        alt_fe171=alt_fe171[np.argsort(alt_fe171)]
        alt_fe171=alt_fe171[~np.isnan(avg_er_fe171) & ~np.isinf(avg_er_fe171)]
        avg_er_fe171=avg_er_fe171[~np.isnan(avg_er_fe171) & ~np.isinf(avg_er_fe171)]        
        sigma_o_171=o[(np.abs(wave - 171).argmin()),0]
        sigma_n2_171=n2[(np.abs(wave - 171).argmin()),0]
        alts.append(alt_fe171)
        ers.append(avg_er_fe171)
        os.append(sigma_o_171)
        n2s.append(sigma_n2_171)
    if 'He304' in channels:
        alt_he304=np.concatenate(alt_list[np.where(channels=='He304')])
        avg_er_he304=np.concatenate(avg_er_list[np.where(channels=='He304')])
        avg_er_he304=avg_er_he304[np.argsort(alt_he304)]
        alt_he304=alt_he304[np.argsort(alt_he304)]
        alt_he304=alt_he304[~np.isnan(avg_er_he304) & ~np.isinf(avg_er_he304)]
        avg_er_he304=avg_er_he304[~np.isnan(avg_er_he304) & ~np.isinf(avg_er_he304)]  
        sigma_o_304=o[(np.abs(wave - 304).argmin()),0]
        sigma_n2_304=n2[(np.abs(wave - 304).argmin()),0]
        alts.append(alt_he304)
        ers.append(avg_er_he304)
        os.append(sigma_o_304)
        n2s.append(sigma_n2_304)
    if 'Fe284' in channels:
        alt_fe284=np.concatenate(alt_list[np.where(channels=='Fe284')])
        avg_er_fe284=np.concatenate(avg_er_list[np.where(channels=='Fe284')])
        avg_er_fe284=avg_er_fe284[np.argsort(alt_fe284)]
        alt_fe284=alt_fe284[np.argsort(alt_fe284)]
        alt_fe284=alt_fe284[~np.isnan(avg_er_fe284) & ~np.isinf(avg_er_fe284)]
        avg_er_fe284=avg_er_fe284[~np.isnan(avg_er_fe284) & ~np.isinf(avg_er_fe284)]  
        sigma_o_284=o[(np.abs(wave - 284).argmin()),0]
        sigma_n2_284=n2[(np.abs(wave - 284).argmin()),0]
        alts.append(alt_fe284)
        ers.append(avg_er_fe284)
        os.append(sigma_o_284)
        n2s.append(sigma_n2_284)
    if 'Fe195' in channels:
        alt_fe195=np.concatenate(alt_list[np.where(channels=='Fe195')])
        avg_er_fe195=np.concatenate(avg_er_list[np.where(channels=='Fe195')])
        avg_er_fe195=avg_er_fe195[np.argsort(alt_fe195)]
        alt_fe195=alt_fe195[np.argsort(alt_fe195)]
        alt_fe195=alt_fe195[~np.isnan(avg_er_fe195) & ~np.isinf(avg_er_fe195)]
        avg_er_fe195=avg_er_fe195[~np.isnan(avg_er_fe195) & ~np.isinf(avg_er_fe195)]  
        sigma_o_195=o[(np.abs(wave - 195).argmin()),0]
        sigma_n2_195=n2[(np.abs(wave - 195).argmin()),0]
        alts.append(alt_fe195)
        ers.append(avg_er_fe195)
        os.append(sigma_o_195)
        n2s.append(sigma_n2_195)
        
    min_all=np.max([np.min(alt) for alt in alts])
    max_all=np.min([np.max(alt) for alt in alts])
    interp_ers=np.array([])
    y=np.matrix([])
    H=np.matrix([])
    
    for i,er in enumerate(ers-1):
        ers[i]=ers[i][(alts[i][:]>min_all) & (alts[i][:]<max_all)]
        alts[i]=ers[i][(alts[i][:]>min_all) & (alts[i][:]<max_all)]
        f_interp=interpolate.interp1d(alts[i],ers[i],kind='cubic')
        interp_ers=np.append(f_interp(alts[0]),interp_ers)
        if i==0:
            y=np.matrix( ((np.log(interp_ers[i]))) )
            H=np.matrix( -1*((os[i],n2s[i])) )
        else:
            y=np.append([np.log(interp_ers[i])],y,axis=0)
            H=np.append(-1*((os[i],n2s[i])), H,axis=0)
            
    x=np.linalg.inv(H.transpose()*H)*H.transpose()*y
    return alts[0],x
