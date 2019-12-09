# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:11:47 2019

@author: susanmeerdink
"""

# Importing Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg') # Needed to have figures display properly. 
import flirimageextractor
import utilities as u
import subprocess

## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
filename = 'C:\\Users\\susanmeerdink\\Dropbox (UFL)\\DARPA_SENTINEL\\Data\\Mandi_Datasets\\FLIR7226_0620.jpg'
flir = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Users\\susanmeerdink\\.utilities\\exiftool.exe")
flir.process_image(filename, RGB=True)

## Examine thermal and full resolution RGB images
# Most FLIR cameras take a thermal image and a corresponding RGB image. 
# The RGB camera is higher resolution and has a larger field of view. 
therm = flir.get_thermal_np()
rgb_fullres = flir.get_rgb_np()
#plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
#plt.imshow(therm)
#plt.title('Thermal Image')
#plt.subplot(1,2,2)
#plt.imshow(rgb_fullres)
#plt.title('RGB Full Resolution Image')
#plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

# Set up Arrays
real2ir = 1.32607507705688
height_range = np.arange(0,rgb_fullres.shape[0],real2ir).astype(int)
width_range = np.arange(0,rgb_fullres.shape[1],real2ir).astype(int)
htv, wdv = np.meshgrid(height_range,width_range)

# Assigning low resolution data
lowres_test = np.swapaxes(rgb_fullres[htv, wdv,  :], 0, 1)

# Get difference in resolution    
ht_diff = (lowres_test.shape[0] - therm.shape[0])/2
wd_diff = (lowres_test.shape[1] - therm.shape[1])/2
#yrange = np.arange(ht_diff,ht_diff+therm.shape[0], dtype=int)
#xrange = np.arange(wd_diff,wd_diff+therm.shape[1], dtype=int)
#htv, wdv = np.meshgrid(xrange,yrange)
#crop_test = lowres_test[htv, wdv, :]

# center of rgb image
ht_center = rgb_fullres.shape[0]/2
wd_center = rgb_fullres.shape[1]/2
htl_center = lowres_test.shape[0]/2
wdl_center = lowres_test.shape[1]/2

# assigning crop images with low res image
yrange = np.arange(htl_center-(therm.shape[0])*real2ir,htl_center+(therm.shape[0])*real2ir, dtype=int)
xrange = np.arange(wdl_center-(therm.shape[1])*real2ir,wdl_center+(therm.shape[1])*real2ir, dtype=int)
htv, wdv = np.meshgrid(yrange,xrange)
crop_test = lowres_test[htv, wdv, :]
#mask = np.zeros((lowres_test.shape[0], lowres_test.shape[1]))
#mask[htv,wdv] = 1
#plt.figure()
#plt.imshow(mask)
#plt.show(block='true')


plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.imshow(therm)
plt.title('Thermal Image')
plt.subplot(1,4,2)
plt.imshow(rgb_fullres)
plt.scatter(wd_center,ht_center)
plt.scatter(wd_center-(therm.shape[1]),ht_center+(therm.shape[0]),color='red')
plt.scatter(wd_center+(therm.shape[1]),ht_center-(therm.shape[0]),color='black')
plt.scatter(wd_center+(therm.shape[1]),ht_center+(therm.shape[0]),color='green')
plt.scatter(wd_center-(therm.shape[1]),ht_center-(therm.shape[0]),color='orange')
plt.title('RGB Full Resolution Image')
plt.subplot(1,4,3)
plt.imshow(lowres_test)
plt.scatter(wdl_center,htl_center)
plt.scatter(wdl_center-(therm.shape[1]),htl_center+(therm.shape[0]),color='red')
plt.scatter(wdl_center+(therm.shape[1]),htl_center-(therm.shape[0]),color='black')
plt.scatter(wdl_center+(therm.shape[1]),htl_center+(therm.shape[0]),color='green')
plt.scatter(wdl_center-(therm.shape[1]),htl_center-(therm.shape[0]),color='orange')
plt.title('RGB Low Res')
plt.subplot(1,4,4)
plt.imshow(crop_test)
plt.title('RGB Cropped Test')
plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

# %%

# Get difference in resolution    
ht_diff = (rgb_fullres.shape[0] - lowres_test.shape[0])/2
wd_diff = (rgb_fullres.shape[1] - lowres_test.shape[1])/2
ht_offset_diff = ht_diff - (ht_diff*real2ir)
wd_offset_diff = wd_diff - (wd_diff*real2ir)
yrange = np.arange(ht_diff+ht_offset_diff,rgb_fullres.shape[0]-ht_diff-ht_offset_diff, dtype=int)
xrange = np.arange(wd_diff+wd_offset_diff,rgb_fullres.shape[1]-wd_diff-wd_offset_diff, dtype=int)
htv, wdv = np.meshgrid(yrange,xrange)
crop_test = rgb_fullres[htv, wdv, :]

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.imshow(therm)
plt.title('Thermal Image')
plt.subplot(1,3,2)
plt.imshow(rgb_fullres)
plt.scatter(wd_diff, ht_diff)
plt.scatter(rgb_fullres.shape[1]-wd_diff, rgb_fullres.shape[0]-ht_diff)
plt.title('RGB Full Resolution Image')
plt.subplot(1,3,3)
plt.imshow(crop_test)
plt.title('RGB Cropped Test')
plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

# %%

## Check how well thermal and rgb registration is without manually correction
# You can see that the images do not line up and there is an offset even after 
# correcting for offset provided in file header
# rgb_lowres, rgb_crop = u.extract_coarse_image(flir)
#
#### Determine manual correction of Thermal and RGB registration 
#offset, pts_temp, pts_rgb = u.manual_img_registration(flir)
#print('X pixel offset is ' + str(offset[0]) + ' and Y pixel offset is ' + str(offset[1]))

## %%
#rgb_lowres, rgb_crop = u.extract_coarse_image(flir, offset=offset, plot=1)

## %% 
#plt.figure(figsize=(10,5))
#plt.subplot(1,2,1)
#plt.imshow(rgb_fullres)
#plt.title('RGB Full')
#plt.subplot(1,2,2)
#plt.imshow(rgb_lowres)
#plt.title('RGB Less Resolution Image')
#plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

## %%
#data = subprocess.check_output([flir.exiftool_path, "-FieldofView", "-b", flir.flir_img_filename])
#print(data)