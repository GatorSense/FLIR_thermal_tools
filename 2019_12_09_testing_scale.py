# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:22:49 2019

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
from skimage.transform import rescale, downscale_local_mean
from skimage import data, color

## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
filename = 'C:\\Users\\susanmeerdink\\Dropbox (UFL)\\DARPA_SENTINEL\\Data\\Mandi_Datasets\\FLIR7226_0620.jpg'
flir = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Users\\susanmeerdink\\.utilities\\exiftool.exe")
flir.process_image(filename, RGB=True)


offset, pts_therm, pts_rgb = u.manual_img_registration(flir)
# %%
pts_diff = pts_therm - pts_rgb
print(pts_diff)
offset = np.around(np.mean(pts_diff[3:len(pts_diff),:], axis=0))
print(offset)
# %%
rgb_fullres = flir.get_rgb_np()
therm = flir.get_thermal_np()

# %% Using Real2ir value
real2ir = 1.32607507705688
#scale = 100/(real2ir*100)
offset_x = -78
offset_y = 37
image_rescaled = rescale(rgb_fullres, scale, anti_aliasing=False, multichannel=1)

# center of rgb image
ht_center = image_rescaled.shape[0]/2
wd_center = image_rescaled.shape[1]/2

yrange = np.arange(ht_center-(therm.shape[0]),ht_center+(therm.shape[0]), dtype=int)
xrange = np.arange(wd_center-(therm.shape[1]),wd_center+(therm.shape[1]), dtype=int)
htv, wdv = np.meshgrid(yrange,xrange)
image_crop = np.swapaxes(image_rescaled[htv, wdv, :],1,0)
#
plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.imshow(rgb_fullres)
plt.text(0, rgb_fullres.shape[0]*1.2,('ht ' + str(rgb_fullres.shape[0])))
plt.text(0, rgb_fullres.shape[0]*1.3,('wd ' + str(rgb_fullres.shape[1])))
plt.title('RGB Full Resolution Image')
plt.subplot(1,4,2)
plt.imshow(image_rescaled)
plt.text(0, image_rescaled.shape[0]*1.2,('ht ' + str(image_rescaled.shape[0])))
plt.text(0, image_rescaled.shape[0]*1.3,('wd ' + str(image_rescaled.shape[1])))
plt.title('RGB rescaled')
plt.subplot(1,4,3)
plt.imshow(image_crop)
plt.text(0, image_crop.shape[0]*1.2,('ht ' + str(image_crop.shape[0])))
plt.text(0, image_crop.shape[0]*1.3,('wd ' + str(image_crop.shape[1])))
plt.title('RGB Crop')
plt.subplot(1,4,4)
plt.imshow(therm)
plt.text(0, therm.shape[0]*1.2,('ht ' + str(therm.shape[0])))
plt.text(0, therm.shape[0]*1.3,('wd ' + str(therm.shape[1])))
plt.title('Therm')
plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

# %% Using full to low RBG changes
#scale = 0.2165
#image_rescaled = rescale(rgb_fullres, scale, anti_aliasing=False, multichannel=1)

## center of rgb image
#ht_center = image_rescaled.shape[0]/2
#wd_center = image_rescaled.shape[1]/2
#
#yrange = np.arange(ht_center-(therm.shape[0]/2),ht_center+(therm.shape[0]/2), dtype=int)
#xrange = np.arange(wd_center-(therm.shape[1]/2),wd_center+(therm.shape[1]/2), dtype=int)
#htv, wdv = np.meshgrid(yrange,xrange)
#image_crop = np.swapaxes(image_rescaled[htv, wdv, :],1,0)
#
#plt.figure(figsize=(15,5))
#plt.subplot(1,3,1)
#plt.imshow(rgb_fullres)
#plt.text(0, rgb_fullres.shape[0]*1.2,('ht ' + str(rgb_fullres.shape[0])))
#plt.text(0, rgb_fullres.shape[0]*1.3,('wd ' + str(rgb_fullres.shape[1])))
#plt.title('RGB Full Resolution Image')
#plt.subplot(1,3,2)
#plt.imshow(image_rescaled)
#plt.text(0, image_rescaled.shape[0]*1.2,('ht ' + str(image_rescaled.shape[0])))
#plt.text(0, image_rescaled.shape[0]*1.3,('wd ' + str(image_rescaled.shape[1])))
#plt.title('RGB rescaled')
#plt.subplot(1,4,3)
#plt.imshow(image_crop)
#plt.text(0, image_crop.shape[0]*1.2,('ht ' + str(image_crop.shape[0])))
#plt.text(0, image_crop.shape[0]*1.3,('wd ' + str(image_crop.shape[1])))
#plt.title('RGB Crop')
#plt.subplot(1,3,3)
#plt.imshow(therm)
#plt.text(0, therm.shape[0]*1.2,('ht ' + str(therm.shape[0])))
#plt.text(0, therm.shape[0]*1.3,('wd ' + str(therm.shape[1])))
#plt.title('Therm')
#plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

# %% Using megapixels

scale = 0.3
image_rescaled = rescale(rgb_fullres, scale, anti_aliasing=False, multichannel=1)

# center of rgb image
ht_center = image_rescaled.shape[0]/2
wd_center = image_rescaled.shape[1]/2
#yrange = np.arange(ht_center-(therm.shape[0]/2),ht_center+(therm.shape[0]/2), dtype=int)
#xrange = np.arange(wd_center-(therm.shape[1]/2),wd_center+(therm.shape[1]/2), dtype=int)
#htv, wdv = np.meshgrid(yrange,xrange)
#image_crop = np.swapaxes(image_rescaled[htv, wdv, :],1,0)
#
height_range = np.arange(-offset[0],-offset[0]+(therm.shape[0])).astype(int)
width_range = np.arange(-offset[1],-offset[1]+(therm.shape[1])).astype(int)
xv, yv = np.meshgrid(height_range,width_range)
image_crop = np.swapaxes(image_rescaled[xv, yv, :],0,1)
    
#
plt.figure(figsize=(15,5))
plt.subplot(1,4,1)
plt.imshow(rgb_fullres)
plt.text(0, rgb_fullres.shape[0]*1.2,('ht ' + str(rgb_fullres.shape[0])))
plt.text(0, rgb_fullres.shape[0]*1.3,('wd ' + str(rgb_fullres.shape[1])))
plt.title('RGB Full Resolution Image')
plt.subplot(1,4,2)
plt.imshow(image_rescaled)
plt.text(0, image_rescaled.shape[0]*1.2,('ht ' + str(image_rescaled.shape[0])))
plt.text(0, image_rescaled.shape[0]*1.3,('wd ' + str(image_rescaled.shape[1])))
plt.title('RGB rescaled')
plt.subplot(1,4,3)
plt.imshow(image_crop)
plt.text(0, image_crop.shape[0]*1.2,('ht ' + str(image_crop.shape[0])))
plt.text(0, image_crop.shape[0]*1.3,('wd ' + str(image_crop.shape[1])))
plt.title('RGB Crop')
plt.subplot(1,4,4)
plt.imshow(therm)
plt.text(0, therm.shape[0]*1.2,('ht ' + str(therm.shape[0])))
plt.text(0, therm.shape[0]*1.3,('wd ' + str(therm.shape[1])))
plt.title('Therm')
plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed