# -*- coding: utf-8 -*-
"""

Testing Gaussian Mixture Models for Classification

Created on Thu Mar 26 10:33:42 2020

@author: sofiavega
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg') # Needed to have figures display properly in spyder. 
import flirimageextractor
import FLIR_thermal_tools.utilities as u
import cv2
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.mixture import GaussianMixture
import random

## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
filename = 'C:\\Users\\sofiavega\\Downloads\\2020-03-02_mandi\\psent2-18-6\\IR_10379.jpg'
flir = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Users\\sofiavega\\AppData\\Local\\Temp\\Temp1_exiftool-11.91.zip\\exiftool(-k)")
flir.process_image(filename, RGB=True)

## Examine thermal and full resolution RGB images
# Most FLIR cameras take a thermal image and a corresponding RGB image. 
# The RGB camera is higher resolution and has a larger field of view. 
therm = flir.get_thermal_np()
rgb_fullres = flir.get_rgb_np()
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(therm)
plt.title('Thermal Image')
plt.subplot(1,2,2)
plt.imshow(rgb_fullres)
plt.title('RGB Full Resolution Image')
plt.show(block='TRUE') # I needed to have block=TRUE for image to remain displayed

## Check how well thermal and rgb registration is without manually correction
# You can see that the images do not line up and there is an offset even after 
# correcting for offset provided in file header
#The rgb cropped seems pretty close
rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

offset = [-69, -71]  # This i the manual offset I got when running the demo images.
rgb_lowres, rgb_crop = u.extract_rescale_image(flir, offset=offset, plot=1)

#plot in spyder plot tab
%matplotlib inline



random.seed(1) #set seed

img = rgb_crop
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
num_class = 7
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))
plt.imshow(label_image)


# Plotting Results
coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img)
ax1.set_title('Original Image') 
ax1.set_xticks([])
ax1.set_yticks([])
ax2 = fig.add_subplot(1,2,2)
cmap = colors.ListedColormap(coloroptions[0:num_class])
ax2.imshow(label_image, cmap=cmap)
ax2.set_title('GMM with RGB Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#try with hsv
random.seed(175)
img = np.array(rgb_crop)
# make sure that values are between 0 and 255, i.e. within 8bit range
img *= 255/img.max() 
# cast to 8bit
img = np.array(img, np.uint8)
hsv_mask = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

vectorized = hsv_mask.reshape((-1,3))
vectorized = np.float32(vectorized)
num_class = 9
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)

    
# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))
plt.imshow(label_image)


# Plotting Results
coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(img)
ax1.set_title('Original Image') 
ax1.set_xticks([])
ax1.set_yticks([])
ax2 = fig.add_subplot(1,2,2)
cmap = colors.ListedColormap(coloroptions[0:num_class])
ax2.imshow(label_image, cmap=cmap)
ax2.set_title('GMM with HSV Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#######################################################
#create GMM classification function
def GMM_rgb(image,num_class,hsv= 0, plot=1): 
    """
    This classifies an RGB image using Gaussian Mixture Modeling.
    Note: only 10 colors are specified, so will have plotting error with K > 10
    INPUTS:
        1) img: a 3D numpy array of rgb image
        2) num_class: number of GMM classes
        3) hsv: transform the image from rgb to hsv or lab
                1 = transform to hsv, 0 = keep RGB (default), 2 = transform to lab
        4) plot: a flag that determine if multiple figures of classified is displayed. 
                1 = plot displayed, 0 = no plot is displayed
    OUTPUTS:
        1) label_image: a 2D numpy array the same x an y dimensions as input rgb image, 
            but each pixel is a GMM class.
        
    """
    img = np.array(image)
   
    if hsv == 2:
        #Transform to LAB
        # make sure that values are between 0 and 255, i.e. within 8bit range
        img *= 255/img.max() 
        # cast to 8bit
        img = np.array(img, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
         #Prepare Image
        vectorized = img.reshape((-1,3))
        vectorized = np.float32(vectorized)
    
        #Gaussian Mixture Modeling
        gmm = GaussianMixture(n_components=num_class).fit(vectorized)
        labels = gmm.predict(vectorized)
        
        
        # Labeled class image
        label_image = labels.reshape((img.shape[0], img.shape[1]))
        
        
        if plot == 1:
            # Plotting Results
            coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
            fig = plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(image)
            ax1.set_title('Original Image') 
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = fig.add_subplot(1,2,2)
            cmap = colors.ListedColormap(coloroptions[0:num_class])
            ax2.imshow(label_image, cmap=cmap)
            ax2.set_title('GMM with LAB Classes = ' + str(num_class) )
            ax2.set_xticks([]) 
            ax2.set_yticks([])
            fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
            plt.show(block='TRUE')
            
            # Plotting just GMM with label
            ticklabels = ['1','2','3','4','5','6','7','8','9','10']
            fig, ax = plt.subplots(figsize=(5,5))
            im = ax.imshow(label_image, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, shrink = 0.6, ticks=np.arange(0,num_class)) 
            cbar.ax.set_yticklabels(ticklabels[0:num_class]) 
            cbar.ax.set_ylabel('Classes')
            plt.show(block='TRUE')
   
    
    if hsv == 1:
        #Transform to HSV
        # make sure that values are between 0 and 255, i.e. within 8bit range
        img *= 255/img.max() 
        # cast to 8bit
        img = np.array(img, np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
         #Prepare Image
        vectorized = img.reshape((-1,3))
        vectorized = np.float32(vectorized)
    
        #Gaussian Mixture Modeling
        gmm = GaussianMixture(n_components=num_class).fit(vectorized)
        labels = gmm.predict(vectorized)
        
        
        # Labeled class image
        label_image = labels.reshape((img.shape[0], img.shape[1]))
        
        
        if plot == 1:
            # Plotting Results
            coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
            fig = plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(image)
            ax1.set_title('Original Image') 
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = fig.add_subplot(1,2,2)
            cmap = colors.ListedColormap(coloroptions[0:num_class])
            ax2.imshow(label_image, cmap=cmap)
            ax2.set_title('GMM with HSV Classes = ' + str(num_class) )
            ax2.set_xticks([]) 
            ax2.set_yticks([])
            fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
            plt.show(block='TRUE')
            
            # Plotting just GMM with label
            ticklabels = ['1','2','3','4','5','6','7','8','9','10']
            fig, ax = plt.subplots(figsize=(5,5))
            im = ax.imshow(label_image, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, shrink = 0.6, ticks=np.arange(0,num_class)) 
            cbar.ax.set_yticklabels(ticklabels[0:num_class]) 
            cbar.ax.set_ylabel('Classes')
            plt.show(block='TRUE')
        
    if hsv == 0:
        #Prepare Image
        vectorized = img.reshape((-1,3))
        vectorized = np.float32(vectorized)
    
        #Gaussian Mixture Modeling
        gmm = GaussianMixture(n_components=num_class).fit(vectorized)
        labels = gmm.predict(vectorized)
        
        
        # Labeled class image
        label_image = labels.reshape((img.shape[0], img.shape[1]))
        
        
        if plot == 1:
            # Plotting Results
            coloroptions = ['b','g','r','c','m','y','k','orange','navy','gray']
            fig = plt.figure(figsize=(10,5))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(image)
            ax1.set_title('Original Image') 
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = fig.add_subplot(1,2,2)
            cmap = colors.ListedColormap(coloroptions[0:num_class])
            ax2.imshow(label_image, cmap=cmap)
            ax2.set_title('GMM with RGB Classes = ' + str(num_class) )
            ax2.set_xticks([]) 
            ax2.set_yticks([])
            fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
            plt.show(block='TRUE')
            
            # Plotting just GMM with label
            ticklabels = ['1','2','3','4','5','6','7','8','9','10']
            fig, ax = plt.subplots(figsize=(5,5))
            im = ax.imshow(label_image, cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, shrink = 0.6, ticks=np.arange(0,num_class)) 
            cbar.ax.set_yticklabels(ticklabels[0:num_class]) 
            cbar.ax.set_ylabel('Classes')
            plt.show(block='TRUE')
        
    return label_image

GMM_rgb(image=rgb_crop,num_class=7,hsv = 0, plot=1)


