# -*- coding: utf-8 -*-
"""
Testing cluster.MeanShift for classification

Created on Sun Mar 29 18:40:21 2020

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
from sklearn.cluster import spectral_clustering
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans

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

#use RGB crop
%matplotlib inline


##Test Mean Shift
img = rgb_crop
vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)


# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(vectorized, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(vectorized)

labels = ms.labels_ 



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

ax2.imshow(label_image)
ax2.set_title('Mean Shift with RGB ' )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#try with hsv
img = np.array(rgb_crop)
# make sure that values are between 0 and 255, i.e. within 8bit range
img *= 255/img.max() 
# cast to 8bit
img = np.array(img, np.uint8)
hsv_mask = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

vectorized = hsv_mask.reshape((-1,3))
vectorized = np.float32(vectorized)

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(vectorized, quantile=0.2, n_samples=500)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(vectorized)

labels = ms.labels_ 



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

ax2.imshow(label_image)
ax2.set_title('Mean Shift with HSV ' )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#############################################################################
#Try Spectral Clustering

img = rgb_crop
img = np.array(img, np.uint8)
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
sc = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0).fit(vectorized)
#Received Memory Error

############################################################################
#Try Hierarchical--didnt work

n_clusters = 2  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                               connectivity=connectivity).fit(vectorized)

###########################################################################
#try MiniBatch KMeans with RGB

img = rgb_crop
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

n_clusters = 5
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6, compute_labels=True).fit_predict(X=vectorized)


label_image = kmeans.reshape((img.shape[0], img.shape[1]))
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
cmap = colors.ListedColormap(coloroptions[0:n_clusters])
ax2.imshow(label_image, cmap=cmap)
ax2.set_title('Mini Batch K Means with RGB and Clusters =  ' + str(n_clusters) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#try with HSV
img = np.array(rgb_crop)
# make sure that values are between 0 and 255, i.e. within 8bit range
img *= 255/img.max() 
# cast to 8bit
img = np.array(img, np.uint8)
hsv_mask = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

vectorized = hsv_mask.reshape((-1,3))
vectorized = np.float32(vectorized)

n_clusters = 5
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6, compute_labels=True).fit_predict(X=vectorized)


label_image = kmeans.reshape((img.shape[0], img.shape[1]))
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
cmap = colors.ListedColormap(coloroptions[0:n_clusters])
ax2.imshow(label_image, cmap=cmap)
ax2.set_title('Mini Batch K Means with HSV and Clusters =  ' + str(n_clusters) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

