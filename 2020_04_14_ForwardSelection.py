# -*- coding: utf-8 -*-
"""
Forward Selection by hand to see which features in R,G,B,H,S,V,L,A,B give the 
best GMM classification.

This was done by hand. I first plotted all of the features individually through the GMM classification
then I picked the best one and added one other features to theclassification until I found the two
best features. Then I added a third feature to the two best features until I found the est three features.
Created on Mon Apr 13 10:28:15 2020

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.cluster import spectral_clustering


filename = 'C:\\Users\\sofiavega\\FLIR_thermal_tools\\FLIR0346.jpg'
flir = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Users\\sofiavega\\AppData\\Local\\Temp\\Temp1_exiftool-11.91.zip\\exiftool(-k)")
flir.process_image(filename, RGB=True)

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

rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

%matplotlib inline

##Forward Selection for rgb
r,g,b = cv2.split(rgb_crop)

#r------------------------------------------------------
#Prepare Image
img = r
image = rgb_crop
num_class = 5
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with R= Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')
   
#g------------------------------------------------------
#Prepare Image
img = g
image = rgb_crop
num_class = 4
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with G = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE') 

#b------------------------------------------------------
#Prepare Image
img = b
image = rgb_crop
num_class = 2
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with B = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#R and G-----------------------------------------------------------
rg = np.dstack((r,g)) #append r and g since those were the best

img = rg
image = rgb_crop
num_class = 6
vectorized = img.reshape((-1,2))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with RG = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#R and G-----------------------------------------------------------
rb = np.dstack((r,b)) #append r and g since those were the best

img = rb
image = rgb_crop
num_class = 4
vectorized = img.reshape((-1,2))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with RB = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

##############################################################################
 #Transform to HSV
# make sure that values are between 0 and 255, i.e. within 8bit range
img = rgb_crop
img *= 255/img.max() 
# cast to 8bit
img = np.array(img, np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

##Forward Selection for hsv
h,s,v = cv2.split(img)

#h------------------------------------------------------
#Prepare Image
img = h
image = rgb_crop
num_class = 4
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with H = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')
   
#s------------------------------------------------------
#Prepare Image
img = s
image = rgb_crop
num_class = 2
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with S = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE') 

#v------------------------------------------------------
#Prepare Image
img = v
image = rgb_crop
num_class = 4
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with V = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#S and V-----------------------------------------------------------
#BEST Two Variable combination
sv = np.dstack((s,v)) 
img = sv
image = rgb_crop
num_class = 8
vectorized = img.reshape((-1,2))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with SV = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')


##############################################################################
 #Transform to LAb
# make sure that values are between 0 and 255, i.e. within 8bit range
img = rgb_crop
img *= 255/img.max() 
# cast to 8bit
img = np.array(img, np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

##Forward Selection for hsv
l,a,b_lab = cv2.split(img)

#l------------------------------------------------------
#Prepare Image
img = l
image = rgb_crop
num_class = 6
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with L = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')
   
#a------------------------------------------------------
#Prepare Image
img = a
image = rgb_crop
num_class = 2
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with A = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE') 

#b------------------------------------------------------
#Prepare Image
img = b_lab
image = rgb_crop
num_class = 4
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with B(LAB) = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#L and A-----------------------------------------------------------
la = np.dstack((l,a)) #append l and a since those were the best

img = la
image = rgb_crop
num_class = 6
vectorized = img.reshape((-1,2))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with LA = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#L and V-----------------------------------------------------------
# SECOND BEST two variable combination
lv = np.dstack((l,v)) 

img = lv
image = rgb_crop
num_class = 10
vectorized = img.reshape((-1,2))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with LV = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#L and V and S-----------------------------------------------------------
#BEST three variable combination
lvs = np.dstack((l,v,s)) #append l,s, and v since those were the best

img = lvs
image = rgb_crop
num_class = 3
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with LVS = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#L and V and and S and B(rgb)-----------------------------------------------------------
lvsb = np.dstack((l,v,s,b)) 

img = lvsb
image = rgb_crop
num_class = 4
vectorized = img.reshape((-1,4))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with LVSB = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

#g and v-------------------------------------------------------------------
vg = np.dstack((v,g)) #append g and v 

img = vg
image = rgb_crop
num_class = 3
vectorized = img.reshape((-1,2))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with VG = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')

# land g and S-----------------------------------------------------------
#Try our other three variable combinations here, none are better than LVS
gvs = np.dstack((g,v,s)) 

img = vgs
image = rgb_crop
num_class = 5
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=num_class).fit(vectorized)
labels = gmm.predict(vectorized)


# Labeled class image
label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
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
ax2.set_title('GMM with SVG = Classes = ' + str(num_class) )
ax2.set_xticks([]) 
ax2.set_yticks([])
fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01, wspace=0.05)
plt.show(block='TRUE')
