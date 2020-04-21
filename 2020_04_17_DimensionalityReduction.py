# -*- coding: utf-8 -*-
"""
Dimensionality Reduction

Created on Fri Apr 17 16:27:46 2020

@author: sofiavega
"""

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
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
from sklearn.metrics import mean_squared_error
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd

##Testing the 2019-02-04 Data Set
## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
filename = 'C:\\Users\\sofiavega\\FLIR_thermal_tools\\FLIR0346.jpg'
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

%matplotlib inline


plant = np.append(np.append(rgb_crop[270:290,140:147].flatten() , rgb_crop[240:270,105:110].flatten()), rgb_crop[450:480,570:575].flatten())
 
plant = np.append(np.append(rgb_crop[270:290,140:147] , rgb_crop[240:270,105:110]), rgb_crop[450:480,570:575])
 

#plant_label = np.zeros((plant.shape[0], plant.shape[1])) + 1
#plant_label = np.zeros((r.shape[0]))  


#split RGB
r,g,b = cv2.split(rgb_crop)
r = np.append(np.append(r[270:290,140:147] , r[240:270,105:110]), r[450:480,570:575])
g = np.append(np.append(g[270:290,140:147] , g[240:270,105:110]), g[450:480,570:575])
b = np.append(np.append(b[270:290,140:147] , b[240:270,105:110]), b[450:480,570:575])
 
r = r.flatten().reshape(440,1)
g = g.flatten().reshape(440,1)
b = b.flatten().reshape(440,1)

#Conver plant to HSV
# make sure that values are between 0 and 255, i.e. within 8bit range
plant_hsv = rgb_crop
plant_hsv *= 255/plant_hsv.max() 
# cast to 8bit
plant_hsv = np.array(plant_hsv, np.uint8)
plant_hsv = cv2.cvtColor(plant_hsv, cv2.COLOR_RGB2HSV)

##Forward Selection for hsv
h,s,v = cv2.split(plant_hsv)
h = np.append(np.append(h[270:290,140:147] , h[240:270,105:110]), h[450:480,570:575])
s = np.append(np.append(s[270:290,140:147] , s[240:270,105:110]), s[450:480,570:575])
v = np.append(np.append(v[270:290,140:147] , v[240:270,105:110]), v[450:480,570:575])
 
h = h.flatten().reshape(440,1)
s = s.flatten().reshape(440,1)
v = v.flatten().reshape(440,1)

#Conver plant to LAB
# make sure that values are between 0 and 255, i.e. within 8bit range
plant_lab = rgb_crop
plant_lab *= 255/plant_lab.max() 
# cast to 8bit
plant_lab = np.array(plant_lab, np.uint8)
plant_lab = cv2.cvtColor(plant_lab, cv2.COLOR_RGB2LAB)

##Forward Selection for hsv
l,a,b_lab = cv2.split(plant_lab)
l = np.append(np.append(l[270:290,140:147] ,l[240:270,105:110]), l[450:480,570:575])
a = np.append(np.append(a[270:290,140:147] , a[240:270,105:110]), a[450:480,570:575])
b_lab = np.append(np.append(b_lab[270:290,140:147] , b_lab[240:270,105:110]), b_lab[450:480,570:575])
 
l = l.flatten().reshape(440,1)
a = a.flatten().reshape(440,1)
b_lab = b_lab.flatten().reshape(440,1)

#Make feature matrix
feat_mat = np.concatenate((r,g,b,h,s,v,l,a,b_lab), axis = 1)

feature_names = ('R', 'G', 'B (RGB)', 'H', 'S', 'V', 'L', 'A', 'B (LAB)' )

feat_mat = pd.DataFrame(feat_mat,columns=feature_names)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=3)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#LAB(LAB)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=2)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#AB(LAB)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=1)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#B(LAB)

########################################################
###############################################################################
filename = 'C:\\Users\\sofiavega\\FLIR_thermal_tools\\FLIR0138.jpg'
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

rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

%matplotlib inline


#extract plant

plant = np.append(np.append(rgb_crop[190:240,89:98].flatten(), rgb_crop[400:433,525:529].flatten()), rgb_crop[125:134,590:613].flatten())


#plant_label = np.zeros((plant.shape[0], plant.shape[1])) + 1
#plant_label = np.zeros((r.shape[0]))  


#split RGB
r,g,b = cv2.split(rgb_crop)
r = np.append(np.append(r[190:240,89:98] , r[125:134,590:613]), r[400:433,525:529])
g = np.append(np.append(g[190:240,89:98] , g[125:134,590:613]), g[400:433,525:529])
b = np.append(np.append(b[190:240,89:98] , b[125:134,590:613]), b[400:433,525:529])
 
r = r.flatten().reshape(r.shape[0],1)
g = g.flatten().reshape(r.shape[0],1)
b = b.flatten().reshape(r.shape[0],1)

#Conver plant to HSV
# make sure that values are between 0 and 255, i.e. within 8bit range
plant_hsv = rgb_crop
plant_hsv *= 255/plant_hsv.max() 
# cast to 8bit
plant_hsv = np.array(plant_hsv, np.uint8)
plant_hsv = cv2.cvtColor(plant_hsv, cv2.COLOR_RGB2HSV)

##Forward Selection for hsv
h,s,v = cv2.split(plant_hsv)
h = np.append(np.append(h[190:240,89:98] , h[125:134,590:613]), h[400:433,525:529])
s = np.append(np.append(s[190:240,89:98] , s[125:134,590:613]), s[400:433,525:529])
v = np.append(np.append(v[190:240,89:98] , v[125:134,590:613]), v[400:433,525:529])
 
h = h.flatten().reshape(r.shape[0],1)
s = s.flatten().reshape(r.shape[0],1)
v = v.flatten().reshape(r.shape[0],1)

#Conver plant to LAB
# make sure that values are between 0 and 255, i.e. within 8bit range
plant_lab = rgb_crop
plant_lab *= 255/plant_lab.max() 
# cast to 8bit
plant_lab = np.array(plant_lab, np.uint8)
plant_lab = cv2.cvtColor(plant_lab, cv2.COLOR_RGB2LAB)

##Forward Selection for hsv
l,a,b_lab = cv2.split(plant_lab)
l = np.append(np.append(l[190:240,89:98] ,l[125:134,590:613]), l[400:433,525:529])
a = np.append(np.append(a[190:240,89:98] , a[125:134,590:613]), a[400:433,525:529])
b_lab = np.append(np.append(b_lab[190:240,89:98] , b_lab[125:134,590:613]), b_lab[400:433,525:529])
 
l = l.flatten().reshape(r.shape[0],1)
a = a.flatten().reshape(r.shape[0],1)
b_lab = b_lab.flatten().reshape(r.shape[0],1)

#Make feature matrix
feat_mat = np.concatenate((r,g,b,h,s,v,l,a,b_lab), axis = 1)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=3)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#LAB(LAB)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=2)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#AB(LAB)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=1)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#B(LAB)

########################################################################

filename = 'C:\\Users\\sofiavega\\FLIR_thermal_tools\\FLIR5424.jpg'
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

rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

%matplotlib inline

r,g,b = cv2.split(rgb_crop)
#extract plant

plant = np.append(np.append(rgb_crop[225:275,193:208].flatten(), rgb_crop[50:80,210:217].flatten()), rgb_crop[400:480,375:380].flatten())


#plant_label = np.zeros((plant.shape[0], plant.shape[1])) + 1
#plant_label = np.zeros((r.shape[0]))  


#split RGB
r,g,b = cv2.split(rgb_crop)
r = np.append(np.append(r[225:275,193:208] , r[50:80,210:217]), r[400:480,375:380])
g = np.append(np.append(g[225:275,193:208] , g[50:80,210:217]), g[400:480,375:380])
b = np.append(np.append(b[225:275,193:208] , b[50:80,210:217]), b[400:480,375:380])
 
r = r.flatten().reshape(r.shape[0],1)
g = g.flatten().reshape(r.shape[0],1)
b = b.flatten().reshape(r.shape[0],1)

#Conver plant to HSV
# make sure that values are between 0 and 255, i.e. within 8bit range
plant_hsv = rgb_crop
plant_hsv *= 255/plant_hsv.max() 
# cast to 8bit
plant_hsv = np.array(plant_hsv, np.uint8)
plant_hsv = cv2.cvtColor(plant_hsv, cv2.COLOR_RGB2HSV)

##Forward Selection for hsv
h,s,v = cv2.split(plant_hsv)
h = np.append(np.append(h[225:275,193:208] , h[50:80,210:217]), h[400:480,375:380])
s = np.append(np.append(s[225:275,193:208] , s[50:80,210:217]), s[400:480,375:380])
v = np.append(np.append(v[225:275,193:208] , v[50:80,210:217]), v[400:480,375:380])
 
h = h.flatten().reshape(r.shape[0],1)
s = s.flatten().reshape(r.shape[0],1)
v = v.flatten().reshape(r.shape[0],1)

#Conver plant to LAB
# make sure that values are between 0 and 255, i.e. within 8bit range
plant_lab = rgb_crop
plant_lab *= 255/plant_lab.max() 
# cast to 8bit
plant_lab = np.array(plant_lab, np.uint8)
plant_lab = cv2.cvtColor(plant_lab, cv2.COLOR_RGB2LAB)

##Forward Selection for hsv
l,a,b_lab = cv2.split(plant_lab)
l = np.append(np.append(l[225:275,193:208] ,l[50:80,210:217]), l[400:480,375:380])
a = np.append(np.append(a[225:275,193:208] , a[50:80,210:217]), a[400:480,375:380])
b_lab = np.append(np.append(b_lab[225:275,193:208] , b_lab[50:80,210:217]), b_lab[400:480,375:380])
 
l = l.flatten().reshape(r.shape[0],1)
a = a.flatten().reshape(r.shape[0],1)
b_lab = b_lab.flatten().reshape(r.shape[0],1)

#Make feature matrix
feat_mat = np.concatenate((r,g,b,h,s,v,l,a,b_lab), axis = 1)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=3)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#LAB(LAB)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=2)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#AB(LAB)

# Create an SelectKBest object to select features with two best ANOVA F-Values
fvalue_selector = SelectKBest(f_classif, k=1)

# Apply the SelectKBest object to the features and target
kbest = fvalue_selector.fit_transform(feat_mat, plant_label)
kbest.shape
#B(LAB)