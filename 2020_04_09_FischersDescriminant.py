# -*- coding: utf-8 -*-
"""
Calculating Fischer's Descriminant for RGB, HSV, LAB'
Created on Thu Apr  9 17:53:40 2020

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


##Testing the 2019-02-04 Data Set
## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
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

## Check how well thermal and rgb registration is without manually correction
# You can see that the images do not line up and there is an offset even after 
# correcting for offset provided in file header
#The rgb cropped seems pretty close
rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

%matplotlib inline


#extract tape
tape = rgb_crop[20:27,150:170]
plt.imshow(tape) #extract tape

#extract plant
plant = rgb_crop[270:290,140:147]
plt.imshow(plant)

#------------------------------------RGB------------------------------------

t_r, t_g, t_b = cv2.split(tape) #split the rgb pixels


#calculate mean vector per class
t_r_mean = np.mean(np.mean(t_r,axis=1))

t_g_mean = np.mean(np.mean(t_g,axis=1))

t_b_mean = np.mean(np.mean(t_b,axis=1))




p_r, p_g, p_b = cv2.split(plant) #split the rgb pixels


#calculate mean vector per class
p_r_mean = np.mean(np.mean(p_r,axis=1))

p_g_mean = np.mean(np.mean(p_g,axis=1))

p_b_mean = np.mean(np.mean(p_b,axis=1))

#Calculate Scatter within for each group
S_t_r = np.sum(np.dot((t_r-t_r_mean), (t_r-t_r_mean).T))
S_t_g = np.sum(np.dot((t_g-t_g_mean), (t_g-t_g_mean).T))
S_t_b = np.sum(np.dot((t_b-t_b_mean), (t_b-t_b_mean).T))

S_p_r = np.sum(np.dot((p_r-p_r_mean), (p_r-p_r_mean).T))
S_p_g = np.sum(np.dot((p_g-p_g_mean), (p_g-p_g_mean).T))
S_p_b = np.sum(np.dot((p_b-p_b_mean), (p_b-p_b_mean).T))

Sw_r = S_t_r + S_p_r
Sw_g = S_t_g + S_p_g
Sw_b = S_t_b + S_p_b

#Calculate Scatter between for each group

Sb_r = (t_r_mean-p_r_mean)**2
Sb_g = (t_g_mean-p_g_mean)**2
Sb_b = (t_b_mean-p_b_mean)**2



#Calculate Fischer's Determinant Value for each group
F_r = Sb_r/Sw_r #0.00037599006415696315 
F_g = Sb_g/Sw_g #0.0002858755861437443
F_b = Sb_b/Sw_b #0.0009716780115566616

#------------------------------------HSV------------------------------------

#Change datatype
img1 = tape

# make sure that values are between 0 and 255, i.e. within 8bit range
img1 *= 255/img1.max() 
# cast to 8bit
img1 = np.array(img1, np.uint8)

#Plot HSV Pixels
tape_hsv = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)

#Change datatype
img2 = plant

# make sure that values are between 0 and 255, i.e. within 8bit range
img2 *= 255/img2.max() 
# cast to 8bit
img2 = np.array(img2, np.uint8)

#Plot HSV Pixels
plant_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

t_h, t_s, t_v = cv2.split(tape_hsv) #split the rgb pixels


#calculate mean vector per class
t_h_mean = np.mean(np.mean(t_h,axis=1))


t_s_mean = np.mean(np.mean(t_s,axis=1))


t_v_mean = np.mean(np.mean(t_v,axis=1))



p_h, p_s, p_v = cv2.split(plant_hsv) #split the rgb pixels

#p_r = StandardScaler().fit_transform(p_r.reshape(-1,1)) 


#calculate mean vector per class
p_h_mean = np.mean(np.mean(p_h,axis=1))

p_s_mean = np.mean(np.mean(p_s,axis=1))

p_v_mean = np.mean(np.mean(p_v,axis=1))

#Calculate Scatter within for each group
S_t_h = np.sum(np.dot((t_h-t_h_mean), (t_h-t_h_mean).T))
S_t_s = np.sum(np.dot((t_s-t_s_mean), (t_s-t_s_mean).T))
S_t_v = np.sum(np.dot((t_v-t_v_mean), (t_v-t_v_mean).T))

S_p_h = np.sum(np.dot((p_h-p_h_mean), (p_h-p_h_mean).T))
S_p_s = np.sum(np.dot((p_s-p_s_mean), (p_s-p_s_mean).T))
S_p_v = np.sum(np.dot((p_v-p_v_mean), (p_v-p_v_mean).T))

Sw_h = S_t_h + S_p_h
Sw_s = S_t_s + S_p_s
Sw_v = S_t_v + S_p_v

#Calculate Scatter between for each group

Sb_h = (t_h_mean-p_h_mean)**2
Sb_s = (t_s_mean-p_s_mean)**2
Sb_v = (t_v_mean-p_v_mean)**2

#Calculate Fischer's Determinant Value for each group
F_h = Sb_h/Sw_h #0.0009786556741161073
F_s = Sb_s/Sw_s #0.001727189676829965
F_v = Sb_v/Sw_v #0.03523341708942245

#------------------------------------LAB------------------------------------

#Change datatype
img1 = tape

# make sure that values are between 0 and 255, i.e. within 8bit range
img1 *= 255/img1.max() 
# cast to 8bit
img1 = np.array(img1, np.uint8)

#Plot HSV Pixels
tape_lab = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)

#Change datatype
img2 = plant

# make sure that values are between 0 and 255, i.e. within 8bit range
img2 *= 255/img2.max() 
# cast to 8bit
img2 = np.array(img2, np.uint8)

#Plot HSV Pixels
plant_lab = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

t_l, t_a, t_b = cv2.split(tape_lab) #split the rgb pixels


#calculate mean vector per class
t_l_mean = np.mean(np.mean(t_l,axis=1))


t_a_mean = np.mean(np.mean(t_a,axis=1))


t_b_mean = np.mean(np.mean(t_b,axis=1))



p_l, p_a, p_b = cv2.split(plant_lab) #split the rgb pixels

#p_r = StandardScaler().fit_transform(p_r.reshape(-1,1)) 


#calculate mean vector per class
p_l_mean = np.mean(np.mean(p_l,axis=1))

p_a_mean = np.mean(np.mean(p_a,axis=1))

p_b_mean = np.mean(np.mean(p_b,axis=1))

#Calculate Scatter within for each group
S_t_l = np.sum(np.dot((t_l-t_l_mean), (t_l-t_l_mean).T))
S_t_a = np.sum(np.dot((t_a-t_a_mean), (t_a-t_a_mean).T))
S_t_b = np.sum(np.dot((t_b-t_b_mean), (t_b-t_b_mean).T))

S_p_l = np.sum(np.dot((p_l-p_l_mean), (p_l-p_l_mean).T))
S_p_a = np.sum(np.dot((p_a-p_a_mean), (p_a-p_a_mean).T))
S_p_b = np.sum(np.dot((p_b-p_b_mean), (p_b-p_b_mean).T))

Sw_l = S_t_l + S_p_l
Sw_a = S_t_a + S_p_a
Sw_b = S_t_b + S_p_b

#Calculate Scatter between for each group

Sb_l = (t_l_mean-p_l_mean)**2
Sb_a = (t_a_mean-p_a_mean)**2
Sb_b = (t_b_mean-p_b_mean)**2

#Calculate Fischer's Determinant Value for each group
F_l = Sb_l/Sw_l #0.02594869684245443
F_a = Sb_a/Sw_a #0.0001565557729940892
F_b = Sb_b/Sw_b #0.0007937717770035445




