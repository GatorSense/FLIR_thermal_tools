# -*- coding: utf-8 -*-
"""

Calculate ROC Curves and RMSE for Training Sets from multiple data sets
Created on Thu Apr 16 23:12:02 2020

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
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import roc_curve, auc

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


#extract plant
plant = np.append(np.append(rgb_crop[270:290,140:147].flatten() , rgb_crop[240:270,105:110].flatten()), rgb_crop[450:480,570:575].flatten())
 


plant_label = np.zeros((plant.shape[0])) + 1


labels = GMM_rgb(image=rgb_crop,num_class=3,hsv = 0, plot=1)

pred = labels[270:290,140:147]
pred = plant_label.flatten()

fpr, tpr, thresholds = roc_curve(plant_label, pred)
tpr = np.array([1,1])
fpr = np.array([0,1])
roc_auc = auc(fpr, tpr)




plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for RGB GMM on plant only')
plt.legend(loc="lower right")
plt.show()

#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0

#-----------------------------------------------------------------
r,g,b = cv2.split(rgb_crop)
r = r.flatten().reshape(140,1)
g = g.flatten().reshape(140,1)
b = b.flatten().reshape(140,1)
#--------r
img = r
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.21320071635561044

plant_label = label_binarize(plant_label, classes=[0, 1, 2])

n_classes = plant_label.shape[1]

y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(plant_label[:, i], pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
    
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for R GMM on plant only')
plt.legend(loc="lower right")
plt.show()

#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0

#------------------------------------g------------------------
img = g
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=4).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0

#------------------------------------b------------------------
img = b
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) +2


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.1716344450071758

##------------------------------Transform to HSV-----------------
plant_hsv = rgb_crop
plant_hsv *= 255/plant_hsv.max() 
# cast to 8bit
plant_hsv = np.array(plant_hsv, np.uint8)
plant_hsv = cv2.cvtColor(plant_hsv, cv2.COLOR_RGB2HSV)


h,s,v = cv2.split(plant_hsv)

##---------------------------------h
img = h
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=4).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) 


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.6004260796311827

##---------------------------------s
img = s
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) +2



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.3751033019046572

##---------------------------------v
img = v
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) 



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0


##------------------------------Transform to LAB-----------------
plant_lab = rgb_crop
plant_lab *= 255/plant_lab.max() 
# cast to 8bit
plant_lab = np.array(plant_lab, np.uint8)
plant_lab = cv2.cvtColor(plant_lab, cv2.COLOR_RGB2LAB)


l,a,b_lab = cv2.split(plant_lab)

##---------------------------------l
img = l
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) 


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0

##---------------------------------a
img = a
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=5).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) +3



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.9378408471299672

##---------------------------------v
img = b_lab
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=4).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[270:290,140:147].flatten(), label_image[240:270,105:110].flatten()), label_image[450:480,570:575].flatten())


plant_label = np.zeros((pred.shape[0])) +1



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.7307381504590882
############################################################################

##Testing the 2019-01-09 Data Set
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

rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

%matplotlib inline

r,g,b = cv2.split(rgb_crop)

#--------r
img = r
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.21320071635561044



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.3433230584063064

#------------------------------------g------------------------
img = g
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=4).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +2


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.07120188545089044

#------------------------------------b------------------------
img = b
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +2


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.6328562007669485

##------------------------------Transform to HSV-----------------
plant_hsv = rgb_crop
plant_hsv *= 255/plant_hsv.max() 
# cast to 8bit
plant_hsv = np.array(plant_hsv, np.uint8)
plant_hsv = cv2.cvtColor(plant_hsv, cv2.COLOR_RGB2HSV)


h,s,v = cv2.split(plant_hsv)

##---------------------------------h
img = h
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.7693428796142727

##---------------------------------s
img = s
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +1



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.3916103699798974

##---------------------------------v
img = v
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +1



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.03560094272544522


##------------------------------Transform to LAB-----------------
plant_lab = rgb_crop
plant_lab *= 255/plant_lab.max() 
# cast to 8bit
plant_lab = np.array(plant_lab, np.uint8)
plant_lab = cv2.cvtColor(plant_lab, cv2.COLOR_RGB2LAB)


l,a,b_lab = cv2.split(plant_lab)

##---------------------------------l
img = l
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +2


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.27576371656986687

##---------------------------------a
img = a
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=5).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) +2



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.273200412441345

##---------------------------------b
img = b_lab
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=5).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[190:240,89:98].flatten(), label_image[400:433,525:529].flatten()), label_image[125:134,590:613].flatten())


plant_label = np.zeros((pred.shape[0])) 



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.748437651313972

#########################################################################
##Testing the 2019-05-05 data set
## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
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

## Check how well thermal and rgb registration is without manually correction
# You can see that the images do not line up and there is an offset even after 
# correcting for offset provided in file header
#The rgb cropped seems pretty close
rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

%matplotlib inline

r,g,b = cv2.split(rgb_crop)

#--------r
img = r
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +2


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.5872418080965205





#------------------------------------g------------------------
img = g
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=4).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.16042223697993696

#------------------------------------b------------------------
img = b
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) 


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.121317502394603

##------------------------------Transform to HSV-----------------
plant_hsv = rgb_crop
plant_hsv *= 255/plant_hsv.max() 
# cast to 8bit
plant_hsv = np.array(plant_hsv, np.uint8)
plant_hsv = cv2.cvtColor(plant_hsv, cv2.COLOR_RGB2HSV)


h,s,v = cv2.split(plant_hsv)

##---------------------------------h
img = h
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.5382433300589439

##---------------------------------s
img = s
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=4).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +3



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #2.2711424955942587

##---------------------------------v
img = v
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +1



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.20109991663496093


##------------------------------Transform to LAB-----------------
plant_lab = rgb_crop
plant_lab *= 255/plant_lab.max() 
# cast to 8bit
plant_lab = np.array(plant_lab, np.uint8)
plant_lab = cv2.cvtColor(plant_lab, cv2.COLOR_RGB2LAB)


l,a,b_lab = cv2.split(plant_lab)

##---------------------------------l
img = l
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +1


#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.5109103754717709

##---------------------------------a
img = a
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=5).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +2



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #1.2340154446549925

##---------------------------------v
img = b_lab
image = rgb_crop
vectorized = img.reshape((-1,1))
vectorized = np.float32(vectorized)

#Gaussian Mixture Modeling
gmm = GaussianMixture(n_components=3).fit(vectorized)
labels = gmm.predict(vectorized)

label_image = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(label_image)
pred = np.append(np.append(label_image[225:275,193:208].flatten(), label_image[50:80,210:217].flatten()), label_image[400:480,375:380].flatten())


plant_label = np.zeros((pred.shape[0])) +1



#calculate mse
mse = mean_squared_error(plant_label,pred)
rmse = math.sqrt(mse) #0.7779119865534464
