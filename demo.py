"""
This script contains a demonstration of the functions and capabilities of this
repository. It will also give an example work flow for a set of images.
Last Updated: Susan Meerdink, May 2020
"""

# Importing Functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg') # Needed to have figures display properly in spyder. 
import flirimageextractor
import utilities as u

## ---GETTING STARTED----------------------------------------------------------
## Setting up some parameters
# You will have change the path of exiftool depending on where it was installed.

dirLoc = 'C:\\Users\\caleb\\MachineLearningLabLocal\\DARPA-Sentinel-Project\\Temperature\\2020-07_mandi\\2020-07-04\\thermal_images\\'
exiftoolpath = 'C:\\Users\\caleb\\Downloads\\exiftool-11.99\\exiftool.exe'

## Load Image using flirimageextractor
filename = dirLoc + 'IR_12737.jpg'
print(filename)
flir = flirimageextractor.FlirImageExtractor(exiftool_path=exiftoolpath)
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

## ---CO-REGISTRATION OF THERMAL & RGB IMAGES----------------------------------
## Check how well thermal and rgb registration is without manually correction
# You can see that the images do not line up and there is an offset even after 
# correcting for offset provided in file header
rgb_lowres, rgb_crop = u.extract_rescale_image(flir)

## Determine manual correction of Thermal and RGB registration 
#offset, pts_temp, pts_rgb = u.manual_img_registration(flir)
#print('X pixel offset is ' + str(offset[0]) + ' and Y pixel offset is ' + str(offset[1]))

## Fix Thermal and RGB registration with manual correction
# You can see with the manually determined offsets that the images are now aligned.
# By doing this we can use the RGB image to classify the material types in the images.
# This is useful if you are interested in one particular part or class type.
#offset = [-73, -73]  # This is the manual offset I got for 2020-07-04
#offset = [-72, -77]  # This is the manual offset I got for 2020-07-06
offset = [-73, -73]
rgb_lowres, rgb_crop = u.extract_rescale_image(flir, offset=offset, plot=1)

## ---SELECTING PIXELS OF INTEREST---------------------------------------------
# With the co-registration of the thermal and RGB images, you can use the RGB 
# pixels to pull specific thermal pixels of interest. For the below example,
# we are interested in pixels that are only plant. There are two methods we are 
# using to cluster pixels and select plant pixels - K Means Clustering and 
# Gaussian Mixture Models. Both of these methods do not require training, but as 
# a result may not be as accurate as a classification method.

# Build a mask of your area of interest
mask = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1]))

# 2020-07-04
mask[0:480, 84:556] = 1
mask[210:480, 0:160] = 0
mask[210:311, 0:216] = 0
mask[0:304, 450:640] = 0
mask[145:304, 436:640] = 0
mask[350:480, 0:196] = 0
mask[34:69, 0:106] = 0
mask[69:82, 0:102] = 0
mask[418:480, 495:640] = 0
mask[413:480, 506:640] = 0
mask[409:480, 510:640] = 0

"""
# 2020-07-06
mask[0:480, 40:568] = 1
mask[0:218, 420:640] = 0
mask[123:480, 0:119] = 0
mask[275:640, 0:250] = 0
mask[406:480, 0:263] = 0
mask[218:380, 522:640] = 0
mask[380:450, 553:640] = 0
mask[29:143, 376:640] = 0
"""
rgb_mask = u.apply_mask_to_rgb(mask, rgb_crop)

# METHOD 2: Gaussian Mixture Models
# We have found this method to be more robust than K-Means Clustering, but 
# still may have issues in particular areas that have similiar values to your 
# class of interest. You may want to develop a classification algorithm for 
# your particular research problem that will yield better results.
# In the following steps, we will use the results from the GMM not KMC.

# Classify using Gaussian Mixture Models the rgb image
rgb_class = u.classify_rgb_GMM(rgb_mask, 3)


# Pull out just the class for plant material
# Vegetation is class 1 2020-07-04
# Vegetation is class 3 2020-07-06
class_mask = u.create_class_mask(classimg=rgb_class, classinterest=[3])

# Save mask of everything except plant pixels
outDir = dirLoc + '..\\class_mask_array'
np.save(outDir, class_mask)

## ---CORRECTING THERMAL IMAGERY-----------------------------------------------
# In order to determine the temperature of an object, it is necessary to also 
# know or assume an emissivity value for that object. Thermal cameras tend to 
# use a default emissivity of 0.95 to generate temperature. However, most objects
# do not have an emissivity of 0.95 which would result in an error in the temperature
# calculation. With the steps above which groups pixels into classes, we can
# correct the emissivity for different classes and update the temperature values.

## Correct temperature imagery for correct emissivity
# This function will ask you what emissivity value you would like to assign to 
# each class. There are tables online for broadband emissivity values. If you
# do not know the emissivity, keep the value at 0.95. 
# In this example, I set the vegetation pixels to 0.98 and everything else to 0.95.
# For 2020-03-02_mandi psent2-18-6 dataset, Class 4 is vegetation (.98). Class 1=2=3=.95
# For 2020-07-04_mandi dataset, Class 1 is vegetation (.98). Class 2=3=.95
emiss_img = u.develop_correct_emissivity(rgb_class)

# Pull out thermal pixels of just plants for single image
print(str(emiss_img.shape))
temp_mask = u.extract_temp(flir, classmask=class_mask, emiss=emiss_img)


## ---PRELIMINARY ANALYSIS-----------------------------------------------------
# This section will pull out all pixels of interest across a timeseries 
# (assuming the camera did not move) and plots the mean, max, and min temperature. 
# It will correct the temperature based on the emissivity value assigned above.

# Pull out thermal pixels of just plants for a set of images
all_temp_mask = u.batch_extract_temp(dirLoc, classmask=class_mask, emiss=emiss_img, exiftoolpath=exiftoolpath)
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(all_temp_mask[:,:,0])
plt.subplot(1,3,2)
plt.imshow(all_temp_mask[:,:,1])
plt.subplot(1,3,3)
plt.imshow(all_temp_mask[:,:,2])
plt.show(block='TRUE')

# Plot timeseries
u.plot_temp_timeseries(all_temp_mask)

## ---EXPORTING CORRECTED THERMAL IMAGES---------------------------------------
# In this example, we are saving each thermal image as a .csv file which can 
# then be imported into your program of choice for further analysis. 
# This is, perhaps obviously, not the only way to exporting the data. 

# First step is to correct the emissivity on all pixels for all images
#all_temp = u.batch_extract_temp(dirLoc, emiss=emiss_img, exiftoolpath=exiftoolpath)

# After correcting temperature
outDir = dirLoc + '..\\CSV_Output\\'
u.output_csv(outDir, all_temp_mask, rgb_class, emiss_img)

## ---END----------------------------------------------------------------------