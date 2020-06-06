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

dirLoc = 'C:\\Users\\caleb\\MachineLearningLabLocal\\DARPA-Sentinel-Project\\Temperature\\2020-03-02_mandi\\psent2-18-6\\test_images\\'
#dirLoc = 'C:\\Users\\caleb\\MachineLearningLabLocal\\FLIR_thermal_tools\\Test_Images\\'
exiftoolpath = 'C:\\Users\\caleb\\Downloads\\exiftool-11.99\\exiftool.exe'

## Load Image using flirimageextractor
filename = dirLoc + 'IR_10379.jpg'
#filename = dirLoc + 'IR_56020.jpg'
print (filename)
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
offset, pts_temp, pts_rgb = u.manual_img_registration(flir)
print('X pixel offset is ' + str(offset[0]) + ' and Y pixel offset is ' + str(offset[1]))

## Fix Thermal and RGB registration with manual correction
# You can see with the manually determined offsets that the images are now aligned.
# By doing this we can use the RGB image to classify the material types in the images.
# This is useful if you are interested in one particular part or class type.
#offset = [-155, -70]  # This i the manual offset I got when running the demo images.
offset = [offset[0], offset[1]]
rgb_lowres, rgb_crop = u.extract_rescale_image(flir, offset=offset, plot=1)

## ---SELECTING PIXELS OF INTEREST---------------------------------------------
# With the co-registration of the thermal and RGB images, you can use the RGB 
# pixels to pull specific thermal pixels of interest. For the below example,
# we are interested in pixels that are only plant. There are two methods we are 
# using to cluster pixels and select plant pixels - K Means Clustering and 
# Gaussian Mixture Models. Both of these methods do not require training, but as 
# a result may not be as accurate as a classification method.

# METHOD 1: K Means Clustering
# We have found this method to be very sensitive to background pixels. 
# So the first step we will build a mask to avoid pixels that will be 
# confused with plant pixels.
# Build a mask of your area of interest 
mask = np.zeros((rgb_crop.shape[0], rgb_crop.shape[1]))
mask[0:450,100:400] = 1
rgb_mask = u.apply_mask_to_rgb(mask, rgb_crop)

# Classify using K-Means Clustering the newly masked rgb image
rgb_class, rgb_qcolor = u.classify_rgb_KMC(rgb_mask, 3)

# Pull out just the class for plant material 
class_mask_KMC = u.create_class_mask(rgb_class, 3)

# METHOD 2: Gaussian Mixture Models
# We have found this method to be more robust than K-Means Clustering, but 
# still may have issues in particular areas that have similiar values to your 
# class of interest. You may want to develop a classification algorithm for 
# your particular research problem that will yield better results.
# In the following steps, we will use the results from the GMM not KMC.

# Classify using Gaussian Mixture Models the rgb image
rgb_class = u.classify_rgb_GMM(rgb_mask, 6)

# Pull out just the class for plant material
class_mask = u.create_class_mask(rgb_class, [3,6])

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
emiss_img = u.develop_correct_emissivity(rgb_class)

# Pull out thermal pixels of just plants for single image
temp_mask = u.extract_temp(flir, class_mask, emiss_img)

## ---PRELIMINARY ANALYSIS-----------------------------------------------------
# This section will pull out all pixels of interest across a timeseries 
# (assuming the camera did not move) and plots the mean, max, and min temperature. 
# It will correct the temperature based on the emissivity value assigned above.

# Pull out thermal pixels of just plants for a set of images
all_temp_mask = u.batch_extract_temp(dirLoc, class_mask, emiss_img, exiftoolpath=exiftoolpath)
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
all_temp = u.batch_extract_temp(dirLoc,emiss=emiss_img, exiftoolpath=exiftoolpath)

# After correcting temperature
outDir = dirLoc + '..\\CSV_Output\\'
u.output_csv(outDir, all_temp, rgb_class, emiss_img)

## ---END----------------------------------------------------------------------