# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:02:02 2019

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
from skimage.transform import rescale

## Load Image using flirimageextractor
# Note: I had to change the path of my exiftool which you may need to also change.
filename = 'C:\\Users\\susanmeerdink\\Dropbox (UFL)\\DARPA_SENTINEL\\Data\\Mandi_Datasets\\FLIR8944_1107.jpg'
flir = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Users\\susanmeerdink\\.utilities\\exiftool.exe")
flir.process_image(filename, RGB=True)
        
img_rescale, img_crop = u.extract_rescale_image(flir)

offset, pts_therm, pts_rgb = u.manual_img_registration(flir)

img_rescale, img_crop = u.extract_rescale_image(flir, offset=offset)

#  Build a mask of your area of interest 
mask = np.zeros((img_crop.shape[0], img_crop.shape[1]))
mask[130:410,125:450] = 1
mask[130:200,435:450] = 0
img_mask = u.apply_mask_to_rgb(mask, img_crop)

# Classify using K-Means the newly masked rgb image
rgb_class, rgb_qcolor = u.classify_rgb(img_mask, 4)

# Pull out just the class for plant material
class_mask = u.create_class_mask(rgb_class, 3)

# %%  Correct temperature imagery for correct emissivity
emiss_img = u.develop_correct_emissivity(rgb_class)

# Pull out thermal pixels of just plants for single image
temp_mask = u.extract_temp_for_class(flir, class_mask, emiss_img)
