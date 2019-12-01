# Importing Functions
import flirimageextractor
from matplotlib import pyplot as plt
import numpy as np
import subprocess

def save_thermal_csv(flirobj, filename):
    """
    Function that saves the numpy array as a .csv
    
    INPUTS:
    1) flirobj: the flirimageextractor object.
    2) filename: a string containing the location of the output csv file. 
    
    OUTPUTS:
    Saves a csv of the thermal image where each value is a pixel in the thermal image. 
    """
    data = flirobj.get_thermal_np()
    np.savetxt(filename, data, delimiter=',')

def extract_coarse_image(flirobj, offset=[0]):
    """
    Function that creates the coarse RGB image that matches the resolution of the thermal image.
    
    INPUTS:
    1) flirobj: the flirimageextractor object.
    
    OUTPUTS:
    1) crop: a 3D numpy arary of RGB image that matches resolution and field of view of thermal image.
    """
    # Get RGB Image
    visual = flirobj.rgb_image_np
    highres_ht = visual.shape[0]
    highres_wd = visual.shape[1]
    
    # Getting Values for Offset
    if len(offset) < 2:
        offsetx = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetX", "-b", flirobj.flir_img_filename])) 
        offsety = int(subprocess.check_output([flirobj.exiftool_path, "-OffsetY", "-b", flirobj.flir_img_filename])) 
    else:
        offsetx = offset[0]
        offsety = offset[1]
    pipx1 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPX1", "-b", flirobj.flir_img_filename])) 
    pipx2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPX2", "-b", flirobj.flir_img_filename])) # Width
    pipy1 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPY1", "-b", flirobj.flir_img_filename])) 
    pipy2 = int(subprocess.check_output([flirobj.exiftool_path, "-PiPY2", "-b", flirobj.flir_img_filename])) # Height
    real2ir = float(subprocess.check_output([flirobj.exiftool_path, "-Real2IR", "-b", flirobj.flir_img_filename]))
    
    # Set up Arrays
    height_range = np.arange(0,highres_ht,real2ir).astype(int)
    width_range = np.arange(0,highres_wd,real2ir).astype(int)
    htv, wdv = np.meshgrid(height_range,width_range)
    
    # Assigning low resolution data
    lowres = np.swapaxes(visual[htv, wdv,  :], 0, 1)
    
    # Getting additional variables
#    center_height = lowres.shape[0]/2
#    center_width = lowres.shape[1]/2
#    h1 = center_height - ((pipy2 - pipy1)/2) + offsety 
#    h2 = center_height + ((pipy2 - pipy1)/2) + offsety 
#    w1 = center_width - ((pipx2 - pipx1)/2) + offsetx
#    w2 = center_width + ((pipx2 - pipx1)/2) + offsetx
#    height_range = np.arange(h1,h2).astype(int)
#    width_range = np.arange(w1,w2).astype(int)
    height_range = np.arange(-offsety,-offsety+pipy2).astype(int)
    width_range = np.arange(-offsetx,-offsetx+pipx2).astype(int)

    # Cropping low resolution data
    xv, yv = np.meshgrid(height_range,width_range)
    crop = np.swapaxes(lowres[xv, yv, :],0,1)
    
    return lowres, crop

def manual_img_registration(filename):
    """
    Function that displays the thermal and RGB image so that similar locations 
    can be selected in both images. 
    INPUTS:
        1) filename: a string with the thermal image location. 
    OUTPUTS:
        1) offset: a numpy array with [x pixel offset, y pixel offset] between thermal and rgb image
        2) pts_therm: a numpy array containing the image registration points for the thermal image. 
        3) pts_rgb: a numpy array containing the coordinates of RGB image matching the thermal image.
    """
    # Getting Images
    flir = flirimageextractor.FlirImageExtractor(exiftool_path="C:\\Users\\susanmeerdink\\.utilities\\exiftool.exe")
    flir.process_image(filename, RGB=True)
    therm = flir.get_thermal_np()
    rgb, junk = extract_coarse_image(flir)
    
    # Plot Images
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(therm)
    ax1.set_title('Thermal')
    ax1.text(0,-100,'Collect points matching features between images. Select location on thermal then RGB image.')
    ax1.text(0,-75,'Right click adds a point. Left click removes most recently added point. Middle click stops point collection.')
    ax1.text(0,-50,'Zoom/Pan add a point, but you can remove with left click. Or use back arrow to get back to original view.')
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(rgb)
    ax2.set_title('RGB')
    fig.subplots_adjust(left=0.05, top = 0.8, bottom=0.01,right=0.95)
    
    # Getting points
    pts = np.asarray(fig.ginput(-1, timeout=-1))    
    idx_therm = np.arange(0,pts.shape[0],2)
    pts_therm = pts[idx_therm,:]
    idx_rgb = np.arange(1,pts.shape[0],2)
    pts_rgb = pts[idx_rgb,:]
    
    # Getting Difference between images to determine offset
    pts_diff = pts_therm - pts_rgb  
    offset = np.around(np.mean(pts_diff, axis=0))
    
    plt.close()
    return offset, pts_therm, pts_rgb