#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:16:12 2022

@author: julian
"""

from PHIPS_processing import process_raw_PHIPS_images, analyze_images  

# Variable for aircraft campaign to make flight-specific corrections
campaign = 'IMPACTS'
    
# Seperate variable for the date so you can't have different input and
# output folder dates by mistake
date = 20200118

# Path for directory containing all PHIPS images
main_path = '/Volumes/Data/IMPACTS_Data/PHIPS_Images/'

# Directory containing raw PHIPS images
initial_import_path = f'{main_path}PHIPS_Data_{date}_raw/'

# Directory containing processed PHIPS images
initial_export_path = f'{main_path}PHIPS_Data_{date}_unlabeled/'

# Directory to place output netCDF file from image analysis
final_export_path = '/Users/julian/Desktop/IMPACTS/PHIPS_habit_classifications/'

# Number of pixels to exclude at edge of image due to the edges being especially dark
edge_buffer = 5

# Grayscale brightness threshold at which pixels are colored black or white
dark_threshold = 255 * 0.4

# Dimensions for which the new image will be processed in; changing these
# will require adjustments in the habit classification
processed_image_x_dim = 340

processed_image_y_dim = 256

# Process the raw images to black and white
#process_raw_PHIPS_images(initial_import_path, initial_export_path, campaign,
#                         edge_buffer, dark_threshold, processed_image_x_dim, processed_image_y_dim)

# Assess image properties and determine habit
analyze_images(initial_export_path, final_export_path, date, campaign, False,
               edge_buffer, dark_threshold)

