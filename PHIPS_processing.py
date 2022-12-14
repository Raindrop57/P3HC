# -*- coding: utf-8 -*-

import tensorflow as tf
import pathlib
import PIL
import numpy as np
import datetime
from os.path import exists
import os
from netCDF4 import Dataset
from scipy.optimize import curve_fit
import shutil
import math
from statistics import mean

def process_raw_PHIPS_images(import_path, export_path, campaign, e_b, dark_threshold, x_dim, y_dim):
    
    '''
    function process_raw_PHIPS_images
    
    parameters:
        import_path: (string) path to directory containing raw PHIPS images in png format
        export_path: (string) path to directory where processed images will be output
        campaign: parameter for a specific aircraft campaign, allows for conditional adjustments
        e_b: (edge buffer) number of pixels at edges of resized image to ignore when processing
            this is used as the edges of images tend to be dark. The "edge" of the
            image is then treated as being this buffer of pixels away from the
            true edge
        dark_threshold: maximum brightness threshold for a pixel to be colored black when 
        converting to black-and-white. (in raw images, 0 = completely black,
                                        255 = completely white)
        
    output:
        directory containing processed PHIPS images
    '''
    
    # Note: 1 pixel is 0.01 mm after processing.
    
    # Path for directory containing raw PHIPS images in .png format
    image_root = pathlib.Path(import_path)
    
    # Replace previously existing images in output directory, if applicable
    noreplace = False
    
    # Read filenames as an ordered list of strings; this is done so that images
    # are processed in an expected order
    # All files in folder ending with *.png are read
    list_ds = list(image_root.glob('*png'))
    list_temp = np.empty(shape=(len(list_ds),), dtype='object')
    ii = 0
    for f in list_ds:
        list_temp[ii] = str(f)
        ii += 1
    list_temp = np.sort(list_temp)
    list_ds = list_temp   
    
    # Number of raw images processed
    num_iterations=0
    
    # Number of images accepted after processing
    num_accepted=0
    
    # Create output directory if it doesn't exist
    if not os.path.isdir(export_path):
        os.mkdir(export_path)
    
    # Process raw images
    for im_path in list_ds: 
        
        if num_iterations % 1000 == 0:
            print(f"{num_iterations} images processed ({datetime.datetime.now()})")
        
        # Get name of image within its directory; used when exporting processed image
        im_name = im_path
        im_name = im_name[len(import_path):]
    
        # Open and resize image, then convert to array format for easy processing
        try:
            im = PIL.Image.open(im_path)
            im = im.resize((x_dim,y_dim))
            im = np.array(im)
            
        except OSError:
            print(f"OSError, image name = {im_name}")
            continue
        
        if campaign == 'IMPACTS':
            im[6:8,13:15] = 255 #Remove dark spot on image
        
        # Convert image to black and white, according to a threshold value    
        im[im < dark_threshold] = 0
        im[im >= dark_threshold] = 255
        
        # Parameters used to apporximate crystal boundaries
        maxx = -1
        maxy = -1
        minx = -1
        miny = -1
        
        on_edge = False
        is_crystal = False
        
        # Look for any activated pixels in each column, find first and last
        # columns with any activated pixels
        for j in range(e_b,x_dim-e_b):
            if np.min(im[e_b:y_dim-e_b,j]) <= dark_threshold:
                if miny < 0:
                    miny = j
                    
                maxy = j
           
        # Look for any fully activated pixels in each row
        for k in range(e_b,y_dim-e_b):
            if np.min(im[k,e_b:x_dim-e_b]) <= dark_threshold:
    
                if minx < 0:
                    minx = k
    
                maxx = k
        
        # Part of crystal is touching the edge if any edge pixels are fully "on"
        if minx == e_b or maxx == y_dim - e_b - 1 or miny == e_b or maxy == x_dim - e_b - 1:
            on_edge = True
        
        # Some shaded areas are present
        if miny >= 0 and minx >= 0 and maxy > miny and maxx > minx:
    
            crystal_rectangle = im[minx:maxx+1,miny:maxy+1]
    
            # Not considered to be a proper crystal if under 5 percent of pixels
            # within smallest coordinate-locked rectangle enclosing the crystal
            # are activated; thus the image is rejected
            if np.percentile(crystal_rectangle, 5) < dark_threshold:
                is_crystal = True

        # Crystal not touching edge; image is accepted    
        if (not on_edge) and is_crystal:
    
            '''
            plt.figure() #Test images
            plt.pcolormesh(im, cmap='Greys_r', vmin=0,vmax=255)
            '''
            
            num_accepted += 1
            
            a_str = str(num_accepted)
            a_str = a_str.zfill(6)
            
            if exists(f'{export_path}{im_name}') and noreplace:
                continue
            else:
                tf.keras.utils.save_img(f'{export_path}{im_name[len(im_name) - 56:len(im_name) - 13]}{a_str}.png', im.reshape(y_dim,x_dim,1), grayscale=True)
        
        # Perform parabolic fit test for crystals that are touching edge       
        elif on_edge and is_crystal:
            
            diode_counts_y = np.zeros(shape=(y_dim - 2*e_b,))
            diode_counts_x = np.zeros(shape=(x_dim - 2*e_b,))
            
            # Count number of darkened pixels in each row
            for kk in range(0, y_dim - 2*e_b):
                diode_counts_y[kk] = x_dim - e_b*2 - np.sum(im[kk,e_b:x_dim-e_b]) / 255
            
            # Count number of darkened pixels in each column
            for jj in range(0, x_dim - 2*e_b):
                diode_counts_x[jj] = y_dim - e_b*2 - np.sum(im[e_b:y_dim-e_b,jj]) / 255
            
            # Apply quadratic regression to darkened pixel counts in each row and column,
            # then look for local maximum point in regression using the second
            # derivative test
            x_poly = np.polyfit(np.arange(len(diode_counts_x)), diode_counts_x, 2)
            x_poly = np.poly1d(x_poly)
            x_crit = x_poly.deriv().r
            r_x_crit = x_crit[x_crit.imag==0].real
            test_2nd_deriv_x = x_poly.deriv(2)(r_x_crit)
            x_max = r_x_crit[test_2nd_deriv_x < 0]
            
            y_poly = np.polyfit(np.arange(len(diode_counts_y)), diode_counts_y, 2)
            y_poly = np.poly1d(y_poly)
            y_crit = y_poly.deriv().r
            r_y_crit = y_crit[y_crit.imag==0].real
            test_2nd_deriv_y = y_poly.deriv(2)(r_y_crit)
            y_max = r_y_crit[test_2nd_deriv_y < 0]
            
            # Check if a local maximum exists in both dimensions
            if (len(x_max) > 0) and (len(y_max) > 0):
            
                
                x_max = x_max[0]
                y_max = y_max[0]
                
                # Check if local maxima are both within the boundaries of the image
                if x_max > e_b and x_max < x_dim-e_b and y_max > e_b and y_max < y_dim-e_b:
                    
                    # The image is considered to be "center-in", so it will be included
                    
                    num_accepted += 1
                    
                    a_str = str(num_accepted)
                    a_str = a_str.zfill(6)
                    
                    if exists(f'{export_path}{im_name}') and noreplace:
                        continue
                    else:
                        tf.keras.utils.save_img(f'{export_path}{im_name[len(im_name) - 56:len(im_name) - 13]}{a_str}.png', im.reshape(y_dim,x_dim,1), grayscale=True)
                    #print("We have a crystal!")
                
        num_iterations += 1
        
def analyze_images(import_path, export_path, date, campaign, write_netCDF, e_b, dark_threshold):
    
    
    '''
    function analyze_images
    
    Analyze preprocessed PHIPS images, and classify by their holroyd habit.
    
    parameters:
        import_path: directory containing preprocessed PHIPS images,
        using the process_raw_PHIPS_images function
        export_path: directory to place output netCDF data file
        date: start date of aircraft flight from which this data was collected
        (format YYYYMMDD) - for file naming purposes
        campaign: parameter allowing for campaign-specific conditional statements
        to be added
        write_netCDF: write a netCDF data file. This should always be "True" except
        for testing purposes
        e_b: number of pixels on edge of image to ignore in analysis
        dark_threshold: brightness level (0-255) considered as a "darkened" pixel
    '''

    pixel_in_mm = 0.01

    # Get maximum dimension of crystal image 'im' (in terms of pixels) by finding the 
    # diagonal length of the smallest enclosing coordinate-locked rectangle
    def get_dmax(im):
        
        maxx = -1
        maxy = -1
        minx = -1
        miny = -1
        
        #len(im[0,:]) = x dimension (default 340)
        #len(im[:,0]) = y dimension (default 256)
        
        # Look for any fully activated pixels in each column, find first and last
        # columns with any activated pixels
        for j in range(e_b,len(im[0,:])-e_b):
    
            if np.min(im[e_b:len(im[:,0])-e_b,j]) < 255:
                if minx < 0:
                    minx = j
                    
                maxx = j
           
        # Look for any fully activated pixels in each row
        for k in range(e_b,len(im[:,0])-e_b):
    
            if np.min(im[k,e_b:len(im[0,:])-e_b]) < 255:
    
                if miny < 0:
                    miny = k
    
                maxy = k
    
        # Return maximum dimension of crystal, as well as x and y dimensions
        # of smallest enclosing rectangle
        D_max = round(np.sqrt((maxy-miny)**2 + (maxx-minx)**2))
        
        x_dim = maxy - miny
        y_dim = maxx - minx
        
        return D_max, x_dim, y_dim
        
    
    # Get perimeter of crystal, defined as the number of changes in pixel
    # state when making passes of each individual row and column
    def get_perim(im):
        
        perim = 0
        area = 0
        
        widths = np.empty(shape = (0,))
        lengths = np.empty(shape = (0,))
        
        for x in range (e_b, len(im[0,:]) - e_b):
            
            min_y_col = -1
            max_y_col = -1
    
            val_prev = im[e_b, x]
            for y in range(e_b + 1, len(im[:,0]) - e_b):
                val = im[y, x]
                
                if val != val_prev:
                    # Add 1 to perimeter for each change in diode status
                    perim += 1
                val_prev = val
                
                if(val < 255):
                    if min_y_col < 0:
                        min_y_col = y
                    max_y_col = y
                    
                if val < 255:
                    area += 1
    
            # Add 1 to perimeter for every illuminated diode on edge
            if im[e_b, x] < 255:
                perim += 1
            if im[len(im[:,0]) - e_b - 1, x] < 255:
                perim += 1
                
            if min_y_col >= 0:
                widths = np.append(widths, max_y_col - min_y_col)
        
            
        for y in range (e_b, len(im[:,e_b])):
            val_prev = im[y, e_b]
            
            min_x_col = -1
            max_x_col = -1
            
            for x in range(e_b + 1, len(im[0,:]) - e_b):
                val = im[y, x]
                
                if val != val_prev:
                    perim += 1
                val_prev = val
                
                if(val < 255):
                    if min_x_col < 0:
                        min_x_col = x
                    max_x_col = x
                    
            if min_x_col >= 0:
                lengths = np.append(lengths, max_x_col - min_x_col)
    
       
            # Add 1 to perimeter for every illuminated diode on edge
            if im[y, e_b] < 255:
                perim += 1
            if im[y, len(im[0,:]) - e_b - 1] < 255:
                perim += 1
        
        w_l_std = min(round(np.std(lengths), 1), round(np.std(widths), 1))
        
        return perim, w_l_std, area        
    
    # Get coefficient of determination for a crystal image
    def get_r2(im):
         
         min_xr = 32768
         max_xr = -32768
         min_yr = 32768
         max_yr = -32768
         
         x_vals = np.empty(shape = (0,))
         y_vals = np.empty(shape = (0,))
         
         xr_vals = np.empty(shape = (0,))
         yr_vals = np.empty(shape = (0,))   
         
         for x in range (e_b, len(im[0,:]) - e_b):
             for y in range(e_b, len(im[:,0]) - e_b):
                 val = im[y, x]
                 if val < 255:
                     x_vals = np.append(x_vals, x)
                     y_vals = np.append(y_vals, y)
                     
         r_2 = round(np.corrcoef(x_vals, len(im[0,:])-y_vals)[0,1]**2, 3)
    
         slope, intercept = np.polyfit(x_vals, len(im[0,:])-y_vals, 1)
         angle_radian = np.arctan(slope)
         
         if(angle_radian < 0):
             angle_radian += math.pi
         
         for x in range (e_b, len(im[0,:]) - e_b):
             for y in range(e_b, len(im[:,0]) - e_b):
                 val = im[y, x]
                 
                 if val < 255:
                     xr = int(round((x * math.cos(angle_radian)) - (y * math.sin(angle_radian))))
                     yr = int(round((x * math.sin(angle_radian)) + (y * math.cos(angle_radian))))
                     
                     xr_vals = np.append(xr_vals, xr)
                     yr_vals = np.append(yr_vals, yr)
                     
                     if(xr < min_xr):
                         min_xr = xr
                         
                     if(xr > max_xr):
                         max_xr = xr
                         
                     if(yr < min_yr):
                         min_yr = yr
                     
                     if(yr > max_yr):
                         max_yr = yr
                         
         x_dim_shifted = int(max_xr - min_xr)
         y_dim_shifted = int(max_yr - min_yr)
         
         widths_r = np.empty(shape = (0,))
         lengths_r = np.empty(shape = (0,))
         
         max_gap_x = -32768
         max_gap_y = -32768
         prev_xr = 32768
         prev_yr = 32768
         
         for x in np.unique(xr_vals):
             yr_vals_row = yr_vals[xr_vals == x]
             widths_r = np.append(widths_r, max(yr_vals_row) - min(yr_vals_row))
    
             if (x - prev_xr > max_gap_x):
                 max_gap_x = x - prev_xr
             prev_xr = x
             
             
         for y in np.unique(yr_vals):
             yr_vals_row = xr_vals[yr_vals == y]
             lengths_r = np.append(lengths_r, max(yr_vals_row) - min(yr_vals_row))
    
             if (y - prev_yr > max_gap_y):
                 max_gap_y = y - prev_yr
             prev_yr = y
         
         w_l_r_percents = min(round(np.percentile(lengths_r, 75) - np.percentile(lengths_r, 25), 1),  
                          round(np.percentile(widths_r, 75) - np.percentile(widths_r, 25), 1))
         #print(f'{round(np.std(widths_r), 1)}, {round(np.std(lengths_r), 1)}')
         #print(f'{round(np.percentile(widths_r, 75) - np.percentile(widths_r, 25), 1)}, {round(np.percentile(lengths_r, 75) - np.percentile(lengths_r, 25), 1)}')
         
         max_gap = max(max_gap_x, max_gap_y)
         
         return r_2, x_dim_shifted, y_dim_shifted, w_l_r_percents, max_gap
    
    # Get opacity coefficient for image, with re_f being a reduction factor
    # for the image resolution
    def get_O(im_string, re_f):
        
        count_fully_occulted = 0
        count_any_occulted = 0
        
        for x in range (e_b, len(im[0,:]) - e_b):

            vertical_slice = im[:,x]
            
            first_y = -1
            last_y = -1
            
            for y in range(e_b, len(im[:,0]) - e_b):
                
                if vertical_slice[y] < 255:
                    if first_y < 0:
                        first_y = y
                    last_y = y
                    
            if first_y >= 0:
                vertical_slice = vertical_slice[first_y:last_y + 1]
                count_any_occulted += 1
                if len(vertical_slice[vertical_slice < 255])/len(vertical_slice) >= 0.8:
                    count_fully_occulted += 1
        
        try:
            O = round(count_fully_occulted/count_any_occulted, 3)
        except ZeroDivisionError:
            O = 0
        return O
        
        '''
        # Get reduced-resolution image
        im_smaller = PIL.Image.open(im_string)
        im_smaller = im_smaller.resize((round(len(im[0,:])/re_f),round(len(im[:,0])/re_f)))
        im_smaller = np.array(im_smaller)
        im_smaller = im_smaller[round(e_b/re_f):round(len(im[:,0])/re_f - e_b/re_f),
                                 round(e_b/re_f):round(len(im[0,:])/re_f - e_b/re_f)]
        
        # Opacity coefficient is the number of darkened pixels in low-resolution image
        # divided by the number of pixels that are not 100% bright
        count_any = len(im_smaller[im_smaller < 255])
        count_total = len(im_smaller[im_smaller < dark_threshold])
        try:
            O = round(count_total/count_any, 3)
        except ZeroDivisionError:
            # I've encountered this before, so as a failsafe, O is set to zero
            O = 0
        return O
        '''
    
    def get_all_params(im):
        
        x_vals = np.empty(shape = (0,))
        y_vals = np.empty(shape = (0,))
        
        for x in range (e_b, len(im[0,:]) - e_b):
            for y in range(e_b, len(im[:,0]) - e_b):
                val = im[y, x]
                if val < 255:
                    x_vals = np.append(x_vals, x)
                    y_vals = np.append(y_vals, y)
                    
        r_2 = round(np.corrcoef(x_vals, len(im[0,:])-y_vals)[0,1]**2, 3)
    
        if(r_2 >= 0.3):
            slope, intercept = np.polyfit(x_vals, len(im[0,:])-y_vals, 1)
            angle_radian = np.arctan(slope)
            
        else:
            angle_radian = 0
        
        if(angle_radian < 0):
            angle_radian += math.pi
        
        count_fully_occulted_r = 0
        count_any_occulted_r = 0
        count_fully_occulted_c = 0
        count_any_occulted_c = 0
     
        perim = 0
        area = 0
        num_on_edge = 0
        
        xr_vals = np.empty(shape = (0,))
        yr_vals = np.empty(shape = (0,))   
        
        min_xr = 32768
        max_xr = -32768
        min_yr = 32768
        max_yr = -32768
        
        for x in range (e_b, len(im[0,:]) - e_b):
    
            val_prev = im[e_b, x]
            for y in range(e_b + 1, len(im[:,0]) - e_b):
                val = im[y, x]
                
                if val != val_prev:
                    # Add 1 to perimeter for each change in diode status
                    perim += 1
                val_prev = val
                
                if(val < 255):
                    
                    xr = int(round((x * math.cos(angle_radian)) - (y * math.sin(angle_radian))))
                    yr = int(round((x * math.sin(angle_radian)) + (y * math.cos(angle_radian))))
                    
                    xr_vals = np.append(xr_vals, xr)
                    yr_vals = np.append(yr_vals, yr)
                    
                    if(xr < min_xr):
                        min_xr = xr
                        
                    if(xr > max_xr):
                        max_xr = xr
                        
                    if(yr < min_yr):
                        min_yr = yr
                    
                    if(yr > max_yr):
                        max_yr = yr
                    
                if val < 255:
                    area += 1
    
            # Add 1 to perimeter for every illuminated diode on edge
            if im[e_b, x] < 255:
                perim += 1
                num_on_edge += 1
            if im[len(im[:,0]) - e_b - 1, x] < 255:
                perim += 1
                num_on_edge += 1
                  
        x_dim_shifted = int(max_xr - min_xr)
        y_dim_shifted = int(max_yr - min_yr)
        
        widths_r = np.empty(shape = (0,))
        lengths_r = np.empty(shape = (0,))
    
        max_gap_x = -32768
        max_gap_y = -32768
        prev_xr = 32768
        prev_yr = 32768
        
        for x in np.unique(xr_vals):
            yr_vals_row = yr_vals[xr_vals == x]
            
            width_r =  max(yr_vals_row) - min(yr_vals_row) + 1
            frac_occulted_r = len(yr_vals_row)/width_r
            
            widths_r = np.append(widths_r, width_r)
            
            if frac_occulted_r >= 0.8:
                count_fully_occulted_r += 1
            count_any_occulted_r += 1
    
            if (x - prev_xr > max_gap_x):
                max_gap_x = x - prev_xr
            
            prev_xr = x
        
        
        for y in np.unique(yr_vals):
            xr_vals_col = xr_vals[yr_vals == y]
            length_c =  max(xr_vals_col) - min(xr_vals_col) + 1
            frac_occulted_c = len(xr_vals_col)/length_c
            lengths_r = np.append(lengths_r, length_c)
            
            if frac_occulted_c >= 0.8:
                count_fully_occulted_c += 1
            count_any_occulted_c += 1
    
            if (y - prev_yr > max_gap_y):
                max_gap_y = y - prev_yr
                
            prev_yr = y
    
        #print(f'{round(np.std(widths_r), 1)}, {round(np.std(lengths_r), 1)}')
        #print(f'{round(np.percentile(widths_r, 75) - np.percentile(widths_r, 25), 1)}, {round(np.percentile(lengths_r, 75) - np.percentile(lengths_r, 25), 1)}')
    
        max_gap = max(max_gap_x, max_gap_y)
            
        for y in range (e_b, len(im[:,e_b])):
            val_prev = im[y, e_b]
            
            for x in range(e_b + 1, len(im[0,:]) - e_b):
                val = im[y, x]
                
                if val != val_prev:
                    perim += 1
                val_prev = val
       
            # Add 1 to perimeter for every illuminated diode on edge
            if im[y, e_b] < 255:
                perim += 1
                num_on_edge += 1
            if im[y, len(im[0,:]) - e_b - 1] < 255:
                perim += 1
                num_on_edge += 1
        
        w_l_r_p = min(round(np.percentile(lengths_r, 75) - np.percentile(lengths_r, 25), 1),  
                     round(np.percentile(widths_r, 75) - np.percentile(widths_r, 25), 1))
            
        d_max = max(x_dim_shifted, y_dim_shifted)
        
        cir_area_ratio = area/(math.pi*((d_max/2)**2))
        
        edge_diam_ratio  = num_on_edge / (2 * d_max)
        
        try:
            O = round(mean([count_fully_occulted_r/count_any_occulted_r, count_fully_occulted_c/count_any_occulted_c]), 3)
        except ZeroDivisionError:
            O = 0
        
        return perim, w_l_r_p, area, O, r_2, x_dim_shifted, y_dim_shifted, max_gap, d_max, cir_area_ratio, edge_diam_ratio
        
        # Calculate the crystal habit based on its morphological properties
 
    '''
    def get_habit(a, r_2, d, O, F, x, y, w_l_p, m_g, c_r):
        if a < 50:
            return 0 # Tiny
        if (m_g > 1 + d/10):
            return 7 # Irregular
        #print(w_l_p)
        
        if (x > (220 - e_b * 2) * 0.8) and (y > (165 - e_b * 2) * 0.8):
            needed_ratio = 5
        else:
            needed_ratio = min((1.2 + (w_l_p/20) + (6/d)*(w_l_p)), 5)
        
        if (x >= needed_ratio * y) or (y >= needed_ratio * x):
            return 1 # Column
        if d >= 100:
            if F <= (30 - (abs(1 - c_r)*20)):
                return 3 # Graupel
            elif F > 100:
                return 6 # Dendrite
            elif d >= 140:
                return 2 # Aggregate
        if F <= (15 - (abs(1 - c_r)*10)):
            return 4 # Sphere
        else:
            if O < 0.30 or F > 100:
                # Hexagonal planar crystals tend to have low opacity
                return 5 # Simple planar
            else:
                return 7 # Irregular
        '''
            
   
    def get_habit(a, r_2, d, O, F, x, y, w_l_p, m_g, c_r, e_d_r):
        if a < 50:
            return 0 # Tiny
        #print(w_l_p)
        
        if m_g > 1 + d/4:
            return 7 # Irregular
        
        needed_ratio = min((1.2 + (w_l_p/20) + (6/d)*(w_l_p) + e_d_r), 5)
        
        needed_ratio +=  m_g/5
        
        aspect_ratio = max(x/y, y/x)
        
        if (aspect_ratio >= needed_ratio):
            return 1 # Column
        if d >= 100:
            if F <= (40 - (abs(1 - c_r)*20)) / aspect_ratio:
                return 3 # Graupel
            elif F > 85 * aspect_ratio:
                return 6 # Dendrite
            elif d >= 140:
                return 2 # Aggregate
        if F <= ((20 - (abs(1 - c_r)*10))) / aspect_ratio:
            return 4 # Sphere
        else:
            if O < 0.30 or F > 100:
                # Hexagonal planar crystals tend to have low opacity
                return 5 # Simple planar
            else:
                return 7 # Irregular
            
        '''
        if a < 100:
            return 0 #Tiny
        if (r_2 >= 0.6) or (d < 150 and (x >= 2*y or y >= 2*x)) or (x >= 4*y or y >= 4*x):
            return 1 #Linear or oriented
        if O >= 0.75 and d >= 100 and d < 250 and F <= 40:
            if F <= 15:
                return 4 # Sphere
            else:
                return 3 # Graupel
        if d >= 250:
            if F <= 40:
                return 3 # Graupel
            elif F > 150:
                return 6 # Dendrite
            else:
                return 2 # Aggregate
        if F <= 20:
            return 4 # Sphere
        if F >= 200:
            return 6 # Dendrite
        else:
            if O < 0.35 and d >= 40:
                # Hexagonal planar crystals tend to have low opacity
                return 5 # Simple planar
            else:
                return 7 # Irregular
        '''

    # How habit categories are labeled in output file
    habit_classes =  \
    {0: 'tiny',
     1: 'column',
     2: 'aggregate',
     3: 'graupel',
     4: 'sphere',
     5: 'planar',
     6: 'dendrite',
     7: 'irregular',}

    # Get and organize processed images from folder. Using raw images will 
    # likely produce unexpected results
    image_root = pathlib.Path(import_path)

    list_ds = list(image_root.glob('*.png'))

    list_strings = np.empty(shape=(len(list_ds),), dtype='object')
    ii = 0
    for f in list_ds:
        list_strings[ii] = str(f)
        ii += 1
    list_strings = np.sort(list_strings)

    # Maximum number of particles to analyze for testing;
    # set to len(list_strings) to run fully
    zmax = len(list_strings)

    filelen = min(len(list_strings), zmax)

    # Create arrays to contain data to later be written into a netCDF file
    habits = np.empty(shape = (filelen,), dtype = 'int16')
    times = np.empty(shape = (filelen,), dtype = 'int64')
    times_sec = np.empty(shape = (filelen,), dtype = 'int64') 
    dates = np.empty(shape = (filelen,), dtype = 'int64')  
    dmaxes = np.empty(shape = (filelen,), dtype = 'float32')
    y_dims = np.empty(shape = (filelen,), dtype = 'float32')
    x_dims = np.empty(shape = (filelen,), dtype = 'float32')
    areas = np.empty(shape = (filelen,), dtype = 'float32')
    o_coeffs = np.empty(shape = (filelen,), dtype = 'float32')
    f_coeffs = np.empty(shape = (filelen,), dtype = 'float32')
    c_coeffs = np.empty(shape = (filelen,), dtype = 'float32')
    perims = np.empty(shape = (filelen,), dtype = 'float32') 

    # Setup framework for output netCDF file
    if write_netCDF:
        ncfile = Dataset(f'{export_path}/PHIPS_habits_{date}.nc',
                         mode = 'w', format='NETCDF4')
        
        time_dim = ncfile.createDimension('time', filelen)
        time_var = ncfile.createVariable('Time', np.int64, ('time',))
        time_in_seconds = ncfile.createVariable('time_in_seconds', np.int64, ('time',))
        date_var = ncfile.createVariable('Date', np.int64, ('time',))
        dmax = ncfile.createVariable('image_diam_minR', np.float32, ('time',))
        y_dimension = ncfile.createVariable('image_width', np.float32, ('time',)) 
        x_dimension = ncfile.createVariable('image_length', np.float32, ('time',)) 
        area = ncfile.createVariable('image_area', np.float32, ('time',)) 
        opacity_coef = ncfile.createVariable('opacity_coef', np.float32, ('time',)) 
        feature_coef = ncfile.createVariable('fine_detail_ratio', np.float32, ('time',))
        corr_coef = ncfile.createVariable('corr_coef', np.float32, ('time',))
        habit_var = ncfile.createVariable('holroyd_habit', np.int16, ('time',)) 
        perim = ncfile.createVariable('image_perimeter',  np.float32, ('time',))  

    z = 0

    # Analyze each crystal image
    for im_string in list_strings: 
        
        # Process only the first zmax images for testing purposes
        if z >= zmax:
            break
        
        if z % 1000 == 0:
            print(f"{z} images analyzed ({datetime.datetime.now()})")
        
        # Parse time from filename
        time_sec = int(im_string[len(im_string) - 13:len(im_string) - 11])
        time_min = int(im_string[len(im_string) - 15:len(im_string) - 13])
        time_hr = int(im_string[len(im_string) - 17:len(im_string) - 15])
        time_overall = int(im_string[len(im_string) - 17:len(im_string) - 11])
        date = int(im_string[len(im_string) - 25:len(im_string) - 17])
        time_sec_only = time_sec + 60*time_min + 3600*time_hr
        
        im = PIL.Image.open(im_string)

        #im = im.resize((x_dim,y_dim))
        im = np.array(im)
        
        p, w_l_p, a, O, r_2, x, y, m_g, d, c_r, e_d_r = get_all_params(im)
        
        # Calculate feature ratio
        F = round(p*d/a, 3)
        
        # Get Holroyd habit
        habit = int(get_habit(a, r_2, d, O, F, x, y, w_l_p, m_g, c_r, e_d_r))
         
        # Populate data arrays
        habits[z] = habit
        times[z] = time_overall
        times_sec[z] = time_sec_only
        dates[z] = date
        dmaxes[z] = d
        y_dims[z] = x
        x_dims[z] = y
        areas[z] = a
        o_coeffs[z] = O
        f_coeffs[z] = F
        c_coeffs[z] = r_2
        perims[z] = p 
        
        # Print out particle info for testing purposes
        '''
        print(habit_classes[habit], end = " ")
        print(f'a = {a}, r_2 = {r_2}, d = {d}, O = {O}, F = {F}, x = {x}, y = {y}, p = {p}')
        '''
        
        '''
        if d * pixel_in_mm > 0.1 and habit > 0:
            shutil.copy(im_string, f'/Users/julian/Desktop/IMPACTS/PHIPS_classification_results/{habit_classes[habit]}/testing_{date}_{z}.png')
        '''

        z += 1

    if write_netCDF:
        
        # Match habit codes to those of UIOOPS processing software for other probes
        habits[habits == 0] = 116
        habits[habits == 1] = 108 # Mark all columns as "linear"
        habits[habits == 2] = 97
        habits[habits == 3] = 103
        habits[habits == 4] = 115
        habits[habits == 5] = 104
        habits[habits == 6] = 100
        habits[habits == 7] = 105
        
        # Populate variables
        time_var[:] = times
        time_in_seconds[:] = times_sec  
        date_var[:] = dates
        dmax[:] = dmaxes * pixel_in_mm
        y_dimension[:] = y_dims * pixel_in_mm 
        x_dimension[:] = x_dims * pixel_in_mm
        area[:] = areas * (pixel_in_mm**2)
        opacity_coef[:] = o_coeffs
        feature_coef[:] = f_coeffs
        corr_coef[:] = c_coeffs
        habit_var[:] = habits
        perim[:] = perims * pixel_in_mm 
        
        ncfile.close()
