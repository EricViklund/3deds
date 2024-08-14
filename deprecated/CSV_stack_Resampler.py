#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:13:36 2023

@author: eric
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from skimage.measure import block_reduce
from PIL import Image
import plotly.express as px
import plotly.io as pio
# pio.renderers.default = 'svg'
pio.renderers.default = 'browser'
from mayavi import mlab


#%%

"""Load Data"""
#PARAMS
directory = "/home/eric/Documents/Fermilab_Superconductors/Nb3Sn_Polishing/Data/E42_EDS_15x30um/"
data_folder = directory+'Sn/'
output_folder =  directory+'output'


files = os.listdir(data_folder)
sorted_files = sorted(files, key=lambda x: int(x.split(' ')[5]))

Sn_stack = []

for file in sorted_files:
    Sn_stack.append(np.genfromtxt(data_folder+file, delimiter=',', dtype=None)[:,:-1])
    
binning = 5
    
Sn_stack = np.array(Sn_stack)
Sn_stack = np.transpose(Sn_stack,(2,1,0))
Sn_stack = np.flip(Sn_stack,axis=1)
# Sn_stack = Sn_stack[:100,:100,:100]
Sn_stack = block_reduce(Sn_stack,block_size=binning)

image_z_resolution = Sn_stack.shape[2]
image_resolution = Sn_stack.shape


"""Calculate Data Point Positions"""
#PARAMS
# pixel_size = np.array((0.040,0.040,0.040))
pixel_size = binning*np.array((40.,40.,40.))
milling_angle = 52*2*np.pi/360
ROI_dimensions = pixel_size*image_resolution

x_coords = pixel_size[0]*np.arange(0,image_resolution[0])
y_coords = pixel_size[1]*np.arange(0,image_resolution[1])
z_coords = pixel_size[2]*np.arange(0,image_resolution[2])

X,Y,Z = np.meshgrid(x_coords,y_coords,z_coords,indexing='ij')

#shift the Y coordinates
Y += np.sin(milling_angle)*Z

#Create a list of coordinates
coords = np.array((X.flatten(),Y.flatten(),Z.flatten()))

#Rotate the coordinates
R_x = np.array([[1, 0, 0],
                [0, np.cos(np.pi/2-milling_angle), -np.sin(np.pi/2-milling_angle)],
                [0, np.sin(np.pi/2-milling_angle), np.cos(np.pi/2-milling_angle)]])
coords = np.dot(R_x,coords)

#%%
"""Resample Data"""

#PARAMS
extent = np.array(((coords[0].min(),coords[0].max()),
          (coords[1].min(),coords[1].max()),
          (coords[2].min(),coords[2].max())))
sampling_pixel_size = np.array((160.,160.,160.))
sampling_res = np.array((int((extent[0,1]-extent[0,0])/sampling_pixel_size[0]),
                        int((extent[1,1]-extent[1,0])/sampling_pixel_size[1]),
                        int((extent[2,1]-extent[2,0])/sampling_pixel_size[2])))

#Calculate new grid coordinates
x_grid = np.linspace(extent[0,0], extent[0,1], num=sampling_res[0])
y_grid = np.linspace(extent[1,0], extent[1,1], num=sampling_res[1])
z_grid = np.linspace(extent[2,0], extent[2,1], num=sampling_res[2])

X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')



Sn = Sn_stack.flatten()

interp = LinearNDInterpolator(coords.T,Sn,fill_value=0.0)

Resampled_Sn = interp(X,Y,Z)

np.save('Resampled_Sn',Resampled_Sn)



#%%

# Plot the volume rendering using Mayavi
mlab.figure()
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(Resampled_Sn), vmin=6000)
mlab.show()









