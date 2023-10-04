#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 17:22:39 2022

@author: eric
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from PIL import Image

def save_im(image,path):
    im = Image.fromarray(image)
    im.save(path)
    
def append_to_stack(z_index, stack_list, h5_data, h5_path):
    data = np.array(f[h5_path],dtype = 'uint32').reshape(image_resolution[1],image_resolution[0]).T
    stack_list[:,:,z_index] = data
    return stack_list

def smooth(data, binning, epsilon):
    data = gaussian_filter(data, epsilon)
    data = block_reduce(data, block_size=binning, func=np.sum)
    return data
    
def crop(data, x, y, z):
    data = data[x:-x,y:-y,z:-z]
    return data
    

directory = "/home/eric/Documents/Fermilab_Superconductors/Nb3Sn_Polishing/Data/G34/"
data_folder = directory+'h5oina/'
output_folder =  directory+'output'


h5_files = os.listdir(data_folder)
sorted_files = sorted(h5_files, key=lambda x: int(x.split(' ')[4]))[:500]

f = h5py.File(data_folder+sorted_files[0], 'r')
pixel_size = (0.020,0.020/np.sin(52/360*2*np.pi),0.020)
image_resolution = (f['1/EDS/Header/X Cells'][0],f['1/EDS/Header/Y Cells'][0],len(sorted_files))
image_z_resolution = len(sorted_files)


#%%

Nb_La1_stack = np.zeros((image_resolution[0],image_resolution[1],image_z_resolution))
Pt_Ma1_stack = np.zeros((image_resolution[0],image_resolution[1],image_z_resolution))
Sn_La1_stack = np.zeros((image_resolution[0],image_resolution[1],image_z_resolution))


# try:
#     os.mkdir(output_folder+'Nb_La1')
#     os.mkdir(output_folder+'Pt_Ma1')
#     os.mkdir(output_folder+'Sn_La1')
# except FileExistsError:
#     print('Output folder already exists.')

for i in range(len(sorted_files)):
    file = sorted_files[i]
    f = h5py.File(data_folder+file, 'r')
    
    # def printname(name):
    #     print(name)
    # f.visit(printname)
    
    # Nb_La1 = np.array(f['1/EDS/Data/Window Integral/Nb La1'],dtype = 'uint32').reshape(image_resolution)   
    # Pt_Ma1 = np.array(f['1/EDS/Data/Window Integral/Pt Ma1'],dtype = 'uint32').reshape(image_resolution)
    # Sn_La1 = np.array(f['1/EDS/Data/Window Integral/Sn La1'],dtype = 'uint32').reshape(image_resolution)
    
    # Nb_La1_img = np.array(255*Nb_La1/np.max(Nb_La1),dtype='uint8')
    # Pt_Ma1_img = np.array(255*Pt_Ma1/np.max(Pt_Ma1),dtype='uint8')
    # Sn_La1_img = np.array(255*Sn_La1/np.max(Sn_La1),dtype='uint8')
    
    # save_im(Nb_La1_img,output_folder+'Nb_La1/'+str(i)+' Nb_La1.tiff')
    # save_im(Pt_Ma1_img,output_folder+'Pt_Ma1/'+str(i)+' Pt_Ma1.tiff')
    # save_im(Sn_La1_img,output_folder+'Sn_La1/'+str(i)+' Sn_La1.tiff')
    
    Nb_La1_stack = append_to_stack(i, Nb_La1_stack, f, '1/EDS/Data/Window Integral/Nb La1')
    Pt_Ma1_stack = append_to_stack(i, Pt_Ma1_stack, f, '1/EDS/Data/Window Integral/Pt Ma1')
    Sn_La1_stack = append_to_stack(i, Sn_La1_stack, f, '1/EDS/Data/Window Integral/Sn La1')
    
#%%

binning_number = 4
cropping = 10

pixel_size = (binning_number*0.020,
              binning_number*0.020/np.sin(52/360*2*np.pi),
              binning_number*0.020)
image_resolution = (image_resolution[0]/binning_number-2*cropping,
                    image_resolution[1]/binning_number-2*cropping,
                    image_resolution[2]/binning_number-2*cropping)

Nb_La1_stack = smooth(Nb_La1_stack, binning_number, 1.0)
Nb_La1_stack = crop(Nb_La1_stack, cropping, cropping, cropping)

Sn_La1_stack = smooth(Sn_La1_stack, binning_number, 2)
Sn_La1_stack = crop(Sn_La1_stack, cropping, cropping, cropping)

Pt_Ma1_stack = smooth(Pt_Ma1_stack, binning_number, 1.0)
Pt_Ma1_stack = crop(Pt_Ma1_stack, cropping, cropping, cropping)

Nb_Sn_ratio = Nb_La1_stack/(8.5*Sn_La1_stack+1e-6)
    
#%%

bounds = [0,image_resolution[0]*pixel_size[0],
          0,image_resolution[1]*pixel_size[1],
          0,image_resolution[2]*pixel_size[2],]

plt.figure()
plt.hist(Sn_La1_stack.flatten(),bins=100)
plt.xlabel('Sn Lα Count')
plt.ylabel('Number of Voxels')
plt.show()

slices = np.linspace(1, image_resolution[2],num=4)

for z_slice in slices:
    z_index = int(z_slice)-1
    plt.figure()
    plt.imshow(Sn_La1_stack[:,:,z_index].T,extent=bounds[0:4])
    plt.xlabel('X [μm]')
    plt.ylabel('Y [μm]')
    plt.title('Z = '+str(round(z_index*pixel_size[2],2))+'μm')
    plt.show()

#%%

# from mayavi import mlab

# mlab.figure(fgcolor=(0., 0., 0.), bgcolor=(1, 1, 1))
    
# data = Sn_La1_stack


# source = mlab.pipeline.scalar_field(data,ranges=bounds)
# data_min = data.min()
# data_max = data.max()
# vol = mlab.pipeline.volume(source, vmin=data_min + 0.3 * (data_max - data_min),
#                                     vmax=data_min + 0.9 * (data_max - data_min))

# # source.spacing = list(pixel_size)


# # mlab.axes(vol)

# # mlab.show()
# # levels = list(np.linspace(125000, 240000, 3))
# # mlab.contour3d(data,contours=[1.5e6],opacity=0.9,extent=bounds)
# mlab.colorbar(title='Sn Count',nb_labels=4)
# mlab.axes(ranges=bounds)
# mlab.xlabel('X [μm]')
# mlab.ylabel('Y [μm]')
# mlab.zlabel('Z [μm]')
# mlab.show()

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure




# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes(Sn_La1_stack, 1.5e6)

#%%

# # Display resulting triangular mesh using Matplotlib. This can also be done
# # with mayavi (see skimage.measure.marching_cubes docstring).
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')

# # Fancy indexing: `verts[faces]` to generate a collection of triangles
# mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
# ax.add_collection3d(mesh)

# # ax.set_xlabel("x-axis: a = 6 per ellipsoid")
# # ax.set_ylabel("y-axis: b = 10")
# # ax.set_zlabel("z-axis: c = 16")

# ax.set_xlim(0, 200)  # a = 6 (times two for 2nd ellipsoid)
# ax.set_ylim(0, 75)  # b = 10
# ax.set_zlim(0, 100)  # c = 16

# plt.tight_layout()
# plt.show()

#%%

import trimesh

mesh = trimesh.Trimesh(vertices=verts,faces=faces, vertex_normals=normals)

sample_resolution = (pixel_size[0],pixel_size[2])
sample_size = (int(image_resolution[0]),int(image_resolution[2]))

x = np.linspace(0,np.max(verts[:,0]),sample_size[0])
z = np.linspace(0,np.max(verts[:,2]),sample_size[1])

xx,zz = np.meshgrid(x,z,indexing='ij')

ray_direction = np.array((0,1,0))

bottom_surface = np.zeros((sample_size[0],sample_size[1]))
top_surface = np.zeros((sample_size[0],sample_size[1]))

for i in range(sample_size[0]):
    for j in range(sample_size[1]):
        ray_origin = np.array((xx[i,j],0,zz[i,j]))

        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origin[None,:],
                                                                       ray_directions=ray_direction[None,:])
        
        bottom_surface[i,j] = np.min(locations[:,1])
        top_surface[i,j] = np.max(locations[:,1])


#%%

thickness_distribution = (top_surface-bottom_surface)*pixel_size[1]

fig, ax = plt.subplots()

im = ax.imshow(thickness_distribution.T,
               cmap='gnuplot',
               extent=[0.0,image_resolution[0]*pixel_size[0],image_resolution[2]*pixel_size[2],0.0])
fig.colorbar(im, ax=ax, label='Film Thickness [μm]',orientation='horizontal',location='top')
ax.set_xlabel('X [μm]')
ax.set_ylabel('Z [μm]')

plt.show()

#%%



plt.figure()
plt.hist(thickness_distribution.flatten(),bins=100)

mean = thickness_distribution.flatten().mean()
std = thickness_distribution.flatten().std()
minimum = np.sort(thickness_distribution.flatten())[1:].min()
maximum = thickness_distribution.flatten().max()

print(mean)
print(std)
print(minimum)
print(maximum)

plt.axvline(x=mean,color='r')
plt.axvline(x=mean+std,color='r',ls='--')
plt.axvline(x=mean-std,color='r',ls='--')

plt.xlabel('Film Thickness [μm]')
plt.ylabel('Relative Frequency')


    
    