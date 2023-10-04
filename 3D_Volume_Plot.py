#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:24:52 2023

@author: eric
"""

import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from skimage import measure
import trimesh




Resampled_Sn = np.load('Resampled_Sn.npy')[:-2,:,20:45]




verts, faces, normals, values = measure.marching_cubes(Resampled_Sn, 600)

# Plot the mesh using Mayavi
                      

mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], faces, scalars=values)
mlab.show()


mesh = trimesh.Trimesh(vertices=verts,faces=faces, vertex_normals=normals)

pixel_size = np.array((160.,160.,160.))
image_resolution = Resampled_Sn.shape

x = np.linspace(0,np.max(verts[:,0]),image_resolution[0])
z = np.linspace(0,np.max(verts[:,2]),image_resolution[2])

xx,zz = np.meshgrid(x,z,indexing='ij')

ray_direction = np.array((0,1,0))

bottom_surface = np.zeros((image_resolution[0],image_resolution[2]))
top_surface = np.zeros((image_resolution[0],image_resolution[2]))

for i in range(image_resolution[0]):
    for j in range(image_resolution[2]):
        ray_origin = np.array((xx[i,j],0,zz[i,j]))

        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins=ray_origin[None,:],
                                                                       ray_directions=ray_direction[None,:])
        
        if locations.shape[0] != 0:
            bottom_surface[i,j] = np.min(locations[:,1])
            top_surface[i,j] = np.max(locations[:,1])


#%%

thickness_distribution = (top_surface-bottom_surface)*pixel_size[1]

fig, ax = plt.subplots()

im = ax.imshow(thickness_distribution.T,
               cmap='gnuplot',
               extent=[0.0,image_resolution[0]*pixel_size[0],image_resolution[2]*pixel_size[2],0.0])
fig.colorbar(im, ax=ax, label='Film Thickness [nm]',orientation='horizontal',location='top')
ax.set_xlabel('X [nm]')
ax.set_ylabel('Z [nm]')

plt.show()

#%%



plt.figure()
plt.hist(thickness_distribution.flatten()[thickness_distribution.flatten()!=0.0],bins=100)

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

plt.xlabel('Film Thickness [nm]')
plt.ylabel('Relative Frequency')

plt.show()


#%%

# Plot the volume rendering using Mayavi
mlab.figure()
vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(Resampled_Sn), vmin=200, vmax=800)
mlab.show()
