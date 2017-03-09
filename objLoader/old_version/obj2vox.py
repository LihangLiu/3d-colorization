import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
from os.path import dirname
 
# IMPORT OBJECT LOADER
from objloader import *

import numpy as np

obj = OBJ(sys.argv[1], swapyz=True)

# discretize the grid as n*n*n
n = 64
delta = 1.0/n
#grid = np.zeros((n,n,n), dtype = np.int)
grid = np.zeros((n,n,n,4), dtype = np.int)

#for i, coord in enumerate(obj.vertices):
#    grid[int((coord[0]+0.5)/delta), int((coord[1]+0.5)/delta), int((coord[2]+0.5)/delta)] += 1

#np.save(dirname(sys.argv[1])+'/voxel.npy', grid)

for face in obj.faces:
    vertices, normals, texture_coords, material = face
    mtl = obj.mtl[material]
    if 'texture_Kd' in mtl:
        img = mtl['map_Kd']
    for i in range(len(vertices)):
        if texture_coords[i] > 0:
            tex_world = (obj.texcoords[texture_coords[i] - 1])
        vertice_world = (obj.vertices[vertices[i] - 1])
        grid[int((vertice_world[0]+0.5)/delta), int((vertice_world[1]+0.5)/delta), int((vertice_world[2]+0.5)/delta),0] += 1

np.save(dirname(sys.argv[1])+'/voxel.npy', grid)

        
