#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm
import numpy as np
from scipy.misc import imread
from os.path import dirname
import os
import sys
import time


# alpha channel: non-0 as 1
# rgb channels: assume to between (0,1)
#		otherwise clipped to be (0,1)


def visimage(imname):
	img = imread(imname)
	plt.imshow(img)
	plt.show()
	

if __name__ == '__main__':
	imname = os.path.abspath(sys.argv[1])

	# load obj file and convert to vox
	visimage(imname)























