
import numpy as np
import os
import skimage
import PIL
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from skimage.exposure import histogram
from skimage.util import random_noise
import math
import cv2
from skimage import io
from skimage import img_as_ubyte , img_as_int

from skimage.filters import threshold_otsu
from scipy.signal import find_peaks
import porespy as ps
import matplotlib.pyplot as plt
from  scipy import stats
from skimage.external import tifffile as tif
from skimage.filters import unsharp_mask


    
def sythetic_gaussian_image(im):
    thresholded_image=img_as_ubyte(im)

    inveted_image=np.where((thresholded_image==0.0), np.random.random_integers(30,40,1),thresholded_image)
    inveted_image=np.where((inveted_image==255.0),np.random.random_integers(100,130,1),inveted_image)
    noise=np.random.normal(size=inveted_image.shape)
    noise=img_as_ubyte(noise/np.maximum(np.absolute(noise.min()),noise.max()))
    blu=gaussian_filter(inveted_image-noise, sigma=5)
    result_3 = unsharp_mask(blu, radius=30, amount=3)

    sharped_image=img_as_ubyte(result_3/result_3.max())
    blu2=gaussian_filter((sharped_image), sigma=1)
    
    return blu2

# python3 synthetic_generator.py blob --3D_dir='3d' --grayscale_image_dir='input' --GT_image_dir='output'

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generating sythentic porous images')
    
    parser.add_argument("command",
                        metavar="<command>",
                        help="'blob' or 'overlapping_spheres' or 'combine'")

    parser.add_argument('--porosity', type=float,required=False,)

    parser.add_argument('--radius_size', type=float,required=False,)

    parser.add_argument('--blob_size', type=float,required=False,)


    parser.add_argument('--fileName', required=False,
                        metavar="name",
                        help='name of file')
    parser.add_argument('--three_3D_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the 3D dataset')

    parser.add_argument('--grayscale_image_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the grayscale dataset')

    parser.add_argument('--GT_image_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the ground truth dataset')
    
   
    args = parser.parse_args()
    
 
    if not os.path.exists(args.grayscale_image_dir):
            os.makedirs(args.grayscale_image_dir)
    if not os.path.exists(args.GT_image_dir):
            os.makedirs(args.GT_image_dir)
    if not os.path.exists(args.three_3D_dir):
            os.makedirs(args.three_3D_dir)



    fileN=args.fileName


        
    


    if args.command == "blob":
       # blob=2
      #  p=0.6
        p=args.porosity
        blob=args.blob_size
        im1 = ps.generators.blobs(shape=[100,256,256], porosity=p, blobiness=blob)
        noise=sythetic_gaussian_image(im1)
        for i in range(100):
            name1=fileN+'_'+'blob'+str(blob)+'_'+'p'+str(p)+str(i)+'.png'     
            io.imsave(os.path.join(args.grayscale_image_dir,name1),sythetic_gaussian_image(im1[i]))
            plt.imsave(os.path.join(args.GT_image_dir,name1),im1[i])

    if args.command == "overlapping_spheres":
    #  p=0.6
     #   r=30
        p=args.porosity
        r=args.radius_size
        im1 = ps.generators.overlapping_spheres(shape=[100,256,256], porosity=p, radius=r)
        noise=sythetic_gaussian_image(im1)

        for i in range(100):
            name1=fileN+'_'+'overlapping_spheres'+str(r)+'_'+'p'+str(p)+str(i)+'.png'     
            io.imsave(os.path.join(args.grayscale_image_dir,name1),sythetic_gaussian_image(im1[i]))
            plt.imsave(os.path.join(args.GT_image_dir,name1),im1[i])
    if args.command == "combine":
  
        p=args.porosity
        r=args.radius_size
        blob=args.blob_size

        a=ps.generators.overlapping_spheres([100,256,256],radius=5, porosity=0.7)
        b=ps.generators.blobs([100,256,256],blobiness=0.5, porosity=0.8)
        im1=np.logical_or(a,b)

        noise=sythetic_gaussian_image(im1)

        for i in range(100):
            name1=fileN+'_'+'combined'+str(r)+'_'+'p'+str(p)+str(i)+'.png'     
            io.imsave(os.path.join(args.grayscale_image_dir,name1),sythetic_gaussian_image(im1[i]))
            plt.imsave(os.path.join(args.GT_image_dir,name1),im1[i])    
