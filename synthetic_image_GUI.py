from tkinter import *
from tkinter import ttk
from tkinter import filedialog
#import config
from PIL import Image
import PIL
import numpy as np
from PIL import ImageTk
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve)



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

blob_btn=0
overlapp_spehers_btn=0

def sythetic_gaussian_image(canvas):
    
    curr = config.current_image.load()
    im=np.array(curr)

    
    thresholded_image=im


    inveted_image=np.where((thresholded_image==0.0), np.random.random_integers(30,40,1),thresholded_image)
    inveted_image=np.where((inveted_image==255.0),np.random.random_integers(100,130,1),inveted_image)
    noise=np.random.normal(size=inveted_image.shape)
    noise=img_as_ubyte(noise/np.maximum(np.absolute(noise.min()),noise.max()))
    blu=gaussian_filter(inveted_image-noise, sigma=5)
    result_3 = unsharp_mask(blu, radius=30, amount=3)

    sharped_image=img_as_ubyte(result_3/result_3.max())
    blu2=gaussian_filter((sharped_image), sigma=1)
    
    new_img = Image.fromarray(blu2)
    
    tk_img = ImageTk.PhotoImage(new_img)
    canvas.create_image(400, 300, image=tk_img)     # (xpos, ypos, imgsrc)
    canvas.image = tk_img
    config.current_image = new_img 
    

def config():
    current_image = None


def Blob(canvas , f):
    global blob_btn
    global overlapp_spehers_btn
    blob_btn=1
    overlapp_spehers_btn=0

        
    img = np.invert(ps.generators.blobs(shape=[256,256], porosity=f[0], blobiness=f[1]))
    #noise=sythetic_gaussian_image(img)
      
    new_img = Image.fromarray(img)
    
    tk_img = ImageTk.PhotoImage(new_img)
    canvas.create_image(400, 300, image=tk_img)     # (xpos, ypos, imgsrc)
    canvas.image = tk_img
    config.current_image = new_img 
    
    

def Overlapping_sepheres(canvas , f):
    global blob_btn
    global overlapp_spehers_btn
    blob_btn=0
    overlapp_spehers_btn=1
 
    img = np.invert(ps.generators.overlapping_spheres(shape=[256,256], porosity=f[0], radius=f[1]))

    #noise=sythetic_gaussian_image(img)
      
    new_img = Image.fromarray(img)
    
    tk_img = ImageTk.PhotoImage(new_img)
    canvas.create_image(400, 300, image=tk_img)     # (xpos, ypos, imgsrc)
    canvas.image = tk_img
    config.current_image = new_img 
    
    
   
def SaveImage(f):
    global saved_file_name
   # saved_file_name = filedialog.askdirectory()   
    seperator = ''
    path=('python3 synthetic_generator.py overlapping_spheres --porosity=',str(f[0]),' --size=',str(f[1]),' --fileName=','"first"',' ','--three_3D_dir=','"synthetic2/3d"',' ','--grayscale_image_dir=','"synthetic2/input"',' ','--GT_image_dir=','"synthetic2/output"')
    
    os.system(seperator.join(path))

    

def loadImage():
    filename = filedialog.askopenfilename()

    img = PIL.Image.open(filename)
    tk_img = ImageTk.PhotoImage(img)
    canvas.create_image(400, 300, image=tk_img)     # (xpos, ypos, imgsrc)
    canvas.image = tk_img		# Keep reference to PhotoImage so Python's garbage collector
                                # does not get rid of it making the image dissapear
    config.current_image = img

root = Tk()
root.title("Synthetic Porous Image Generator")

mainframe = ttk.Frame(root,padding="3 3 12 12")
mainframe.grid(row=0, column=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

canvas = Canvas(mainframe, width=800, height=600)
canvas.grid(row=0, column=0, rowspan=3)		# put in row 0 col 0 and span 2 rows
canvas.columnconfigure(0, weight=3)

buttons_frame = ttk.Frame(mainframe)
buttons_frame.grid(row=0, column=1, sticky=N)

load_image_button = ttk.Button(buttons_frame, text="Load Image...", command=loadImage)
load_image_button.grid(row=0, column=1, sticky=N+W+E, columnspan=2, pady=10)
#rotate_button_label = ttk.Label(buttons_frame, text="Rotation").grid(row=2, column=1, columnspan = 2, sticky=S, pady=8)



ttk.Label(buttons_frame, text="paramters").grid(row=14, column=1)

#matrix_frame = ttk.Frame(buttons_frame)
#matrix_frame.grid(row=19, column=1, columnspan=2)

ttk.Label(buttons_frame, text="porosity").grid(row=15, column=1,columnspan=2)

a11 = ttk.Entry(buttons_frame, width=3, justify=CENTER)
a11.grid(row=15, column=2)

ttk.Label(buttons_frame, text="size").grid(row=16, column=1,columnspan=2)

a12 = ttk.Entry(buttons_frame, width=3, justify=CENTER)
a12.grid(row=16, column=2)

# filter_button = ttk.Button(buttons_frame, text="Filter",
#                            command=lambda: filter(canvas, [ [float(a11.get()), float(a12.get()), float(a13.get())],
#                                                             [float(a21.get()), float(a22.get()), float(a23.get())],
#                                                             [float(a31.get()), float(a32.get()), float(a33.get())] ] ) )

filter_button = ttk.Button(buttons_frame, text="Blob Shape Generator",
                           command=lambda: Blob(canvas , [float(a11.get()),float(a12.get())]) )


filter_button.grid(row=20, column=1, columnspan=2, sticky=(W, E))

filter_button = ttk.Button(buttons_frame, text="Overlapping Sepheres Generator",
                           command=lambda: Overlapping_sepheres(canvas , [float(a11.get()),float(a12.get())]) )


filter_button.grid(row=21, column=1, columnspan=2, sticky=(W, E))

rotate_left_button = ttk.Button(buttons_frame, text="Generate Grayscale Image", command=lambda: sythetic_gaussian_image(canvas))
rotate_left_button.grid(row=22, column=1, sticky=N+W+E, columnspan=2)

save_button = ttk.Button(buttons_frame, text="Save", command=lambda: SaveImage([float(a11.get()),float(a12.get())]))
save_button.grid(row=23, column=1, sticky=N+W+E, columnspan=2)


root.mainloop()

