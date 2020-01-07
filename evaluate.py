import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.filters import threshold_otsu
import pandas as pd
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='passing')

    parser.add_argument('--image_gt_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training dataset')


    parser.add_argument('--image_pr_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the gt dataset')
    args = parser.parse_args()      

## accuracy for bread 

    a1=[]
    a2=[]
    b1=[]
    b2=[]
    c=[]
    d=[]
    image_names=os.listdir(args.image_gt_dir)
    for i in range(len(image_names)):
    

        gt_image = io.imread(os.path.join(args.image_gt_dir,image_names[i]))
        pr_mask = io.imread(os.path.join(args.image_pr_dir,image_names[i]))
        pr_mask=(pr_mask<50)*255
    
        predic_acc_0=np.logical_and(pr_mask==0,gt_image==0).sum()/(gt_image==0).sum()
        predic_acc_255=np.logical_and(pr_mask==255,gt_image==255).sum()/(gt_image==255).sum()
    
        overall_acc=(pr_mask==gt_image).sum()/gt_image.size
        b1.append(predic_acc_0)# pore_bread unet
        b2.append(predic_acc_255)# pore_bread unet
        c.append(overall_acc)# overall unet
        #plt.imsave(merge_image_mask(image,pr_mask))
        #plt.imsave(merge_image_mask(image,(pr_mask==0)*255))

        
        b1_f=pd.DataFrame(b1)
        b2_f=pd.DataFrame(b2)
        frames = [ b1_f, b2_f]

        result = pd.concat(frames,axis=1, keys=['Unet 0','Unet 255'])
        result.to_csv('result.csv')