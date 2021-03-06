
# coding: utf-8

# In[ ]:


import segmentation_models as sm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_int, img_as_ubyte
from skimage.filters import threshold_otsu
from keras import backend as K

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.util import img_as_ubyte
from PIL import Image
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import gaussian_filter


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def combine_histogram(image, ground_truth, classe_names,class_vlues, color_list):
    
    alpa_value=0.3
    for i in range(len(class_vlues)):
        classe_image=(ground_truth==class_vlues[i])*image
        plt.hist(classe_image.flatten(),256, [1, 255],color=color_list[i],alpha=alpa_value,label=classe_names[i])
        alpa_value+=0.1
    plt.legend(loc='upper right')

def merge_image_mask(image,predicted):
    
# this used to visualised two images; original image and semgneted image (mask). 
    mask=np.zeros((image.shape[0],image.shape[1],3))
    mask[:,:,0]=predicted
    mask=np.array(mask,dtype=np.int8)
    image_pil=Image.fromarray(image)
    mask_pil=Image.fromarray(mask,'RGB')
    
    image_pil_RGBA = image_pil.convert("RGBA")
    mask_pil_RGBA = mask_pil.convert("RGBA")
    
    alphaBlended1 = Image.blend(image_pil_RGBA, mask_pil_RGBA, alpha=.1)
    return np.array(alphaBlended1)

def merge_image_mask_ground_truth(image,predicted,ground_truth):

# this used to visualise the orginal image and segmented image and ground truth image. 

    # Yellow is where it is correctly predicted
    #red is where it is not predicted
    # green is where is wrongly predicted
    
    # summary red is undersegmentation , green is oversegmentation
    mask=np.zeros((image.shape[0],image.shape[1],3))
    mask[:,:,0]=ground_truth
    mask[:,:,1]=predicted
    mask=np.array(mask,dtype=np.int8)
    image_pil=Image.fromarray(image)
    mask_pil=Image.fromarray(mask,'RGB')
    
    image_pil_RGBA = image_pil.convert("RGBA")
    mask_pil_RGBA = mask_pil.convert("RGBA")
    
    alphaBlended1 = Image.blend(image_pil_RGBA, mask_pil_RGBA, alpha=.15)
    return np.array(alphaBlended1)

    # train model
def train(model,train_dataloader,valid_dataloader,epoch,model_dir):
    history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=epoch, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),)
    model.save_weights(model_dir)
    
def predict(model,x_test_dir,y_test_dir,model_dir):
    model.load_weights(model_dir)
    names=os.listdir(x_test_dir)
  
    for i in range(len(names)):
        image=cv2.imread(os.path.join(x_test_dir,names[i]))[:192,:192]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).squeeze()
        pr_mask=img_as_ubyte(pr_mask)
        #pr_mask=img_as_ubyte(pr_mask<50)

        #pr_mask=img_as_ubyte(pr_mask)


        io.imsave(os.path.join(y_test_dir,names[i]),pr_mask)

# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
  
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            class_values=None,
            start=None,
            end=None,
            row=None,
            column=None,
            ini_val=None
        
        
        
    ):
        self.ids = os.listdir(images_dir)#[start: end]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = class_values#[self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.row=row
        self.column=column
        self.ini_val=ini_val
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image =image[self.ini_val:self.row+self.ini_val,self.ini_val:self.column+self.ini_val] 
        image=gaussian_filter((image), sigma=3)


       # image=image[580:-579,580:-579,:]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = mask[self.ini_val:self.row+self.ini_val,self.ini_val:self.column+self.ini_val] 
        mask=img_as_ubyte(mask<50)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
       
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

   

      
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='passing')

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'predict'")
    parser.add_argument('--epoch', type=int,required=False,)
    parser.add_argument('--BATCH_SIZE', type=int,required=False,)
    parser.add_argument('--LR', type=float,required=False,)
    parser.add_argument('--loss', metavar="loss",required=False,)

    parser.add_argument('--model', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training dataset')
    parser.add_argument('--optimiser', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training dataset')


    parser.add_argument('--train_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training dataset')


    parser.add_argument('--valid_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the validation dataset')


    parser.add_argument('--test_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the test dataset')
    

    parser.add_argument('--model_dir', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the training weight')

    args = parser.parse_args()      
  
   



    BACKBONE = 'vgg16'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    LR = 0.07
    LR=args.LR
    optim = keras.optimizers.SGD(LR)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


    CLASSES=['pore']

    BATCH_SIZE = 8
    BATCH_SIZE = args.BATCH_SIZE

    EPOCHS = 10




 




    #keras.backend.set_image_data_format('channels_last')
    keras.backend.set_image_data_format('channels_last')
    if(args.model=='unet'):
          model = sm.Unet(BACKBONE, encoder_weights='imagenet',classes=1,input_shape=(192, 192, 3),activation='sigmoid')
    if(args.model=='Linknet'):
          model = sm.Linknet(BACKBONE, encoder_weights='imagenet',classes=1,input_shape=(192, 192, 3),activation='sigmoid')

    callbacks = [
        keras.callbacks.ModelCheckpoint('./best_model_1.h5', save_weights_only=True, save_best_only=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(),
    ]


    # define optomizer
    if(args.optimiser=='Adam'):    
        optim = keras.optimizers.Adam(LR)
    if(args.optimiser=='Adagrad'): 
       optim = keras.optimizers.Adagrad(LR)
    if(args.optimiser=='SGD'):    
       optim = keras.optimizers.SGD(LR)



    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
 
    #optim = keras.optimizers.Adam(LR)
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5),'accuracy']
    if(args.loss=='binary_crossentropy'):
        LOSS=keras.losses.binary_crossentropy
    if(args.loss=='binary_focal_dice_loss'):
        dice_loss = sm.losses.DiceLoss()
        focal_loss = sm.losses.BinaryFocalLoss() 
        total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
        LOSS = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

        
    # model.compile(
    #     optim,
    #     loss=keras.losses.binary_crossentropy,
    #     metrics=metrics,
    # )

    model.compile(
    optim,
    loss=LOSS,
    metrics=metrics,
    )
    if args.command == "train":
        x_train_dir = os.path.join(args.train_dir,'input')
        y_train_dir = os.path.join(args.train_dir,'output')

        x_valid_dir = os.path.join(args.valid_dir,'input')
        y_valid_dir = os.path.join(args.valid_dir,'output')

    # Dataset for train images
        train_dataset = Dataset(x_train_dir, 
        y_train_dir, 
        classes=CLASSES, class_values=[0], row=192,column=192,ini_val=0)

    # Dataset for validation images
        valid_dataset = Dataset(x_valid_dir,y_valid_dir, 
        classes=CLASSES, class_values=[0], row=192,column=192,ini_val=0)
        train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = Dataloder(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
        
        
        train(model,train_dataloader,valid_dataloader,args.epoch,args.model_dir)
    elif args.command == "predict":
        x_test_dir = os.path.join(args.test_dir,'input')
        y_test_dir = os.path.join(args.test_dir,'output')
        test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
        classes=CLASSES,  class_values=[0],  row=192,column=192,ini_val=0
    )

        test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)


        predict(model,x_test_dir,y_test_dir,args.model_dir)



# %%
