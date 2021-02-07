import os
from os.path import isdir, exists, abspath, join
import cv2 
from colorful_world.models import motionblur
from colorful_world.models.motionblur import Kernel

import random

import numpy as np
from PIL import Image, ImageFilter
import glob

from torchvision import transforms

import numpy as np
import time
import random

from wand.image import Image as Image2
import numpy as np
import cv2
import io

class DataLoader():
    
    def __init__(self, root_dir='data', batch_size=1, no_epochs = 1):
        self.batch_size = batch_size
        self.no_epochs = no_epochs
        
        self.root_dir = abspath(root_dir)
        #self.train_img_path = join(self.root_dir, '/Users/n12i/Desktop/masterThesis/image_inpainting_resnet_unet-master/image_inpainting_resnet_unet-master/inpainting_set_UNET/data/lfw2')
        #self.test_img_path = join(self.root_dir, '/Users/n12i/Desktop/masterThesis/image_inpainting_resnet_unet-master/image_inpainting_resnet_unet-master/inpainting_set_UNET/data/lfw2')
        #self.train_img_path = join(self.root_dir, '/Users/n12i/Desktop/ProjectDataSet/ADE20K_new')
        #self.test_img_path = join(self.root_dir, '/Users/n12i/Desktop/ProjectDataSet/ADE20K_new')
        self.test_img_path = join(self.root_dir, '../../../Building/results')
        self.train_img_path = join(self.root_dir, '../../../Building')
    def __iter__(self):

        if self.mode == 'train':
            no_epochs = 600
            img_path = self.train_img_path
            crop_width = 450
            endId = no_epochs

        elif self.mode == 'test':
            no_epochs = 1
            img_path = self.test_img_path
            crop_width = 30
            endId = no_epochs

        current = 0
        dist="blur"
        reset=0
        while current < endId:
            #print (img_path)
            stop=0
            files = os.listdir(img_path)
            for i in files:
                if not(i.endswith('.jpg')):
                    files.remove(i)
            moveOn=0
            
            
            files2 = os.listdir(join(self.root_dir, '../../../flares'))
            for i in files2:
                if not(i.endswith('.jpg')):
                    files2.remove(i)
            
            
            kSize= random.randint(15,45)
            kInt= random.randint(2,10)/10.0
            # Initialise Kernel
            kernel = Kernel(size=(kSize, kSize), intensity=kInt)

            # Display kernel
            #kernel.displayKernel()

            # Get kernel as numpy array
            kernel.kernelMatrix
            
            while moveOn==0:
                try:
                    if(dist=="rand"):
                        distList=["blur","lens","fish"]
                        dist=random.choice(distList)
                        reset=1
                    file=random.choice(files)
                    #print (join(img_path,file))
                    #print (join('C:/Users/n12i/Desktop/ProjectDataSet/ADE20K_motionblur',file))
                    if (dist=="blur"):
                        test_image = Image.open(join(img_path,file)).convert('RGB')#Image.open(join(img_path,random.choice(files)))
                        width, height = test_image.size
                        scale_percent = 220 # percent of original size
                        width = int(width * scale_percent / 100)
                        height = int(height * scale_percent / 100)
                        dim = (width, height)
                        # resize image
                    
                        test_image = test_image.resize(dim)
                        border= random.randint(128,width)
                        nheight= random.randint(0,height-border)
                        nwidth= random.randint(0,width-border)
                        test_image=test_image.crop((nwidth,nheight,nwidth+border,nheight+border))#test_image[width:width+128,height:height+128]
                        test_image = test_image.resize((128,128),Image.ANTIALIAS)
                        #data_image = Image.open(join('C:/Users/n12i/Desktop/ProjectDataSet/ADE20K_motionblur',file)).convert('RGB')
                        data_image = kernel.applyTo(test_image, keep_image_dim=True)
                    
                    if (dist=="fish"):
                        with Image2(filename=join(img_path,file)) as img:
                            #print(img.size)
                            img.virtual_pixel = 'transparent'
                            scale_percent = 220 # percent of original size
                            width=int(img.width* scale_percent/100)
                            height=int(img.height * scale_percent/100)
                            img.sample(width,height)
                            border= random.randint(128,width)
                            nheight= random.randint(0,height-border)
                            nwidth= random.randint(0,width-border)
                            img.crop(nwidth,nheight,nwidth+border,nheight+border)
                            img.sample(128,128)
                            #test_image = Image.fromarray(np.array(img), 'RGB')
                            test_image = Image.open(io.BytesIO(img.make_blob("png"))).convert('RGB')
                           
                            a= random.randint(0,75)/100.0
                            b=random.randint(0,75)/100.0
                            c=random.randint(0,75)/100.0
                            d=random.randint(0,75)/100.0
                            img.distort('barrel', (a, b, c, d))
                            #data_image= Image.fromarray(np.array(img), 'RGB')
                            data_image = Image.open(io.BytesIO(img.make_blob("png"))).convert('RGB')
                    
                    if (dist=="lens"):
                        B=random.choice(files2)
                        test_image = Image.open(join(img_path,file)).convert('RGB')
                        width, height = test_image.size
                        scale_percent = 220 # percent of original size
                        width = int(width * scale_percent / 100)
                        height = int(height * scale_percent / 100)
                        dim = (width, height)
                        # resize image
                    
                        test_image = test_image.resize(dim)
                        border= random.randint(128,width)
                        nheight= random.randint(0,height-border)
                        nwidth= random.randint(0,width-border)
                        test_image=test_image.crop((nwidth,nheight,nwidth+border,nheight+border))#test_image[width:width+128,height:height+128]
                        test_image = test_image.resize((128,128),Image.ANTIALIAS)                        
                        flip= random.randint(-1,1)
                        B = Image.open(join(self.root_dir, '../../../flares',B)).convert('RGB')
                        if random.random() > 0.4 or random.random() < 0.1:
                            hue_factor = random.randint(1, 5) / 10
                            B = transforms.functional.adjust_hue(B, hue_factor)
        
                            gamma = random.randint(10, 12) / 10
                            B = transforms.functional.adjust_gamma(B,  gamma, gain=1)
    
                        if random.random() > 0.55 or random.random()<0.3:
                            B = transforms.functional.hflip(B)
                        if random.random() > 0.8 or random.random() < 0.2:
                            B = transforms.functional.vflip(B)
    
    
                        height, width = test_image.size
    
                        dim = (width, height)
    
    
                        resized = B.resize((height,width),Image.ANTIALIAS)
                        strength= random.randint(20,60)/100
                        data_image = Image.blend(test_image, resized, strength)#(A, 1-strength, resized, strength, 0,resized) 
                           
                    data_image = data_image.resize((128,128),Image.ANTIALIAS)
                    moveOn=1
                    
                    if reset==1:
                        reset=0
                        dist="rand"
                    
                except:
                #except AssertionError as error:
                #    print (error)
                    pass

            #dim = (width, height)
            # resize image
            #test_image = test_image.resize(dim)            
            #data_image = data_image.resize(dim)
            #test_image = test_image.resize((350,350))
            #data_image = data_image.resize((350,350))
            data_list = list()
            gt_list = list()

            data_list.clear()
            gt_list.clear()
            for i in range(self.batch_size):
                
                #crop_width = 350
                #crop_height = 350
                #crop_size = 350
                #crop_x = 0 #random.randint(0,test_image.height - crop_height)
                #crop_y = 0 #random.randint(0,test_image.width - crop_width)

                #data_image = transforms.functional.resized_crop(data_image, crop_x, crop_y, crop_width, crop_height, crop_size, Image.BILINEAR)
                #test_image = transforms.functional.resized_crop(test_image, crop_x, crop_y, crop_width, crop_height, crop_size, Image.BILINEAR)
                if self.mode == 'train':
                    if random.random() > 0.4 or random.random() < 0.1:
                        
                        hue_factor = random.randint(1, 5) / 10
                        data_image = transforms.functional.adjust_hue(data_image, hue_factor)
                        test_image = transforms.functional.adjust_hue(test_image, hue_factor)

                        gamma = random.randint(10, 12) / 10
                        data_image = transforms.functional.adjust_gamma(data_image, gamma, gain=1)
                        test_image = transforms.functional.adjust_gamma(test_image,  gamma, gain=1)
                    
                    if random.random() > 0.55 or random.random()<0.3:                      
                        data_image = transforms.functional.hflip(data_image)
                        test_image = transforms.functional.hflip(test_image)
                    if random.random() > 0.8 or random.random() < 0.2:
                        test_image = transforms.functional.vflip(test_image)
                        data_image = transforms.functional.vflip(data_image)

                    #if random.random() > 0.5:
                        #data_image = transforms.functional.rotate(data_image, random.randint(-45, 45))
                        #data_image = np.asarray(transforms.functional.five_crop(data_image, 150)[4])
                        #data_image = Image.fromarray(data_image)

                    #if random.random() > 0.5:
                        #rValue=random.randint(200,500)
                        #data_image = transforms.functional.resize(data_image, rValue)
                        #test_image = transforms.functional.resize(test_image, rValue) 


                gt_image = np.array(test_image)
                #mask = self.generateRandomMask (128, 64, 8)
                
                #mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                #data_image = cv2.blur(data_image,(10,10))#(data_image * (mask / 255))
                #data_image = data_image.filter(ImageFilter.GaussianBlur(radius = 5))

                #temp_mask = mask[:, :, 0]
                #temp_mask = np.expand_dims(temp_mask, axis=2)

                # making the data_image and mask of appropriate size
                data_image= np.array(data_image)
                
                # append data image and mask to list
                data_list.append(data_image)
                gt_list.append(gt_image)

            data_list = np.array(data_list)
            gt_list = np.array(gt_list)

            data_list = data_list / 255
            gt_list = gt_list / 255
            
            current += 1

            yield (data_list, gt_list)

    def generateRandomMask (self, size, max_rec_width, max_rec_height):

        mask = np.full((size, size), 255)

        for i in range(5):

            if random.random() > 0.5:
                rec_x = random.randint(0,size - max_rec_width)
                rec_y = random.randint(0,size - max_rec_height)
                mask[rec_x:rec_x+max_rec_width, rec_y:rec_y+max_rec_height] = 0

            else:
                rec_x = random.randint(0,size - max_rec_height)
                rec_y = random.randint(0,size - max_rec_width)
                mask[rec_x:rec_x+max_rec_height, rec_y:rec_y+max_rec_width] = 0

        return mask

    def setMode(self, mode):
        self.mode = mode

    # def n_train(self):
    #     data_length = len(self.data_files)
    #     return np.int_(data_length - np.floor(data_length * self.test_percent))
        
