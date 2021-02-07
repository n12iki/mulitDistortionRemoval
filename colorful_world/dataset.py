import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from colorful_world.config import Config


from wand.image import Image as Image2
import numpy as np
import cv2
import io
import random

config = Config()


class DatasetColorBW(Dataset):

    def __init__(self, root_dir: str, colored: bool = True, bw: bool = True):
        self.root_dir = root_dir
        self.image_files = os.listdir(self.root_dir)
        self.colored = colored
        self.bw = bw

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        file = os.path.join(self.root_dir, self.image_files[idx])

        clr, bw = self.generate_data(
            file=file,
            img_size=config.image_size,
            colored=self.colored,
            bw=self.bw
        )

        sample = {'clr': clr, 'bw': bw}

        if not self.colored: sample.pop("clr")
        if not self.bw: sample.pop("bw")

        return sample

    def generate_data(self, file: str, img_size: int, colored: bool, bw: bool):

        img_clr = Image.open(file).convert('RGB')
        img_clr = img_clr.resize((img_size, img_size))
        dist="fish"
        data_image=0
        moveOn=0
        border=0
        nheight=0
        nwidth=0
        while moveOn==0:
            try:
                test_image = img_clr#Image.open(join(img_path,random.choice(files)))
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
                img_clr=test_image
                moveOn=1
                if colored:
                    img_clr_array = np.array(img_clr)
                    # Scale the images to [-1, 1]
                    img_clr_array = ((img_clr_array / 256) - 0.5) * 2.0
                    # tensor shape 3 (channels) x img_size x img_size
                    #print (file)
                    img_clr_tensor = torch.from_numpy(img_clr_array).type(torch.FloatTensor).permute(2, 0, 1)
        
                else:
                    img_clr_tensor = None
        
                if bw:
                    #img_bw = img_clr.convert('L')
                    if (dist=="fish"):
                        with Image2(filename=file) as img:
                            #print(img.size)
                            img.virtual_pixel = 'transparent'
                            scale_percent = 220 # percent of original size
                            width=int(img.width* scale_percent/100)
                            height=int(img.height * scale_percent/100)
                            img.sample(width,height)
                            #border= random.randint(128,width)
                            #nheight= random.randint(0,height-border)
                            #nwidth= random.randint(0,width-border)
                            img.crop(nwidth,nheight,nwidth+border,nheight+border)
                            img.sample(128,128)
                            #test_image = Image.fromarray(np.array(img), 'RGB')
                            #test_image = Image.open(io.BytesIO(img.make_blob("png"))).convert('RGB')
                                
                            a= random.randint(0,75)/100.0
                            b=random.randint(0,75)/100.0
                            c=random.randint(0,75)/100.0
                            d=random.randint(0,75)/100.0
                            img.distort('barrel', (a, b, c, d))
                            #data_image= Image.fromarray(np.array(img), 'RGB')
                            #img.virtual_pixel = 'black'
                            data_image = Image.open(io.BytesIO(img.make_blob("png"))).convert('RGB')
                            #print(data_image.size)
                    img_bw_array = np.array(data_image)
                    # Scale the images to [-1, 1]
                    img_bw_array = ((img_bw_array / 256) - 0.5) * 2.0
                    # tensor shape 1 (channel) x img_size x img_size
                    img_bw_tensor = torch.from_numpy(img_bw_array).type(torch.FloatTensor).permute(2, 0, 1)
                    
        
                else:
                    img_bw_tensor = None
                
                #img_bw_tensor=img_clr_tensor
                
            except:
            #except AssertionError as error:
            #    print (error)
                pass


        return img_clr_tensor, img_bw_tensor
