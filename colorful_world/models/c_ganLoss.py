from colorful_world.config import Config
from colorful_world.dataset import DatasetColorBW
from colorful_world.models import Discriminator, Generator

import os
from os.path import isdir, exists, abspath, join
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from colorful_world.models.dataloader import DataLoader as DataLoader2

from torch.autograd import Variable

import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import measure




def calcPer(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    #print(imageA.size)
    #print(imageB.size)
    #print(imageA.shape)
    #print(imageB.shape)
    #print(imageA.dtype)
    #print(imageB.dtype)
    difference = cv2.absdiff(imageA, imageB)
    b, g, r = cv2.split(difference)
    b[b < .1] = 0
    g[g < .1] = 0
    r[r < .1] = 0
    
    total= np.add(np.add(np.array(b),np.array(g)),np.array(r))*1000
    #print (b)
    #print ("\n\n\n")
    #print (np.int64(total))
    #print ("\n\n\n")
    err=100-100*(cv2.countNonZero(np.int64(total)))/total.size
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def mseFun(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

class cGANLoss(object):

    def __init__(self, config: Config):
        self.config = config
        self.data_init()
        self.model_init()
        self.is_trained = False
        self.continueTrain=False
        self.startValue=271
        self.dist="Building"
        self.datatype="blur"
        

    def data_init(self):
        self.training_dataset = DatasetColorBW(self.config.train_dir)
        self.training_data_loader = DataLoader(
            dataset=self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.prediction_dataset = DatasetColorBW(
            self.config.prediction_dir,
            colored=False,
            bw=True
        )

        self.prediction_data_loader = DataLoader(
            dataset=self.prediction_dataset,
            batch_size=1,
            shuffle=False
        )

    def model_init(self):
        self.dis_model = Discriminator(image_size=self.config.image_size)
        self.gen_model = Generator()
        self.optimizer_dis = optim.Adam(self.dis_model.parameters(), lr=self.config.lr_dis)
        self.optimizer_gen = optim.Adam(self.gen_model.parameters(), lr=self.config.lr_gen, weight_decay=0)



        if self.config.use_L1_loss:
            self.L1_loss = nn.SmoothL1Loss()
            self.lambda_L1 = self.config.lambda_L1
        else:
            self.L1_loss = None
            self.lambda_L1 = 0

    # ------------------------------

    def train(self):
        return self.training(
            dis_model=self.dis_model, 
            gen_model=self.gen_model,
            data_loader=self.training_data_loader,
            dis_optimizer=self.optimizer_dis,
            gen_optimizer=self.optimizer_gen,
            n_epochs=self.config.n_epochs,
            L1_loss=self.L1_loss,
            lambda_L1=self.lambda_L1,
            train_on_colab=self.config.train_on_colab
        )

    def training(
            self,
            dis_model: Discriminator, gen_model: Generator,
            data_loader: DataLoader,
            dis_optimizer: torch.optim, gen_optimizer: torch.optim,
            n_epochs: int = 1,
            L1_loss: nn.SmoothL1Loss = None, lambda_L1: float = 1.,
            train_on_colab=False
    ):
        if self.continueTrain==True:
            if self.config.gpu:
                gen_model = torch.load(os.path.join(self.config.model_dir,self.datatype, self.dist, 'gen_model_%s.pk' % str(self.startValue - 1)))
                dis_model = torch.load(os.path.join(self.config.model_dir, self.datatype, self.dist, 'dis_model_%s.pk' % str(self.startValue - 1)))
            else:
                gen_model = torch.load(os.path.join(self.config.model_dir,self.datatype, self.dist, 'gen_model_%s.pk' % str(self.startValue - 1)),map_location='cpu')
                dis_model = torch.load(os.path.join(self.config.model_dir, self.datatype, self.dist, 'dis_model_%s.pk' % str(self.startValue - 1)),map_location='cpu')
            #gen_model.load_state_dict(checkpointGen['model_state_dict'])
            #dis_model.load_state_dict(checkpointDis['model_state_dict'])
            #gen_optimizer.load_state_dict(checkpointGen['optimizer_state_dict'])
            #dis_optimizer.load_state_dict(checkpointDis['optimizer_state_dict'])
            #epoch = checkpoint['epoch']
            #g_loss = checkpointGen['loss']
            #d_loss = checkpointDis['loss']

        print ("hi")
        mse=[]#mean squared error between images
        structSim=[]#structural similularity between images
        perP=[]#percentage of matching pixels
        dlosses=[]
        glosses=[]
        
        tMse=[]#mean squared error between images
        tStructSim=[]#structural similularity between images
        tPerP=[]#percentage of matching pixels

        EPS = 1e-12
        patch = (1, 128 // 2 ** 4, 128 // 2 ** 4)
        criterion_GAN = torch.nn.MSELoss()
        criterion_pixelwise = torch.nn.SmoothL1Loss()
        use_gpu = self.config.gpu
        Tensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        
        if use_gpu:
            torch.cuda.set_device(0)
            dis_model = dis_model.cuda()
            gen_model = gen_model.cuda()
            criterion_GAN.cuda()
            criterion_pixelwise.cuda()            
            if L1_loss:
                L1_loss = L1_loss.cuda()

        dis_model.train(True)
        gen_model.train(True)

        if self.config.save_every_epoch:
            dis_loss = np.zeros(n_epochs)
            gen_loss = np.zeros(n_epochs)
        else:
            dis_loss = []
            gen_loss = []

        if self.config.show_color_evolution:
            dataset_color_bw = DatasetColorBW(self.config.train_dir)
            _, bw_example = dataset_color_bw.generate_data(
                file=self.config.picture_color_evolution,
                img_size=self.config.image_size,
                colored=True,
                bw=True,
            )
            bw_example = bw_example.unsqueeze(0)
            if use_gpu:
                bw_example = bw_example.cuda()

        if train_on_colab:
            from google.colab import drive
            drive.mount('/content/gdrive')

        t = 0
        bw_img=0        
        loader = DataLoader2("data/")
        for epoch_num in range(self.startValue,n_epochs):
            print ("start epoch")
            epoch=epoch_num
            print (epoch)
            loader.setMode('train')
            dis_running_loss = 0.0
            gen_running_loss = 0.0
            size = 0
            step=0
            for i, (img, label) in enumerate(loader):
                #print ("step:" + str(step))
                #print (i)
                #print (epoch)
                #step=step+1
                img_tensor = torch.from_numpy(img).float()
                label_tensor = torch.from_numpy(label).float()

                img_tensor = img_tensor.permute(0,3,1,2) 
                label_tensor = label_tensor.permute(0,3,1,2)
                #label_tensor = torch.squeeze(label_tensor)
                
                clr_img=label_tensor
                bw_img=img_tensor
                if use_gpu:
                    clr_img = clr_img.cuda()
                    bw_img = bw_img.cuda()



                batch_size = clr_img.size(0)
                size += batch_size

                dis_optimizer.zero_grad()
                gen_optimizer.zero_grad()

                #if t % 2 == 1:
                #    dis_model.train(False)
                #    gen_model.train(True)                    
                #    Gx = gen_model.forward(bw_img)  # Generates fake colored images

                #else:
                #    dis_model.train(True)
                #    gen_model.train(False) 
                #    Gx = gen_model.forward(bw_img).detach()  # Detach the generated images for training the discriminator only

                valid = Variable(Tensor(np.ones((clr_img.size(0), *patch))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((clr_img.size(0), *patch))), requires_grad=False)
                
                fake_B = gen_model.forward(bw_img)
                clr_img = Variable(clr_img.type(Tensor))
                pred_fake = dis_model(fake_B, clr_img)
                
                loss_GAN = criterion_GAN(pred_fake, valid)
                loss_pixel = criterion_pixelwise(fake_B, clr_img)
                
                # Total loss
                lambda_pixel = 100
                #print(loss_GAN)
                #print(loss_pixel)
                g_loss = loss_GAN + lambda_pixel * loss_pixel
                #criterion = nn.SmoothL1Loss()
                #g_loss=criterion(fake_B, clr_img)

                # ---------------------
                #  Train Discriminator
                # ---------------------
        
                #optimizer_D.zero_grad()
        
                # Real loss
                pred_real = dis_model(clr_img, clr_img)
                loss_real = criterion_GAN(pred_real, valid)
        
                # Fake loss
                pred_fake = dis_model(fake_B.detach(), clr_img)
                loss_fake = criterion_GAN(pred_fake, fake)
        
                # Total loss
                d_loss = 0.5 * (loss_real + loss_fake)       
                
                dis_running_loss += d_loss.data.cpu().numpy()
                
                d_loss.backward()
                dis_optimizer.step()
                # Run backprop and update the weights of the Discriminator accordingly
                gen_running_loss += g_loss.data.cpu().numpy()
                #if t % 2 == 1:
                #    g_loss.backward()
                #    gen_optimizer.step()
                g_loss.backward()
                gen_optimizer.step()
                
                t+=1
                #Dx = dis_model(clr_img, bw_img)  # Produces probabilities for real images
                #Dg = dis_model(Gx, bw_img)  # Produces probabilities for generator images

                #d_loss = -torch.mean(
                #    torch.log(Dx + EPS) + torch.log(1. - Dg + EPS))  # Loss function of the discriminator.
                #g_loss = - torch.mean(torch.log(Dg + EPS))  # Loss function of the generator.

                #if L1_loss:
                #    g_loss = g_loss + lambda_L1 * L1_loss(Gx, clr_img)

                # Run backprop and update the weights of the Generator accordingly
                #dis_running_loss += d_loss.data.cpu().numpy()
                #if t % 2 == 0:
                #    d_loss.backward()
                #    dis_optimizer.step()

                # Run backprop and update the weights of the Discriminator accordingly
                #gen_running_loss += g_loss.data.cpu().numpy()
                #if t % 2 == 1:
                #    g_loss.backward()
                #    gen_optimizer.step()

                #t += 1

                if not self.config.save_every_epoch:

                    gen_loss.append(g_loss.data.cpu().numpy())
                    dis_loss.append(d_loss.data.cpu().numpy())

                    if t % self.config.save_frequency == 0:
                        torch.save(gen_model, os.path.join(self.config.model_dir,self.datatype, self.dist, f'gen_model_step_{t}.pk'))
                        torch.save(dis_model, os.path.join(self.config.model_dir,self.datatype, self.dist, f'dis_model_step_{t}.pk'))
                        if train_on_colab:
                            torch.save(
                                gen_model,
                                os.path.join("/content/gdrive", "My Drive", "pix2pix", f"gen_model_step_{t}.pk")
                            )
                            torch.save(
                                dis_model,
                                os.path.join("/content/gdrive", "My Drive", "pix2pix", f"dis_model_step_{t}.pk")
                            )
                            with open('gen_loss_lst.pk', 'wb') as f:
                                pickle.dump(gen_loss, f)
                            with open('dis_loss_lst.pk', 'wb') as f:
                                pickle.dump(dis_loss, f)

                        print(f"Saved Model at step {t}")

            
            images=img_tensor.cpu().numpy()[0]
            labels = label_tensor.cpu().numpy()[0]
            predictions = fake_B.cpu().detach().numpy()[0]#Gx.cpu().detach().numpy()[0]
            
            images = np.transpose(images, (1,2,0))
            labels = np.transpose(labels, (1,2,0))
            predictions = np.transpose(predictions, (1,2,0))
            
            
            #show the image
            #plt.imshow(images)
            #plt.show() 
            
            
            #print ((images[0,:,:,:3]).shape)
            index = 0#random.randint(0, 15)
            #print (len(images[0,0,:,:3]))
            #test=images[index,:,:,:]
            #print (test.shape)
            data_dir="data/"
            image = str(epoch + 1) + "train_input_" + ".png"
            print (join(data_dir, 'samples', self.datatype, self.dist, image))
            plt.imsave(join(data_dir,'samples', self.datatype, self.dist,  image), images);
            #plt.imsave(join(data_dir, 'samples', image), images[index,:,:,:3])
            
            image = str(epoch + 1) + "train_groundtruth_" + ".png"
            plt.imsave(join(data_dir,'samples', self.datatype, self.dist,  image), labels);
            print (join(data_dir, 'samples', self.datatype, self.dist, image))
            
            image = str(epoch + 1) + "train_output_" + ".png"
            plt.imsave(join(data_dir, 'samples', self.datatype, self.dist, image), predictions);            
  
            epoch_dis_loss = dis_running_loss / size
            epoch_gen_loss = gen_running_loss / size

            dlosses.append(epoch_dis_loss)
            glosses.append(epoch_gen_loss)
            tMse.append(mseFun(predictions,labels))
            tPerP.append(calcPer(predictions, labels))
            tStructSim.append(measure.compare_ssim(predictions, labels,  multichannel=True))            
            
            


            if self.config.save_every_epoch:
                dis_loss[epoch_num] = epoch_dis_loss
                gen_loss[epoch_num] = epoch_gen_loss

            print('Train - Discriminator Loss: {:.4f} Generator Loss: {:.4f}'.format(epoch_dis_loss, epoch_gen_loss))

            if self.config.show_color_evolution:
                Gx_example = gen_model(bw_example).detach()
                Gx_example_img = Image.fromarray(
                    np.uint8((Gx_example[0].permute(1, 2, 0).cpu().numpy() / 2 + 0.5) * 256)
                )
                Gx_example_img.save(
                    fp=os.path.join(self.config.result_dir, "color_evolution", f"Gx_epoch_{epoch_num}.png"),
                    format="png"
                )

            if epoch_num % self.config.save_frequency == 0 or epoch_num == n_epochs-1:
                torch.save(gen_model, os.path.join(self.config.model_dir,self.datatype, self.dist, f'gen_model_{epoch_num}.pk'))
                torch.save(dis_model, os.path.join(self.config.model_dir,self.datatype, self.dist, f'dis_model_{epoch_num}.pk'))
                if train_on_colab:
                    torch.save(gen_model, os.path.join("/content/gdrive","My Drive","pix2pix", f"gen_model_{epoch_num}.pk"))
                    torch.save(dis_model,os.path.join("/content/gdrive", "My Drive", "pix2pix", f"dis_model_{epoch_num}.pk"))
                print("Saved Model")
                
            sMse=[]#mean squared error between images
            sStructSim=[]#structural similularity between images
            sPerP=[]#percentage of matching pixels
            loader.setMode('test')
            i=0
            while i<20:
                with torch.no_grad():
                    for _, (img, label) in enumerate(loader):
                        img_tensor = torch.from_numpy(img).float()
                        label_tensor = torch.from_numpy(label).float()
    
                        #print(img_tensor.shape)
                        #print(label_tensor.shape)
                        img_tensor = img_tensor.permute(0,3,1,2) 
                        label_tensor = label_tensor.permute(0,3,1,2)
    
                        # todo: load image tensor to gpu
                        if use_gpu:
                            img_tensor = Variable(img_tensor.cuda())
                            label_tensor = Variable(label_tensor.cuda())
                            
                            
                        clr_img=label_tensor
                        bw_img=img_tensor
            
                        # todo: get prediction
                        pred = gen_model.forward(bw_img)
    
                images=img_tensor.cpu().numpy()[0]
                labels = label_tensor.cpu().numpy()[0]
                predictions = pred.cpu().detach().numpy()[0]
            
                images = np.transpose(images, (1,2,0))
                labels = np.transpose(labels, (1,2,0))
                predictions = np.transpose(predictions, (1,2,0))
            
                sMse.append(mseFun(predictions,labels))
                sPerP.append(calcPer(predictions, labels))
                sStructSim.append(measure.compare_ssim(predictions, labels,  multichannel=True))
            
                index = 0#random.randint(0, 15)
                #print (i)
                i=i+1
            #print ("done")
            mse.append(np.mean(sMse))
            perP.append(np.mean(sPerP))
            structSim.append(np.mean(sStructSim))
            
            image = str(epoch + 1) + "test_input_" + ".png"
            print (join(data_dir,'samples',self.datatype, self.dist,  image))
            plt.imsave(join(data_dir,'samples',self.datatype, self.dist,  image), images);
            #plt.imsave(join(data_dir, 'samples', image), images[index,:,:,:3])
            
            image = str(epoch + 1) + "test_groundtruth_" + ".png"
            plt.imsave(join(data_dir,'samples',self.datatype, self.dist,  image), labels);
            print (join(data_dir,'samples',self.datatype, self.dist,  image))
            
            image = str(epoch + 1) + "test_output_" + ".png"
            plt.imsave(join(data_dir,'samples', self.datatype, self.dist,  image), predictions);
            
            if epoch_num % self.config.save_frequency == 0 or epoch_num == n_epochs-1:
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"MSE_%s.csv" % str(self.startValue)), np.array(mse), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"TMSE_%s.csv" % str(self.startValue)), np.array(tMse), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"structSim_%s.csv" % str(self.startValue)), np.array(structSim), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"TstructSim_%s.csv" % str(self.startValue)), np.array(tStructSim), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"perP_%s.csv" % str(self.startValue)), np.array(perP), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"TperP_%s.csv" % str(self.startValue)), np.array(tPerP), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"gloss_%s.csv" % str(self.startValue)), np.array(glosses), delimiter=",")
                np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"dloss_%s.csv" % str(self.startValue)), np.array(dlosses), delimiter=",")
            
        fig1=plt.figure(1)
        plt.plot(mse,label="test")
        plt.plot(tMse,label="train")
        plt.plot(pd.Series(mse).rolling(6).mean(),linestyle="dashed", label="average test")
        plt.plot(pd.Series(tMse).rolling(6).mean(),linestyle="dashed", label="average train")
        plt.ylabel('mean square error')
        plt.xlabel('epoch')
        plt.legend()
        plt.title("Mean Square Error")
        fig1.savefig(os.path.join(self.config.result_dir,self.datatype, self.dist, "MSE.png"), format="png")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"MSE.csv"), np.array(mse), delimiter=",")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"TMSE.csv"), np.array(tMse), delimiter=",")
        
        
        fig2=plt.figure(2)
        plt.plot(structSim,label="test")
        plt.plot(tStructSim,label="train")
        plt.plot(pd.Series(structSim).rolling(6).mean(),linestyle="dashed",label="average test")
        plt.plot(pd.Series(tStructSim).rolling(6).mean(),linestyle="dashed", label="average train")
        plt.ylabel('structural similarity index measure')
        plt.xlabel('epoch')
        plt.legend()
        plt.title("Structural Similarity")
        fig2.savefig(os.path.join(self.config.result_dir,self.datatype, self.dist, "StuctSim.png"), format="png")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"structSim.csv"), np.array(structSim), delimiter=",")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"TstructSim.csv"), np.array(tStructSim), delimiter=",")        
        
        fig3=plt.figure(3)
        plt.plot(perP,label="test")
        plt.plot(tPerP,label="train")
        plt.plot(pd.Series(perP).rolling(6).mean(),linestyle="dashed", label="average test")
        plt.plot(pd.Series(tPerP).rolling(6).mean(),linestyle="dashed", label="average train")
        plt.ylabel('percentage')
        plt.xlabel('epoch')
        plt.title("Percentage of matching pixels")
        plt.legend()
        fig3.savefig(os.path.join(self.config.result_dir,self.datatype, self.dist, "MatchPix.png"), format="png")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"perP.csv"), np.array(perP), delimiter=",")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"TperP.csv"), np.array(tPerP), delimiter=",")
        
        fig4=plt.figure(4)
        plt.plot(glosses,label="Generator Loss")
        plt.plot(dlosses,label="Discriminator Loss")
        plt.plot(pd.Series(glosses).rolling(6).mean(), linestyle="dashed", label="Generator loss average")
        plt.plot(pd.Series(dlosses).rolling(6).mean(),linestyle="dashed", label="Discriminator loss average")
        plt.ylabel('Loss Value')
        plt.xlabel('epoch')
        plt.title("Loss")
        plt.legend()
        plt.show()
        fig4.savefig(os.path.join(self.config.result_dir,self.datatype, self.dist, "Loss.png"), format="png")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"gloss.csv"), np.array(glosses), delimiter=",")
        np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"dloss.csv"), np.array(dlosses), delimiter=",")
        
        if self.config.plot_loss:
            fig = plt.figure(5)
            plt.plot(list(range(n_epochs)), dis_loss, label="discriminator")
            plt.plot(list(range(n_epochs)), gen_loss, label="generator")
            plt.title("Evolution of the Discriminator and Generator loss during the training")
            plt.grid()
            plt.legend(loc='upper right')
            plt.show()
            fig.savefig(os.path.join(self.config.result_dir,self.datatype, self.dist, "loss_graph.png"), format="png")
            np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"genloss.csv"), np.array(gen_loss), delimiter=",")
            np.savetxt(os.path.join(self.config.result_dir,self.datatype, self.dist,"disloss.csv"), np.array(dis_loss), delimiter=",")

        self.is_trained = True

        return cGANLoss

    # ------------------------------

    def predict(self, path_to_model: str = None):
        if path_to_model is not None:
            self.predict_generator = torch.load(
                os.path.join(path_to_model)
            )
        elif not self.is_trained:
            # Load a model with which to make the prediction
            self.predict_generator = torch.load(
                os.path.join(self.config.model_dir,self.datatype, self.dist,'gen_model_%s.pk' % str(self.config.n_epochs - 1))
            )
        else:
            self.predict_generator = self.gen_model

        self.predict_generator.eval()
        return self.predicting(self.predict_generator, self.prediction_data_loader)

    def predicting(self, gen_model, data_loader):
        use_gpu = self.config.gpu
        if use_gpu:
            torch.cuda.set_device(0)

        gen_model.eval()

        imgs = []

        for data in data_loader:
            bw_img = data['bw']

            if use_gpu:
                bw_img = bw_img.cuda()

            fake_img = gen_model(bw_img).detach()

            for i in range(len(fake_img)):
                img_array = fake_img.cpu().numpy()[i].transpose(1, 2, 0)
                img_array = (((img_array / 2.0 + 0.5) * 256).astype('uint8'))
                img = Image.fromarray(img_array)
                imgs.append(img)

        return imgs
