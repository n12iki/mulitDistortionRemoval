import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):  #Unet
    def __init__(self):
        super(Generator, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.down1 = downStep(3, 64) #downStep(4, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down4 = downStep(256, 512)
        self.down5 = downStep(512, 1024)
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64, withReLU = False)
        #self.outputConv = nn.Conv2d(64, n_classes, kernel_size = 1)
        self.maxpool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        # todo
        #print(x.shape)
        x1 = self.down1(x)
        x2 = self.maxpool2d(x1)
        #print("x1 ", x1.shape)
        x3 = self.down2(x2)
        x4 = self.maxpool2d(x3)
        #print("x3", x3.shape)
        x5 = self.down3(x4)
        x6 = self.maxpool2d(x5)
        #print("x5", x5.shape)
        x7 = self.down4(x6)
        x8 = self.maxpool2d(x7)
        #print("x7", x7.shape)
        x9 = self.down5(x8)
        #print("x9", x9.shape)

        x = self.up1(x9, x7)
        #print("x10", x.shape)
        x = self.up2(x, x5)
        #print("x11", x.shape)
        x = self.up3(x, x3)
        #print("x12", x.shape)
        x = self.up4(x, x1)
        #print("x13", x.shape)
        #x = self.outputConv(x)
        #print(x.shape)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.convLayer1 = nn.Conv2d(inC, outC, kernel_size = 3, padding= 1)
        self.convLayer2 = nn.Conv2d(outC, outC, kernel_size = 3, padding= 1)
        #self.batchNorm = nn.BatchNorm2d(outC)
        self.relU = nn.ReLU(inplace=True)

    def forward(self, x):
        # todo
        x = self.convLayer1(x)
        x = self.relU(x)
        #x = self.batchNorm(x)
        x = self.convLayer2(x)
        x = self.relU(x)
        #x = self.batchNorm(x)
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        #self.up = nn.ConvTranspose2d(inC, inC//2, 2, stride=2)
        self.convLayer1 = nn.Conv2d(inC, outC, kernel_size = 3, padding= 1)
        self.convLayer2 = nn.Conv2d(outC, outC, kernel_size = 3, padding= 1)
        self.convLayer3 = nn.Conv2d(outC, 3, kernel_size = 3, padding= 1)
        self.deconvLayer1 = nn.ConvTranspose2d(inC, outC , kernel_size = 2, stride=2)
        self.sig = nn.Sigmoid()
        self.relU = nn.ReLU(inplace=True)
        self.withReLU = withReLU

    def forward(self, x, x_down):
        # todo
        x = self.deconvLayer1(x)

        # _, _, x_down_height, x_down_width = x_down.size()
        # diff_y = (x_down_height - x.shape[2:][0]) // 2
        # diff_x = (x_down_width - x.shape[2:][1]) // 2
        # crop = x_down[:, :, diff_y:(diff_y + x.shape[2:][0]), diff_x:(diff_x + x.shape[2:][1])]

        x = torch.cat([x, x_down], 1)
        x = self.convLayer1(x)
        if self.withReLU:
            x = self.relU(x)
            #x = self.batchNorm(x)
        x = self.convLayer2(x)
        if self.withReLU:
            x = self.relU(x)
            #x = self.batchNorm(x)
        else:
            x = self.convLayer3(x)
            x = self.sig(x)
        return x


"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class resBlock(nn.Module): #Resnet
    def __init__(self, inchannel, outchannel, downsample=False, upsample=False):
        super(resBlock, self).__init__()

        self.inchannel = inchannel
        self.outchannel = outchannel
        self.downsample = downsample
        self.upsample = upsample

        self.reflectpad1 = nn.ReflectionPad2d(1)

        if (self.downsample == True):
            self.down = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(inchannel,outchannel,kernel_size =3,stride=2,padding=0,bias=False),
                                      nn.BatchNorm2d(outchannel),
                                      nn.ReLU(),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(outchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                                      nn.BatchNorm2d(outchannel))
            self.skipDown = nn.Conv2d(inchannel,outchannel,kernel_size =1,stride=2,padding=0,bias=False)


        if (self.downsample == False and self.upsample == False):
            self.standard = nn.Sequential(nn.ReflectionPad2d(1),
                            nn.Conv2d(inchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                            nn.BatchNorm2d(outchannel),
                            nn.ReLU(),
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(outchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                            nn.BatchNorm2d(outchannel))


        if (self.upsample == True):
            self.up = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.ConvTranspose2d(inchannel,outchannel,kernel_size =3,stride=2,padding=3,output_padding=1,bias=False),
                                    nn.BatchNorm2d(outchannel),
                                    nn.ReLU(),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(outchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                                    nn.BatchNorm2d(outchannel))
            self.skipUp = nn.Conv2d(inchannel,outchannel,kernel_size =1,stride=1,padding=0,bias=False)






    def forward(self, x):

        if (self.downsample == True):
            x1 = self.down(x)
            x1 += self.skipDown(x)


        if (self.downsample == False and self.upsample == False):
            x1 = self.standard(x)
            x1 += x


        if (self.upsample == True):
            x1 = self.up(x)
            x2 = self.skipUp(x)
            x1 += F.interpolate(x2,scale_factor=2)


        return F.relu(x1)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.beforeRes = nn.Sequential(nn.ReflectionPad2d(3),
                         nn.Conv2d(3,32,kernel_size =7,stride=1,padding=0,bias=False), #nn.Conv2d(4,32,kernel_size =7,stride=1,padding=0,bias=False),
                         nn.ReLU(),
                         nn.ReflectionPad2d(2),
                         nn.Conv2d(32,64,kernel_size =5,stride=2,padding=0,bias=False),
                         nn.ReLU())

        self.block0 = resBlock(64,128,downsample=True,upsample=False)

        self.block1 = nn.ModuleList([resBlock(128,128,downsample=False,upsample=False ) for i in range(6)])

        self.block2 = resBlock(128,64,downsample=False,upsample=True)

        self.afterRes = nn.Sequential(nn.ReflectionPad2d(1),
                        nn.ConvTranspose2d(64,32,kernel_size =5,stride=2,padding=4,output_padding=1,bias=False),
                        nn.ReLU(),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(32,3,kernel_size =7,stride=1,padding=0,bias=True),
                        nn.Sigmoid())


    def forward(self, x):

        x1 = self.beforeRes(x)

        x1 = self.block0(x1)

        for i in range(6):
            x1 = self.block1[i](x1)

        x1 = self.block2(x1)

        x1 = self.afterRes(x1)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # --------- Encoder ---------
        # *bs = batch size
        # input = bs x 256 x 256 x 1  / output = bs x 128 x 128 x 64
        self.encod1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=1, stride=2),
        )
        # input = bs x 128 x 128 x 64  / output = bs x 64 x 64 x 128
        self.encod2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(128),
        )
        # input = bs x 64 x 64 x 128 / output = bs x 32 x 32 x 256
        self.encod3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(256),
        )
        # input = bs x 32 x 32 x 256 / output = bs x 16 x 16 x 512
        self.encod4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 16 x 16 x 512 / output = bs x 8 x 8 x 512
        self.encod5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 8 x 8 x 512 / output = bs x 4 x 4 x 512
        self.encod6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 4 x 4 x 512 / output = bs x 2 x 2 x 512
        self.encod7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 2 x 2 x 512 / output = bs x 1 x 1 x 512
        self.encod8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
        )

        # --------- Decoder ---------
        # input = bs x 1 x 1 x 512 / output = bs x 2 x 2 x 512
        self.decod8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 2 x 2 x 2*512 / output = bs x 4 x 4 x 512
        self.decod7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 4 x 4 x 2*512 / output = bs x 8 x 8 x 512
        self.decod6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 8 x 8 x 2*512 / output = bs x 16 x 16 x 512
        self.decod5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=512, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(512),
        )
        # input = bs x 16 x 16 x 2*512 / output = bs x 32 x 32 x 256
        self.decod4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 512, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(256),
        )
        # input = bs x 32 x 32 x 2*256 / output = bs x 64 x 64 x 128
        self.decod3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 256, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(128),
        )
        # input = bs x 32 x 32 x 2*128 / output = bs x 128 x 128 x 64
        self.decod2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 128, out_channels=64, kernel_size=4, padding=1, stride=2),
            nn.ELU(),
            # nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),
        )
        # input = bs x 128 x 128 x 2*64 / output = bs x 256 x 256 x 3
        self.decodout = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2 * 64, out_channels=3, kernel_size=4, padding=1, stride=2),
            nn.Tanh())

    def forward(self, x: torch.Tensor):
        # --------- Encoder ---------
        e1 = self.encod1(x)
        e2 = self.encod2(e1)
        e3 = self.encod3(e2)
        e4 = self.encod4(e3)
        e5 = self.encod5(e4)
        e6 = self.encod6(e5)
        e7 = self.encod7(e6)
        e8 = self.encod8(e7)

        # --------- Decoder ---------
        d8 = self.decod8(e8)
        d7 = self.decod7(torch.cat([d8, e7], 1))  # concatenating layers cf. U-net
        d6 = self.decod6(torch.cat([d7, e6], 1))  # concatenating layers cf. U-net
        d5 = self.decod5(torch.cat([d6, e5], 1))  # concatenating layers cf. U-net
        d4 = self.decod4(torch.cat([d5, e4], 1))  # concatenating layers cf. U-net
        d3 = self.decod3(torch.cat([d4, e3], 1))  # concatenating layers cf. U-net
        d2 = self.decod2(torch.cat([d3, e2], 1))  # concatenating layers cf. U-net

        out = self.decodout(torch.cat([d2, e1], 1))

        return out
"""