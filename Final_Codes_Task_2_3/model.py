import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# **1_Defining DeepLabV3 for Task3**

# **resnet**

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) 

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) 

    return layer


# Defining the block to be used in the resnet18 as layer 4 and 5
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (input shape: (batch_sz, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) 
        out = self.bn2(self.conv2(out)) 

        out = out + self.downsample(x) 

        out = F.relu(out) 

        return out


# Block for basical ResNet18 definition with pretrained weights loading
# We pass an input of size (batch_sz, 3, h, w) while take the output from the 5th layer
class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self):
        super(ResNet_BasicBlock_OS8, self).__init__()

        resnet = models.resnet18()
        # loading pretrained resnet18 model
        resnet.load_state_dict(torch.load("./pretrained_model/resnet18.pth"))
        # removing the fully connected layer, avg pool layer, layer4 and the  layer5
        self.resnet = nn.Sequential(*list(resnet.children())[:-4])

        num_blocks_layer_4 = 2
        num_blocks_layer_5 = 2

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)

    def forward(self, x):
        # (input shape : (batch_sz, 3, h, w))

        # passing input through the pretrained ResNet:
        c3 = self.resnet(x) 

        output = self.layer4(c3) 
        output = self.layer5(output) 

        return output


def ResNet18_OS8():
    return ResNet_BasicBlock_OS8()


# **ASPP [Atrous Spatial Pyramid Pooling]**

class ASPP(nn.Module):
    def __init__(self, num_classes):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1) 
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):

        feature_map_h = feature_map.size()[2] 
        feature_map_w = feature_map.size()[3] 
        
        # One 1x1 conv followed by three 3x3 conv with atrous rates of
        # 6, 12, and 18 respectively.
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) 
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) 
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) 
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) 
        
        # Using average pooling to get the global context
        out_img = self.avg_pool(feature_map) 
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) 
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") 
        
        # Concatenating all the features
        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) 
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) 
        out = self.conv_1x1_4(out) # Getting the final logits

        return out


# **Final_Model_DeepLabV3**

class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()

        self.num_classes = 19

        self.resnet = ResNet18_OS8() # specifying the resnet model
        self.aspp = ASPP(num_classes=self.num_classes) 

    def forward(self, x):
        # input x has the shape (batch_size, 3, h, w)

        h = x.size()[2]
        w = x.size()[3]
        
        # Getting the feature map after passing the input through ResNet18
        feature_map = self.resnet(x) 
        
        # Passing the feature map through ASPP
        output = self.aspp(feature_map) 
        
        # Upsampling the final logits we get from ASPP to the original
        # Image size using bilinear upsampling
        output = F.upsample(output, size=(h, w), mode="bilinear") 

        return output


# **2_Defining R2UNet for Task2**

# +
# Define up convolution
class up_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(up_conv, self).__init__()
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c,out_c,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    def forward(self, image):
        image = self.upconv(image)
        return image

# Define recurrent block
class rec_block(nn.Module):
    def __init__(self, out_c, t):
        super(rec_block, self).__init__()
        self.t = t
        self.out_c = out_c
        
        self.rec_conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, image):
        for i in range(self.t):
            if i == 0:
                x1 = self.rec_conv(image)
            x1 = self.rec_conv(image+x1)
        return x1

# Define recurrent residual block
class rr_block(nn.Module):
    def __init__(self, in_c, out_c, t=3):
        super(rr_block, self).__init__()
        self.rcnn = nn.Sequential(
            rec_block(out_c, t=t),
            rec_block(out_c, t=t)
        )
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)
    
    def forward(self, image):
        image = self.conv(image)
        x = self.rcnn(image)
        out = image+x
        return out

# R2U-Net Architecture
class R2UNet(nn.Module):
    #img_ch=3, output_ch=1, t=2
    def __init__(self):
        super(R2UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        
        self.RR_block1 = rr_block(3, 64, t=3)
        self.RR_block2 = rr_block(64, 128, t=3)
        self.RR_block3 = rr_block(128, 256, t=3)
        self.RR_block4 = rr_block(256, 512, t=3)
        self.RR_block5 = rr_block(512, 1024, t=3)
        
        self.Up5 = up_conv(1024, 512)
        self.Up_rr_block5 = rr_block(1024, 512, t=3)
        
        self.Up4 = up_conv(512, 256)
        self.Up_rr_block4 = rr_block(512, 256, t=3)
        
        self.Up3 = up_conv(256, 128)
        self.Up_rr_block3 = rr_block(256, 128, t=3)
        
        self.Up2 = up_conv(128, 64)
        self.Up_rr_block2 = rr_block(128, 64, t=3)
        
        self.Conv = nn.Conv2d(64, 19, kernel_size=1, stride=1, padding=0)
        
    def forward(self, image): 
        #encoding path
        x1 = self.RR_block1(image)
        
        x2 = self.Maxpool(x1)
        x2 = self.RR_block2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RR_block3(x3)
        
        x4 = self.Maxpool(x3)
        x4 = self.RR_block4(x4)
        
        x5 = self.Maxpool(x4)
        x5 = self.RR_block5(x5)
        
        #decoding path
        y5 = self.Up5(x5)
        y5 = torch.cat((x4, y5), dim=1)
        y5 = self.Up_rr_block5(y5)
        
        y4 = self.Up4(y5)
        y4 = torch.cat((x3, y4), dim=1)
        y4 = self.Up_rr_block4(y4)
        
        y3 = self.Up3(y4)
        y3 = torch.cat((x2, y3), dim=1)
        y3 = self.Up_rr_block3(y3)
        
        y2 = self.Up2(y3)
        y2 = torch.cat((x1, y2), dim=1)
        y2 = self.Up_rr_block2(y2)
        
        out = self.Conv(y2)
        
        return out

# **3_Defining UNet3+ for Task3**

# Define convolution block
class conv2(nn.Module):
    def __init__(self, in_c, out_c, t=2, kernel=3, stride=1, padding=1):
        super(conv2, self).__init__()
        self.t = t
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        for i in range(1, t + 1):
            conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel, stride, padding),
                                 nn.BatchNorm2d(out_c),
                                 nn.ReLU(inplace=True)
                                 )
            setattr(self, 'conv%d' % i, conv)
            in_c = out_c

    def forward(self, image):
        x = image
        for i in range(1, self.t + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

# Define UNet3+ Architecture
class UNet3Plus(nn.Module):
    def __init__(self, n_classes=1, bilinear=True, feature_scale=4,
                 is_deconv=True):
        super(UNet3Plus, self).__init__()

        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        filters = [64, 128, 256, 512, 1024]

        ## -------------Encoder--------------
        self.conv1 = conv2(3, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = conv2(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = conv2(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = conv2(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = conv2(filters[3], filters[4])

        ## -------------Decoder--------------
        self.cat_channels = filters[0]
        self.cat_blocks = 5
        self.up_channels = self.cat_channels * self.cat_blocks #for full scale skip connections

        # each stage represents a decoder block
        '''
        Maxpool - Connection between encoder-decoder
        Upsample - Connection between decoder-decoder
        '''
        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.cat_channels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.cat_channels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.cat_channels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.cat_channels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.cat_channels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.cat_channels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.cat_channels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.cat_channels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.cat_channels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.up_channels, self.up_channels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.up_channels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.cat_channels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.cat_channels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.cat_channels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.cat_channels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.cat_channels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.cat_channels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.up_channels, self.cat_channels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.cat_channels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.up_channels, self.up_channels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.up_channels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.cat_channels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.cat_channels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.cat_channels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.cat_channels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.up_channels, self.cat_channels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.up_channels, self.cat_channels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.cat_channels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.up_channels, self.up_channels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.up_channels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.cat_channels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.cat_channels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.up_channels, self.cat_channels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.up_channels, self.cat_channels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.up_channels, self.cat_channels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear')  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.cat_channels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.cat_channels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.up_channels, self.up_channels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.up_channels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # output
        self.out = nn.Conv2d(self.up_channels, 20, 3, padding=1)

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->320*320*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->160*160*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->80*80*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->40*40*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->20*20*1024

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d1 = self.out(hd1)  # d1->320*320*n_classes
        return d1
