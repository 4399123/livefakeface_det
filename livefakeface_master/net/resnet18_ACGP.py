
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchsummary import summary
class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0=x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return x0*self.sigmoid(x)


class GDC(nn.Module):
    def __init__(self,in_ch):
        super(GDC, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=7,
                                    stride=1,
                                    padding=0,
                                    groups=in_ch)
    def forward(self,input):
        out = self.depth_conv(input)
        return out

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff
class FBasicConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(FBasicConv2d, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=1,bias=False,**kwargs)
        self.bn= nn.GroupNorm(num_channels=out_channels,num_groups=4)
        self.av=nn.ReLU(inplace=True)
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.av(x)
        return x


# 编写卷积+bn+relu模块
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channals, style='normal',**kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = Conv2d_cd(in_channels, out_channals,kernel_size=3, padding=1,stride=1,bias=False,**kwargs)
        self.bn = nn.GroupNorm(num_groups=4,num_channels=out_channals)
        self.conv_3x1=nn.Conv2d(out_channals,out_channals,kernel_size=(3,1),stride=1,padding=(1,0),bias=False,**kwargs)
        self.conv_1x3 = nn.Conv2d(out_channals, out_channals, kernel_size=(1, 3), stride=1, padding=(0, 1),bias=False,**kwargs)
        self.relu=nn.SiLU(inplace=True)
        if style=='downsample':
            self.downsample=nn.Conv2d(in_channels,out_channals,kernel_size=1, padding=0,stride=1)
        self.style=style
        self.sa = SpatialAttention(kernel=3)
    def forward(self, x):
        if self.style=='downsample':
            x0=self.downsample(x)
        else:
            x0=x
        x = self.conv(x)
        x = self.bn(x)
        x=self.relu(x)
        x1=x
        x = self.conv_3x1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_1x3(x)
        x = self.bn(x)
        x=x+x1
        x = self.relu(x)
        x = self.sa(x)
        x=x+x0
        x = self.relu(x)
        return x
group=8
class Model(nn.Module):
    def __init__(self,blocks=[2,4,5,5],scale=0.25):
        inplane=int(64*scale)
        self.explansion=2
        self.lastchannel=1024
        self.scale=scale
        super(Model, self).__init__()

        ####he初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        self.fconv1=FBasicConv2d(3,inplane)
        self.fconv2 = FBasicConv2d(inplane, inplane)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)

        self.layer1 = self.make_layer(BasicConv2d, int(128*scale), blocks[0])
        self.layer2 = self.make_layer(BasicConv2d, int(256*scale), blocks[1])
        self.layer3 = self.make_layer(BasicConv2d, int(512*scale), blocks[2])
        self.layer4 = self.make_layer(BasicConv2d, int(512*scale), blocks[3],style='equal')
        self.gdc = GDC(self.lastchannel)
        self.linear = nn.Linear(self.lastchannel, 2)
        # self.sa3 = SpatialAttention(kernel=3)
        self.global_conection=nn.AdaptiveAvgPool2d((7,7))
        self.global_conv=nn.Conv2d(inplane,int(512*scale),kernel_size=1,stride=1,padding=0)
        self.lastconv=nn.Conv2d(int(512*scale),self.lastchannel,kernel_size=1,padding=0,stride=1)


    def make_layer(self,bsconv,plane,blocks,style='normal'):
        layers = []
        if style=='equal':
            layers.append(bsconv(in_channels=plane, out_channals=plane, groups=group))
        else:
            layers.append(bsconv(in_channels=plane // 2, out_channals=plane, groups=group,style='downsample'))

        for i in range(1,blocks):
            layers.append(bsconv(in_channels=plane, out_channals=plane, groups=group))
        # layers.append(SpatialAttention(kernel=3))
        if self.scale>=0.5:
            layers.append(nn.Dropout(0.4))
        layers.append(nn.MaxPool2d(3, 2, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.fconv1(x)
        sx=self.global_conv(self.global_conection(x))
        x=self.fconv2(x)
        x=self.maxpool(x)
        x =self.layer1(x)
        x =self.layer2(x)
        x =self.layer3(x)
        x =self.layer4(x)
        x=x+sx
        x=F.silu(x)
        x=self.lastconv(x)
        out = self.gdc(x)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out
if __name__=='__main__':
    if torch.cuda.is_available():
        net=Model().cuda()
    else:
        net = Model()
    summary(net,(3,224,224))