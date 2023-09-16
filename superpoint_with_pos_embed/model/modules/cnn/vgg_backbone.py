import math
import torch


def image_positional_encoding(d_model, height, width):

    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)   # Each dimension use half of d_model
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class VGGBackbone(torch.nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """
    def __init__(self, config, input_channel=1, device='cpu'):
        super(VGGBackbone, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
        )


    def forward(self, x):
        out = self.block1_1(x)

        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)

        # print("x.shape = ", x.detach().cpu().numpy().shape)
        # print("feat_map.shape = ", feat_map.detach().cpu().numpy().shape)

        print("USE VGG-BACKBONE.")

        return feat_map


class VGGBackboneBN(torch.nn.Module):
    """vgg backbone to extract feature
    Note:set eps=1e-3 for BatchNorm2d to reproduce results
         of pretrained model `superpoint_bn.pth`
    """
    def __init__(self, config, input_channel=1, device='cpu'):
        super(VGGBackboneBN, self).__init__()
        self.device = device
        channels = config['channels']

        self.block1_1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[0]),
        )
        self.block1_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[1]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[2]),
        )
        self.block2_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[3]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block3_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[3], channels[4], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[4]),
        )
        self.block3_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[5]),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block4_1 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[5], channels[6], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[6]),
        )
        self.block4_2 = torch.nn.Sequential(
            torch.nn.Conv2d(channels[6], channels[7], kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.BatchNorm2d(channels[7]),
        )

    def forward(self, x):
        out = self.block1_1(x)
        out = self.block1_2(out)
        out = self.block2_1(out)
        out = self.block2_2(out)
        out = self.block3_1(out)
        out = self.block3_2(out)
        out = self.block4_1(out)
        feat_map = self.block4_2(out)

        # print("x.shape = ", x.detach().cpu().numpy().shape)
        # print("feat_map.shape = ", feat_map.detach().cpu().numpy().shape)

        return feat_map


class ConvLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, max_pool):
        super(ConvLayer, self).__init__()
        if max_pool == True:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1), 
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(out_dim),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )
        else: # max_pool = False
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1), 
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(out_dim),
            )

    def forward(self, x):
        return self.conv(x)

# class ConvLayer(torch.nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(ConvLayer, self).__init__()
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.BatchNorm2d(out_dim),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         )

#     def forward(self, x):
#         return self.conv(x)


class PEBackbone_MOD(torch.nn.Module):

    def __init__(self, config, input_channel=1, device='cpu'):
        super(PEBackbone_MOD, self).__init__()
        self.device = device
        # channels = config['channels']

        in_dim, out_dim, latent_dim = 128, 128, 64
        self.encoder1 = ConvLayer(in_dim, latent_dim, max_pool=True)
        self.encoder2 = ConvLayer(latent_dim, latent_dim, max_pool=True) 
        self.decoder1 = ConvLayer(latent_dim, latent_dim, max_pool=True)
        self.decoder2 = ConvLayer(latent_dim, out_dim, max_pool=False)
        # self.conv1 = ConvLayer(out_dim, channels[0], max_pool=True)
        # self.conv2 = ConvLayer(channels[0], channels[1], max_pool=True)
        # self.conv3 = ConvLayer(channels[1], channels[2], max_pool=True)
        # self.conv4 = ConvLayer(channels[2], channels[3], max_pool=False)

        # # channels: [64,64,128,128]
        # self.block1_2 = ConvLayer(input_channel, channels[0], max_pool=True)
        # self.block2_2 = ConvLayer(channels[0], channels[1], max_pool=True)
        # self.block3_2 = ConvLayer(channels[1], channels[2], max_pool=True)
        # self.block4_2 = ConvLayer(channels[2], channels[3], max_pool=False)

        # full vesion
        # self.block1_2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(input_channel, channels[0], kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.BatchNorm2d(channels[0]),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.block2_2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.BatchNorm2d(channels[1]),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.block3_2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.BatchNorm2d(channels[2]),
        #     torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.block4_2 = torch.nn.Sequential(
        #     torch.nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.BatchNorm2d(channels[3]),
        # )

    def forward(self, x):
        # print("x.shape = ", x.detach().cpu().numpy().shape)
        
        pe = image_positional_encoding(128, 240, 320).cuda()
        # print("pe.shape = ", pe.detach().cpu().numpy().shape)
        # print("out0.shape = ", (x + pe).detach().cpu().numpy().shape)

        out1 = self.encoder1(x + pe.unsqueeze(0))
        # print("out1.shape = ", out1.detach().cpu().numpy().shape)
        out2 = self.encoder2(out1)
        # print("out2.shape = ", out2.detach().cpu().numpy().shape)
        out3 = self.decoder1(out2)
        # print("out3.shape = ", out3.detach().cpu().numpy().shape)
        feat_map = self.decoder2(out3)
        # print("feat_map.shape = ", feat_map.detach().cpu().numpy().shape)

        # out4 = self.conv1(out3)
        # print("out4.shape = ", out4.detach().cpu().numpy().shape)
        # out5 = self.conv2(out4)
        # print("out5.shape = ", out5.detach().cpu().numpy().shape)
        # out6 = self.conv3(out5)
        # print("out6.shape = ", out6.detach().cpu().numpy().shape)
        # feat_map = self.conv4(out6)
        # print("feat_map.shape = ", feat_map.detach().cpu().numpy().shape)

        '''
        x.shape =  (2, 1, 240, 320) <-- pe.shape =  (128, 240, 320)
        (x + pe).shape =  (2, 128, 240, 320)
        out1.shape =  (2, 64, 120, 160)
        out2.shape =  (2, 64, 60, 80)
        out3.shape =  (2, 64, 30, 40)
        feat_map.shape =  (2, 128, 30, 40)
        '''
        
        # print("\n=================")

        # print("x.shape = ", x.detach().cpu().numpy().shape)
        # out1 = self.block1_2(x)
        # print("out1.shape = ", out1.detach().cpu().numpy().shape)
        # out2 = self.block2_2(out1)
        # print("out2.shape = ", out2.detach().cpu().numpy().shape)
        # out3 = self.block3_2(out2)
        # print("out3.shape = ", out3.detach().cpu().numpy().shape)
        # feat_map = self.block4_2(out3)        
        # print("feat_map.shape = ", feat_map.detach().cpu().numpy().shape)

        '''
        x.shape =  (2, 1, 240, 320)
        out1.shape =  (2, 64, 120, 160)
        out2.shape =  (2, 64, 60, 80)
        out3.shape =  (2, 128, 30, 40)
        feat_map.shape =  (2, 128, 30, 40)
        '''

        # exit()
        print("USE PE.")

        return feat_map
