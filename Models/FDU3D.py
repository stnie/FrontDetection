import torch
import torch.nn.functional as F
from collections import OrderedDict 

class FDU2DNetLargeEmbedCombineModular(torch.nn.Module):


    def contracting_block(self, in_channels, mid_channels, out_channels, kernel_size=3, padding = 0, dropoutRate = 0.2, num_conv_blocks = 3):
        Parts = OrderedDict()
        
        Parts["conv_0"] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=in_channels, out_channels = mid_channels, padding=padding)
        Parts["relu_0"] = torch.nn.ReLU()
        Parts["BatchNorm2d_0"] = torch.nn.BatchNorm2d(mid_channels)
        
        for blockID in range(1,num_conv_blocks-1):
            Parts["conv_"+str(blockID)] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=mid_channels, out_channels = mid_channels, padding=padding)
            Parts["relu_"+str(blockID)] = torch.nn.ReLU()
            Parts["BatchNorm2d_"+str(blockID)] = torch.nn.BatchNorm2d(mid_channels)
        
        Parts["conv_"+str(num_conv_blocks-1)] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=mid_channels, out_channels = out_channels, padding=padding)
        Parts["relu_"+str(num_conv_blocks-1)] = torch.nn.ReLU()
        Parts["BatchNorm2d_"+str(num_conv_blocks-1)] = torch.nn.BatchNorm2d(out_channels)
        Parts["Dropout2d"] = torch.nn.Dropout2d(dropoutRate, inplace=True)
        
        block = torch.nn.Sequential(Parts)
        return block
    
    def bottleneck_block(self, in_channels, mid_channels, out_channels, kernel_size=3, padding = 0):
        block = torch.nn.Sequential(
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channels, padding=padding),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channels, out_channels=mid_channels,padding=padding),
                torch.nn.ReLU(),
                torch.nn.BatchNorm2d(mid_channels),
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
                )
        return block

    def expansive_block(self, in_channels, mid_channels, out_channels, kernel_size=3, padding = 0, num_conv_blocks = 3):
        Parts = OrderedDict()
        
        Parts["conv_0"] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=in_channels, out_channels = mid_channels, padding = padding)
        Parts["relu_0"] = torch.nn.ReLU()
        Parts["BatchNorm2d_0"] = torch.nn.BatchNorm2d(mid_channels)
        
        for blockID in range(1,num_conv_blocks-1):
            Parts["conv_"+str(blockID)] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=mid_channels, out_channels = mid_channels, padding=padding)
            Parts["relu_"+str(blockID)] = torch.nn.ReLU()
            Parts["BatchNorm2d_"+str(blockID)] = torch.nn.BatchNorm2d(mid_channels)
        
        Parts["conv_"+str(num_conv_blocks-1)] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=mid_channels, out_channels = out_channels, padding=padding)
        Parts["relu_"+str(num_conv_blocks-1)] = torch.nn.ReLU()
        Parts["BatchNorm2d_"+str(num_conv_blocks-1)] = torch.nn.BatchNorm2d(out_channels)

        Parts["Upsample"] = torch.nn.Upsample(scale_factor=2)
        Parts["conv_up"] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=out_channels, out_channels = out_channels, padding=padding)
        Parts["relu_up"] = torch.nn.ReLU()
        Parts["BatchNorm2d_up"] = torch.nn.BatchNorm2d(out_channels)
        block = torch.nn.Sequential(Parts)
        return block
    def final_block(self, in_channels, mid_channels, out_channels, kernel_size=3, padding = 0, num_conv_blocks = 3):
        Parts = OrderedDict()
        
        Parts["conv_0"] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=in_channels, out_channels = mid_channels, padding = padding)
        Parts["relu_0"] = torch.nn.ReLU()
        Parts["BatchNorm2d_0"] = torch.nn.BatchNorm2d(mid_channels)
        
        for blockID in range(1,num_conv_blocks-1):
            Parts["conv_"+str(blockID)] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=mid_channels, out_channels = mid_channels, padding=padding)
            Parts["relu_"+str(blockID)] = torch.nn.ReLU()
            Parts["BatchNorm2d_"+str(blockID)] = torch.nn.BatchNorm2d(mid_channels)
        
        Parts["conv_"+str(num_conv_blocks-1)] = torch.nn.Conv2d(kernel_size = kernel_size, in_channels=mid_channels, out_channels = out_channels, padding=padding)
        Parts["relu_"+str(num_conv_blocks-1)] = torch.nn.ReLU()
        Parts["BatchNorm2d_"+str(num_conv_blocks-1)] = torch.nn.BatchNorm2d(out_channels)

        block = torch.nn.Sequential(Parts)
        return block

    def embed_block(self, in_channels, out_channels, kernel_size = 3, padding = 0):
        block = torch.nn.Sequential(
                #torch.nn.Dropout2d(0.2, inplace = True),
                torch.nn.Conv2d(kernel_size = kernel_size, in_channels=in_channels, out_channels = out_channels, groups = in_channels, padding = padding),
                torch.nn.BatchNorm2d(out_channels, affine = True)
            )
        return block

    def __init__(self, in_channel, out_channel, kernel_size = 5, sub_blocks = (3,3,3), embedding_factor = 6):
        super(FDU2DNetLargeEmbedCombineModular, self).__init__()
        kernelsz = kernel_size
        padding = kernelsz//2-1
        noLosspadding = kernelsz//2
        embeddingFactor = embedding_factor
        embeddingKernelSz = 1
        embeddingNoLossPadding = embeddingKernelSz//2
        embedded_channel = int(in_channel*embeddingFactor)
        stg1_chnl = 64
        stg2_chnl = stg1_chnl*2
        stg3_chnl = stg2_chnl*2
        btlnk_chnl = stg3_chnl*2
        EncodeSubBlocks = sub_blocks[0]
        DecodeSubBlocks = sub_blocks[1]
        FinalSubBlocks = sub_blocks[2]
        
        #Encode 
        self.embedder = self.contracting_block(in_channels=in_channel, mid_channels = max(embedded_channel, in_channel), out_channels = embedded_channel, kernel_size = embeddingKernelSz, padding = embeddingNoLossPadding, num_conv_blocks = 2)
        self.conv_encode1 = self.contracting_block(in_channels=embedded_channel, mid_channels = stg1_chnl, out_channels=stg1_chnl, kernel_size = kernelsz, padding = noLosspadding, num_conv_blocks = EncodeSubBlocks)
        self.conv_maxpool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(in_channels=stg1_chnl, mid_channels = stg2_chnl, out_channels=stg2_chnl, kernel_size = kernelsz, padding = noLosspadding, num_conv_blocks = EncodeSubBlocks)
        self.conv_maxpool2 = torch.nn.AvgPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(in_channels=stg2_chnl, mid_channels = stg3_chnl, out_channels=stg3_chnl, kernel_size = kernelsz, padding = noLosspadding, num_conv_blocks = EncodeSubBlocks)
        self.conv_maxpool3 = torch.nn.AvgPool2d(kernel_size=2)

        #Bottleneck
        self.bottleneck = self.expansive_block(stg3_chnl, btlnk_chnl, stg3_chnl, kernel_size=kernelsz, padding = noLosspadding, num_conv_blocks = DecodeSubBlocks)

        #Decode
        self.conv_decode3 = self.expansive_block(btlnk_chnl,stg3_chnl,stg2_chnl, kernel_size = kernelsz, padding = noLosspadding, num_conv_blocks = DecodeSubBlocks)
        self.conv_decode2 = self.expansive_block(stg3_chnl,stg2_chnl,stg1_chnl, kernel_size = kernelsz, padding = noLosspadding, num_conv_blocks = DecodeSubBlocks)
        self.final_layer = self.final_block(stg2_chnl, stg1_chnl, out_channel, kernel_size = kernelsz, padding = noLosspadding, num_conv_blocks = FinalSubBlocks)


    def crop_and_concat(self, upsampled, bypass, crop=False):
        if(crop):
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            c2 = (bypass.size()[3] - upsampled.size()[3]) // 2
            bypass = F.pad(bypass, (-c, -c, -c2, -c2))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        #Embed
        x_embedded = self.embedder(x)
        #Encode 
        encode_block1 = self.conv_encode1(x_embedded)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        #Bottleneck
        bottleneck1 = self.bottleneck(encode_pool3)
        #Decode
        cat_layer3 = self.crop_and_concat(bottleneck1, encode_block3, crop = True)
        decode_block2 = self.conv_decode3(cat_layer3)
        cat_layer2 = self.crop_and_concat(decode_block2, encode_block2, crop = True)
        decode_block1 = self.conv_decode2(cat_layer2)
        cat_layer1 = self.crop_and_concat(decode_block1, encode_block1, crop = True)
        final_layer = self.final_layer(cat_layer1)
        return final_layer
