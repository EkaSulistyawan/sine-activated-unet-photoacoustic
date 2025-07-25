import numpy as np
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, TensorDataset

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate    

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
def sine_init(m, omega=30):
    if isinstance(m, nn.Conv2d):
        with torch.no_grad():
            m.weight.uniform_(-1/omega, 1/omega)

class SoftShrinkageActivation(nn.Module):
    def __init__(self, lambd):
        super(SoftShrinkageActivation, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return torch.sign(x) * torch.maximum(torch.abs(x) - self.lambd, torch.zeros_like(x))

from torch.nn.functional import interpolate
def split_into_patches(image: torch.Tensor, patch_size: tuple) -> torch.Tensor:
    """
    Splits an image into non-overlapping patches and treats each patch as a batch.
    
    Args:
        image (torch.Tensor): Input tensor of shape (Batch, Channel, Height, Width).
        patch_size (tuple): Tuple indicating the size of each patch (Height, Width).
        
    Returns:
        torch.Tensor: Patches of shape (NumPatches, Channels, PatchHeight, PatchWidth).
    """
    # Calculate padding
    _, _, height, width = image.shape
    pad_h = (patch_size[0] - height % patch_size[0]) % patch_size[0]
    pad_w = (patch_size[1] - width % patch_size[1]) % patch_size[1]

    # Pad the image
    padded_image = F.pad(image, (0, pad_w, 0, pad_h))  # Padding: (Left, Right, Top, Bottom)

    # Use nn.Unfold to extract patches
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(padded_image)  # Shape: (Batch, PatchSize*PatchSize*Channels, NumPatches)

    # Reshape to make each patch a batch
    patches = patches.permute(0, 2, 1)  # Move patches to the second dimension
    patches = patches.view(-1, image.shape[1], patch_size[0], patch_size[1])  # (NumPatches, Channels, PatchHeight, PatchWidth)

    return patches

def reconstruct_from_patches(patches: torch.Tensor, original_size: tuple, patch_size: tuple) -> torch.Tensor:
    """
    Reconstructs the original image from non-overlapping patches.
    
    Args:
        patches (torch.Tensor): Patches of shape (NumPatches, Channels, PatchHeight, PatchWidth).
        original_size (tuple): Original size of the image (Height, Width).
        patch_size (tuple): Tuple indicating the size of each patch (PatchHeight, PatchWidth).
        
    Returns:
        torch.Tensor: Reconstructed image of shape (Batch, Channels, Height, Width).
    """
    # Get the number of patches per dimension
    batch_size = 1  # Assuming patches came from one image
    channels = patches.shape[1]
    padded_height = (original_size[0] + patch_size[0] - 1) // patch_size[0] * patch_size[0]
    padded_width = (original_size[1] + patch_size[1] - 1) // patch_size[1] * patch_size[1]
    
    # Reshape patches back to the unfolded shape
    num_patches_h = padded_height // patch_size[0]
    num_patches_w = padded_width // patch_size[1]
    patches = patches.view(batch_size, num_patches_h * num_patches_w, -1).permute(0, 2, 1)

    # Reconstruct the padded image using Fold
    fold = torch.nn.Fold(output_size=(padded_height, padded_width), kernel_size=patch_size, stride=patch_size)
    reconstructed_image = fold(patches)

    # Remove padding to restore original dimensions
    reconstructed_image = reconstructed_image[:, :, :original_size[0], :original_size[1]]
    return reconstructed_image
    
class DNNSinogramSR(nn.Module):

    def __init__(self, activ_func1, activ_func2, f = 32):

        super(DNNSinogramSR, self).__init__()
        # print('Old')
        # print('\n')
        self.f = f
        activ_func = activ_func1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.f,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f, out_channels=self.f,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=f, out_channels=self.f*2,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f*2,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f*4,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*4,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*8,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*8,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*16,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*16, out_channels=self.f*16,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=self.f*16, out_channels=self.f*8, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*16, out_channels=self.f*8,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*8,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=self.f*8, out_channels=self.f*4, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*8, out_channels=self.f*4,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*4,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=self.f*4, out_channels=self.f*2, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*4, out_channels=self.f*2,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f*2,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=self.f*2, out_channels=self.f, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*2, out_channels=self.f,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f, out_channels=self.f,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func,
        )

        # Final output
        # self.conv_final = nn.Conv2d(in_channels=32, out_channels=2,
        #                             kernel_size=1, padding=0, stride=1)

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.f, out_channels=1, # use channels 2 for positive negative output
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func2,
            nn.Conv2d(in_channels=1, out_channels=1,
                      kernel_size=3, padding=1,padding_mode='zeros', stride=1),
            activ_func2,
        )

    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv5', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)
        # print('after conv4', x.shape)

        # Midpoint
        x = self.conv5_block(x)
        # print('mid', x.shape)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)

        # if positive 2 channels
        # if (x.shape[1] == 2):
            # x = -x[:,0,:,:] + x[:,1,:,:]
            # x = x.unsqueeze(1)

        return x
    import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseBlock(nn.Module):
    def __init__(self,activ_func, in_channels, growth_rate,feature_map, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(channels, feature_map, kernel_size=1, padding=0, stride=1),
                activ_func,
                nn.Conv2d(feature_map, growth_rate, kernel_size=3, padding=1, stride=1),
                activ_func # the activation function
            ))
            channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


class Down(nn.Module):
    def __init__(self,activ_func, in_channels, growth_rate, feature_map, num_layers):
        super().__init__()
        self.dense = DenseBlock(activ_func,in_channels, growth_rate, feature_map, num_layers)
        out_channels = in_channels + growth_rate * num_layers
        
        self.pool = nn.MaxPool2d(2)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.dense(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class Up(nn.Module):
    def __init__(self, activ_func, in_channels, skip_channels, growth_rate, feature_map, num_layers):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1,padding=0,stride=1),
            activ_func
        )
        self.dense = DenseBlock(activ_func,in_channels//4, growth_rate, feature_map, num_layers)
        self.out_channels = (in_channels//4) + growth_rate * num_layers

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.first_layer(x)
        return self.dense(x)


class FullyDenseUNet(nn.Module):
    def __init__(self, 
                 in_channels=1,
                 out_channels=1,
                 growth_rate=8,
                 feature_map=32,
                 num_layers=4,
                 activ_func  = nn.ReLU(),
                 activ_func2 = nn.ELU()):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=in_channels, kernel_size=3, padding=1, stride=1),
            activ_func,
        )

        self.enc1 = Down(activ_func,in_channels, growth_rate, feature_map,num_layers)  # output: c1
        self.enc2 = Down(activ_func,self.enc1.out_channels, growth_rate*2, feature_map,num_layers)  # c2
        self.enc3 = Down(activ_func,self.enc2.out_channels, growth_rate*4, feature_map,num_layers)  # c3
        self.enc4 = Down(activ_func,self.enc3.out_channels, growth_rate*8, feature_map,num_layers)  # c4
        # self.enc5 = Down(activ_func,self.enc4.out_channels, growth_rate*16, feature_map,num_layers)  # c4


        self.bottleneck_dense = DenseBlock(activ_func, self.enc4.out_channels, growth_rate*16, feature_map,num_layers)
        self.bottleneck_out_channels = self.enc4.out_channels + growth_rate*16 * num_layers

        # self.dec5 = Up(activ_func,self.bottleneck_out_channels, self.enc5.out_channels, growth_rate*16, feature_map, num_layers)
        self.dec4 = Up(activ_func,self.bottleneck_out_channels, self.enc4.out_channels, growth_rate*8, feature_map, num_layers)
        self.dec3 = Up(activ_func,self.dec4.out_channels, self.enc3.out_channels, growth_rate*4, feature_map, num_layers)
        self.dec2 = Up(activ_func,self.dec3.out_channels, self.enc2.out_channels, growth_rate*2, feature_map, num_layers)
        self.dec1 = Up(activ_func,self.dec2.out_channels, self.enc1.out_channels, growth_rate, feature_map, num_layers)

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.dec1.out_channels, out_channels=out_channels, # use channels 2 for positive negative output
                      kernel_size=3, padding=1, stride=1),
            activ_func2,
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, stride=1),
            activ_func2,
        )

    def forward(self, x):
        xinp  = x
        x     = self.first_layer(x)
        x1, x = self.enc1(x)
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)
        # x5, x = self.enc5(x)

        x = self.bottleneck_dense(x)
        # print(x.shape)

        # x = self.dec5(x, x5)
        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        x = self.conv_final(x + xinp)

        return x

class ResNet(nn.Module):

    def __init__(self,ich,och,ksz,padsz,activ_func,activ_func2,tail=False):
        super(ResNet, self).__init__()
        self.tail = tail
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=ich,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
            nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz),
            activ_func,
        )
        self.layer5 = nn.Conv2d(in_channels=och,out_channels=och,kernel_size=ksz,padding=padsz)
        self.layer6 = nn.Conv2d(in_channels=och,out_channels=ich,kernel_size=1,padding=0)

        # tail layer
        self.tail_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=ksz,padding=padsz),
            activ_func2,
            nn.Conv2d(in_channels=1,out_channels=1,kernel_size=ksz,padding=padsz),
            activ_func2,
        )
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x1 + x2)
        x4 = self.layer4(x2 + x3)
        x5 = self.layer5(x3 + x4)
        x6 = self.layer6(x + x5)
        if self.tail:
            x7 = self.tail_layer(x6)
            return x7
        else:
            return x6 

class MirroredReLU(nn.Module):
    def __init__(self, kernel_size=3, padding=1, bias=True):
        super().__init__()

    def forward(self, x):
        x_pos = F.relu(x)
        x_neg = -F.relu(-x)
        x_cat = torch.cat([x_pos, x_neg], dim=1)  # (N, 2C, H, W)

        return x_cat
class UNETMODIFIED(nn.Module):

    def __init__(self, activ_func1, activ_func2,in_channels=1,out_channels=1, f = 32,multiplier=1):

        super(UNETMODIFIED, self).__init__()
        # print('Old')
        # print('\n')
        self.f = f
        activ_func = activ_func1
        ksize = 3
        padsize = 1
        if isinstance(activ_func1, AbsoluteReLU):
            multiplier = 2
            self.dualrelutail = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels,
                      kernel_size=ksize, padding=padsize, stride=1)
        elif isinstance(activ_func1, MirroredReLU):
            multiplier = 2
            self.dualrelutail = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels,
                      kernel_size=ksize, padding=padsize, stride=1)
        else:
            self.dualrelutail = None
        
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*multiplier, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=f*multiplier, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*2*multiplier, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2_block = nn.Sequential(
        #     nn.Conv2d(in_channels=f*multiplier, out_channels=self.f*2,
        #               kernel_size=ksize, padding=padsize, stride=1),
        #     activ_func,
        # )
        # self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*2*multiplier, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*4*multiplier, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*4*multiplier, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*8*multiplier, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=self.f*8*multiplier, out_channels=self.f*16,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*16*multiplier, out_channels=self.f*16,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )
        
        

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=self.f*16*multiplier, out_channels=self.f*8*multiplier, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*16*multiplier, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*8*multiplier, out_channels=self.f*8,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=self.f*8*multiplier, out_channels=self.f*4*multiplier, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*8*multiplier, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*4*multiplier, out_channels=self.f*4,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=self.f*4*multiplier, out_channels=self.f*2*multiplier, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*4*multiplier, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*2*multiplier, out_channels=self.f*2,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=self.f*2*multiplier, out_channels=self.f*multiplier, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.f*2*multiplier, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
            nn.Conv2d(in_channels=self.f*multiplier, out_channels=self.f,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func,
        )

        # Final output
        # self.conv_final = nn.Conv2d(in_channels=32*multiplier, out_channels=2,
        #                             kernel_size=1, padding=0, stride=1)

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=self.f*multiplier, out_channels=out_channels, # use channels 2 for positive negative output
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func2,
            nn.Conv2d(in_channels=out_channels*multiplier, out_channels=out_channels,
                      kernel_size=ksize, padding=padsize, stride=1),
            activ_func2,
        )

    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv4', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)
        # print('after conv4', x.shape)

        # Midpoint
        x = self.conv5_block(x)
        # print('mid', x.shape)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2]) / 2)
        upper = int(conv4_dim - lower)
        conv4_out_modified = conv4_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv4_out_modified], dim=1)
        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, lower:upper, lower:upper]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)
        # print('test', x.shape)

        if self.dualrelutail is not None:
            x = self.dualrelutail(x)

        return x
        
class AbsoluteReLU(nn.Module):
    def __init__(self, kernel_size=3, padding=1, bias=True):
        super().__init__()

    def forward(self, x):
        x_pos = F.relu(x)
        x_neg = F.relu(-x)
        x_cat = torch.cat([x_pos, x_neg], dim=1)  # (N, 2C, H, W)

        return x_cat

