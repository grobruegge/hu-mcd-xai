import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import conv2d

def neighbor_padding(x, mask, iterator=4):
    n_channels = x.shape[1]
    # define 3x3 filters with ones
    filter_x = torch.ones((n_channels, 1, 3, 3), device=x.device)
    filter_mask = torch.ones((1, 1, 3, 3), device=x.device)
    x = x*mask
    for i in range(iterator):
        with torch.no_grad():
            # sum of neighboring pixel values for each pixel in the padded image
            neighbor_sums = conv2d(x, filter_x, groups=n_channels, padding='same')
            # count of valid neighboring pixels for each pixel
            non_zero_neighbor_count = conv2d(mask, filter_mask, groups=1, padding='same')
            # fill the adjacent pixels to the masks using the average value of its unmasked neighbors
            edge_fills = (1-mask)*(neighbor_sums/(non_zero_neighbor_count.clamp(min=1)))
        # apply neighborhod padding for 1 pixel surrounding the masks
        x = x + edge_fills.detach()
        # expand the mask
        mask = (mask + non_zero_neighbor_count).clip(0,1)
    return x

class MyConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = torch.nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=self.padding)
        return
    
    def forward(self, input):
        if isinstance(input, Tensor):
            return self._conv_forward(input, self.weight, self.bias)
        
        elif isinstance(input, tuple):
            x, mask, mode = input
            assert torch.all((mask == 0) | (mask == 1)), "[ERROR] masks is not binary"
            # mode = -1 : pass unmasked image to conv layer 
            # mode = 0 : pass masked image with hard borders to conv layer (mask edge
            # information is propagated)
            # mode = 1 : pass (neighbor) padded masked image to conv layer
            if mode >= 0: # apply neighbor padding
                kernel_size = self.weight.shape[-1] if mode > 0 else 0
                x_masked = neighbor_padding(x, mask, kernel_size)
            else: # use unmasked input 
                x_masked = x
            conv_inp = self._conv_forward(x_masked, self.weight, self.bias)
            with torch.no_grad():
                conv_mask = self.maxpool(mask)
            return (conv_inp*conv_mask, conv_mask, mode)
        
        else:
            raise Exception(f'[ERROR] Encountered input type {type(input)}')
        
class MyMaxPool2d(nn.MaxPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            x, mask, mode = input
            x_pool = super().forward(x)
            with torch.no_grad():
                mask_pool = super().forward(mask)
            return (x_pool*mask_pool, mask_pool, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyAvgPool2d(nn.AvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            x, mask, mode = input
            x_pool = super().forward(x)
            with torch.no_grad():
                mask_pool = (super().forward(mask) > 0).float()
            return (x_pool*mask_pool, mask_pool, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')
    
class MyBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp*mask)*mask, mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            out = super().forward(inp*mask)
            mask_avg = super().forward(mask) + 1e-8
            return out/mask_avg
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MySigmoid(nn.Sigmoid):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp*mask)*mask, mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')

class MyReLU(nn.ReLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp*mask)*mask, mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')

            
class MyHardsigmoid(nn.Hardsigmoid):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp*mask)*mask, mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyHardswish(nn.Hardswish):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp*mask)*mask, mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MySiLU(nn.SiLU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp*mask)*mask, mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyGELU(nn.GELU):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp), mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')
            
class MyIdentity(nn.Identity):
    def forward(self, input):
        if isinstance(input, Tensor):
            return super().forward(input)
        elif isinstance(input, tuple):
            inp, mask, mode = input
            return (super().forward(inp), mask, mode)
        else:
            raise Exception(f'Encountered input type {type(input)}')