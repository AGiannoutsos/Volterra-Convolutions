import torch
import numpy as np
from torch import nn


class VolterraKernel2d(torch.nn.Module):
    def __init__(self,
            kernel_space_size,
            order=1,
            masking=True,
            scaling=True):
        super(VolterraKernel2d, self).__init__()
        self.order             = order
        self.kernel_space_size = kernel_space_size
        self.masking           = masking if order>1 else False
        self.scaling           = scaling if order>1 else False

        self.mask =           self.get_superdiogonal_mask(self.kernel_space_size, self.order) 
        self.mask =           nn.Parameter(self.mask, requires_grad=False) 
        self.scaling_factor = 1/torch.sqrt((self.mask.numel() - self.mask.sum()))
        self.scaling_factor = nn.Parameter(self.scaling_factor, requires_grad=False) 

    def get_superdiogonal_mask(self, kernel_space_size, order):    
        # get all the one time possible compinations
        ker = torch.arange(start=0, end=kernel_space_size)
        if order == 1:
            return torch.zeros(kernel_space_size ** order, dtype=torch.bool, requires_grad=False)
        else:
            combinations = torch.combinations(ker, r=order, with_replacement=True)
        mask = torch.ones(kernel_space_size ** order, dtype=torch.bool, requires_grad=False)
        mask = mask.reshape([kernel_space_size for k in range(order)])
        for index in combinations:
            mask[tuple(index)] = 0
        
        mask = mask.view(-1)
        return mask

    def forward(self,x):

        # resize image in (batch, in_channels, kernel_space_size, tiles)
        size = x.size()
        x = x.view(size[0], -1, self.kernel_space_size, size[2])

        x = x.permute(0, 1, 3, 2)
        kernel = x.unsqueeze(4).transpose(4,3) 

        # multiply kernel by its self order times
        for order in range(self.order-1):
            size = x.size()
            x = x.view(size[0], size[1], size[2], -1)
            x = x.unsqueeze(4)
            x = x @ kernel

        size = x.size()
        x = x.view(size[0], size[1], size[2], -1)

        # add mask
        if self.masking:
            x = x.masked_fill(self.mask, 0)
        x = x.permute(0, 1, 3, 2)

        # scaling factor
        if self.scaling:
            x = x * self.scaling_factor

        # resize back image in (batch, in_channels*kernel_space_size, tiles)
        size = x.size()
        x = x.reshape(size[0], -1, size[3])

        return x

class VolterraConv2d_(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int or tuple,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            orders=[1],
            masking=True,
            scaling=True,
            verbose=False):
        super(VolterraConv2d_, self).__init__()

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        
        if type(padding) is not tuple:
            padding = (padding, padding)
        
        if type(stride) is not tuple:
            stride = (stride, stride)
        
        if type(padding) is not tuple:
            padding = (padding, padding)

        self.verbose      = verbose
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = groups
        self.bias         = bias
        self.orders       = orders
        self.masking      = masking
        self.scaling      = scaling

        # get the space of the kernel unfolded
        self.kernel_space_size = self.kernel_size[0] * self.kernel_size[1] 
        self.unfold = nn.Unfold(self.kernel_size, self.dilation, self.padding, self.stride)

        # set volterra weights
        self.volterra_weights = nn.ParameterList([])
        self.volterra_kernels = nn.ModuleList([])
        for order in self.orders:
            # weights
            weight = torch.Tensor(self.out_channels, self.in_channels, self.kernel_space_size ** order)
            # reshape in order to fit in the matmul with volterra kernels
            weight = weight.view(weight.size(0), -1).t()
            weight = nn.init.xavier_uniform_(weight, gain=np.sqrt(2))
            self.volterra_weights.append( nn.Parameter(weight) )
            # kernel extractor
            self.volterra_kernels.append( VolterraKernel2d(self.kernel_space_size, order, self.masking, self.scaling) )

        if self.bias:
            self.bias_weights = nn.Parameter(torch.Tensor(self.out_channels, 1, 1))
            nn.init.zeros_(self.bias_weights)
        
    def conv_output_shape(self, h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
        w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
        return h, w
    

    def forward(self,x):

        # get images out dimension after convolution
        self.out_height, self.out_width = self.conv_output_shape((x.size()[-2], x.size()[-1]), self.kernel_size, self.stride, self.padding, self.dilation)

        # unfold image
        unfolded_input_image = self.unfold(x)

        # get every volterra kernel and multiply the parameters
        out_image = 0
        out_image_linear = 0
        for order, volterra_kernel, volterra_weight in zip(self.orders, self.volterra_kernels, self.volterra_weights):
            out_image += volterra_kernel(unfolded_input_image).transpose(1, 2) @ volterra_weight

        # fold back to image
        folded_out_image = out_image.view(-1, self.out_channels, self.out_height, self.out_width) + out_image_linear
        # folded_out_image = nn.functional.fold(out_image, (self.out_height, self.out_width), self.kernel_size, self.dilation, self.padding, self.stride)

        if self.bias:
            folded_out_image += self.bias_weights
        
        return folded_out_image

    
class VolterraConv2d(torch.nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            kernel_size: int or tuple,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            orders=[1],
            masking=True,
            scaling=True,
            verbose=False):
        super(VolterraConv2d, self).__init__()

        self.orders = orders
        self.bias   = bias
        self.convs  = nn.ModuleList([])

        for order in self.orders:
            if order == 1:
                self.convs.append(nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=False))
            else:
                self.convs.append(VolterraConv2d_(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=False,
                                orders=[order],
                                masking=masking,
                                scaling=scaling,
                                verbose=verbose))       
        if self.bias:
            self.bias_weights = nn.Parameter(torch.Tensor(out_channels, 1, 1))
            nn.init.zeros_(self.bias_weights)                   

    def forward(self,x):
        out_image = 0
        for conv in self.convs:
            out_image += conv(x)


        if self.bias:
            out_image += self.bias_weights
        
        return out_image

