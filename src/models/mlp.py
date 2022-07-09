import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class PatchReconstructor(nn.Module):
    """
    Patch Reconstructor: MLP-based patch reconstructor
    (translates a patch into a single pixel color value)

    ...

    Parameters
    ----------
    scales : list of ints
        scales of the input patch
    margin : list of ints
        margin of the input patch
    channels : int
        number of color channels
    width : int
        number of neurons in a single layer
    depth : int
        number of layers of the MLP
    linear : bool
        if false, ReLU activation is used
    bias : bool
        if true, bias is used in the layers
    parallel : bool
        if true, all color channels are processed independently by single-color reconstruction model

    Output
    -------
    sigmoid(MLP(input_patch))
    
    """

    def __init__(self,
                 scales,
                 margin = [1, 1],
                 channels = 3,
                 width = 64,
                 depth = 8,
                 linear = True,
                 bias = False,
                 parallel = False):
        super().__init__()
        
        self.scales = scales
        self.margin = margin
        self.channels = channels
        self.width = width
        self.depth = depth
        self.linear = linear       
        self.bias = bias
        self.parallel = parallel
        
        if not parallel:
            self.model = self.make_model(self.channels)
            
        else:
            self.model = nn.ModuleList()
            
            for ch in range(channels):
                self.model.append(self.make_model(1))
                
    def make_model(self, channels):
        # entry layer
        comb_layers = [nn.Linear(channels * (1 + 2*self.margin[0]) * (1 + 2*self.margin[1]) * len(self.scales), self.width, bias = self.bias)]

        if not self.linear:
                comb_layers.append(torch.nn.ReLU())
        # core
        for layer in range(self.depth-2):
            comb_layers.append(
                torch.nn.Linear(self.width, self.width, bias = self.bias)
            )
            if not self.linear:
                comb_layers.append(torch.nn.ReLU())
        # output
        comb_layers.append(
            nn.Sequential(
                torch.nn.Linear(self.width, channels, self.bias),
                torch.nn.Sigmoid()
            )
        )
        return nn.Sequential(*comb_layers)
        
    def forward(self, input):
        if not self.parallel:
            return self.model(input.view(input.shape[0], -1))
        else:
            return torch.stack([self.model[ch](input[:,ch,...].view(input.shape[0], -1)) for ch in range(self.channels)], dim = 1)
    
class Sine(nn.Module):
    # Based on https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=Q43NiU-eJCv6
    
    def __init__(self, bias=True):
        super().__init__()

    def forward(self, input):
        return torch.sin(input)
        
class MLP(nn.Module):
    """
    MLP: Coordinate-based MLP with Internal Patch Representation
    (translates a coordinate into a stack of patches and single pixel color value)
    
    * This module is composed of two submodules:
        1. Patch MLP (coordinate to patch)
        2. Patch Reconstructor (patch to pixel)

    ...

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    patch_margin : list of ints
        margin of the input patch
    patch_scales : list of ints
        scales of the input patch
    patch_mlp_depth : int
        number of layers in the patch MLP
    patch_mlp_width : int
        layer width of the patch MLP
    patch_reconstructor : int
        Type [1-3] of the patch reconstructor
    parallel_reconstructor : bool
        Indicates if patch reconstruction is done separately for every channel
    patch_reconstructor_width : int
        number of neurons in a single layer of the patch reconstructor
    patch_reconstructor_depth : int
        number of layers of the MLP of the patch reconstructor
    mapping_fn : callback
        callback to the mapping function of the input coordinate

    Output
    -------
    patchReconstructor[ patchMLP ( coordinate ) ]
    
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 default_resolution = (256, 256),
                 patch_margin = [0,0],
                 patch_scales = [1],
                 patch_mlp_depth = 4,
                 patch_mlp_width = 256,
                 parallel_reconstructor = False,
                 patch_reconstructor_depth: int = 8,
                 patch_reconstructor_width: int = 8,
                 mapping_fn = None,
                 **kwargs):
        super().__init__()
        
        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels     
        self.default_resolution = default_resolution
        self.patch_margin = patch_margin
        self.patch_scales = patch_scales
        self.patch_mlp_depth = patch_mlp_depth
        self.patch_mlp_width = patch_mlp_width  
        self.patch_reconstructor_depth = max([patch_reconstructor_depth, 2])
        self.mapping_fn = mapping_fn
        self.out_features = self.out_channels * (1 + 2*patch_margin[0]) * (1 + 2*patch_margin[1]) * len(self.patch_scales)
        
        # Make Patch MLP model...
        if self.patch_mlp_depth == 1:
            layers = [nn.Sequential(
                    torch.nn.Linear(self.in_channels, self.out_features),
                    #torch.nn.ReLU(),
                )]
        else:
            # a) input
            layers = [nn.Sequential(
                        torch.nn.Linear(self.in_channels, self.patch_mlp_width),
                        torch.nn.ReLU(),
                    )]
            # b) core
            for layer in range(self.patch_mlp_depth-2):
                layers.append(
                    nn.Sequential(
                        torch.nn.Linear(self.patch_mlp_width, self.patch_mlp_width),
                        torch.nn.ReLU()
                    )
                )
            # c) output
            layers.append(
                    nn.Sequential(
                        torch.nn.Linear(self.patch_mlp_width, self.out_features),                    
                    )
            )

        layers.append(torch.nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.patch_reconstructor = not(self.patch_margin == None or self.patch_margin == [0, 0] or self.patch_margin == 0)
        
        # Patch Reconstructor
        if self.patch_reconstructor:
            self.reverse_model = PatchReconstructor(scales = self.patch_scales,
                                                     margin = patch_margin,
                                                     channels = self.out_channels,
                                                     width = patch_reconstructor_width,
                                                     depth = self.patch_reconstructor_depth,
                                                     linear = True,
                                                     bias = False,
                                                     parallel = parallel_reconstructor)            
        
        # Precompute comparison margins for cross-patch consistency
        self.compute_scale_limits()

    def forward(self, input, transform = True, patch_output = False, device = 'cpu', **kwargs):
        #device = next(self.model.parameters()).device
        
        if not torch.is_tensor(input):
            input = torch.FloatTensor(input)            
                
        if len(input.shape) > 2:
            input = torch.flatten(input, start_dim=0, end_dim=-2) 

        if transform and self.mapping_fn is not None:
            input = self.mapping_fn(input)
            
        #input = input.to(device)
            
        # Single Pixel
        if self.patch_margin == None or self.patch_margin == [0, 0] or self.patch_margin == 0:
            return self.model(input).view(input.shape[0], self.out_channels)
        # Patch
        else:
            # Single Patch
            if len(self.patch_scales) == 1:
                patches = self.model(input).view(input.shape[0], self.out_channels, 1 + 2*self.patch_margin[0], 1 + 2*self.patch_margin[1])
            # Multi-Scale Patch
            else:
                patches = self.model(input).view(input.shape[0], self.out_channels, len(self.patch_scales), 1 + 2*self.patch_margin[0], 1 + 2*self.patch_margin[1])
                
            if self.patch_reconstructor and not patch_output:
                return patches, self.reverse_model(patches.unsqueeze(2))
            else:
                return patches    
                
                
    
    def generate_image(self, inputs = None, resolution = None, coordinate_lims = (0,1), from_patches = False, center_pixel = True, device = torch.device("cpu")):
        """
        Parameters
        ----------
        inputs : tensor
            input coordinates
        resolution : list (H,W)
            number of coordinates to compute in each dimension (if inputs not provided)
        coordinate_lims : list
            range of signal coordinates, (0,1) is the original learned scope
        from_patches : bool
            option to average predictions from patches (if False (default), Patch Reconstructor is used)
        center_pixel : bool
            option to only extract center pixel reference from the patches
            
        Output
        -------
        patchReconstructor[ patchMLP ( coordinate ) ]
    
        """
        
        
        # initialize inputs if not provided
        if inputs is None:
            if resolution is None:
                resolution = self.default_resolution   
            x_range = np.linspace(coordinate_lims[0], coordinate_lims[1], resolution[0], endpoint=False)
            y_range = np.linspace(coordinate_lims[0], coordinate_lims[1], resolution[1], endpoint=False)

            if device == torch.device("cpu"):
                inputs = torch.FloatTensor([[x_idx, y_idx] for x_idx in x_range for y_idx in y_range])
            else:
                inputs = torch.cuda.FloatTensor([[x_idx, y_idx] for x_idx in x_range for y_idx in y_range], device = device)
                
        x_pixel_range = range(resolution[0])
        y_pixel_range = range(resolution[1])
        
        #  Get Output (if not from_patches, directly returns Patch Reconstructor output)      
        if self.patch_reconstructor:
            full_out, pixels = self.forward(inputs)
            
            if not from_patches:
                if self.out_channels > 1:
                    return pixels.view(len(x_pixel_range), len(y_pixel_range), self.out_channels).cpu().detach().numpy()
                else:
                    return pixels.view(len(x_pixel_range), len(y_pixel_range)).cpu().detach().numpy()
        else:
            full_out = self.forward(inputs)
       
        full_out = full_out.cpu().detach().numpy()

        image = [[[] for i in y_pixel_range] for j in x_pixel_range]

        # for each pixel
        for x_coord in x_pixel_range:
            for y_coord in y_pixel_range:

                out_idx = int((x_coord)*resolution[1]+ (y_coord))
                
                # index patch stack
                out = full_out[out_idx]

                # if patch-based...
                if self.patch_margin is not None and self.patch_margin != [0,0] and self.patch_margin != 0:
                    # for every scale
                    for scale_idx in range(len(self.patch_scales)):                          
                        scale = self.patch_scales[scale_idx]

                        if len(self.patch_scales) == 1:
                            scale_idx = None
                        
                        # 
                        if not center_pixel:  
                            for x_margin in range(1 + 2*self.patch_margin[0]):
                                for y_margin in range(1 + 2*self.patch_margin[1]):
                                    # get net index
                                    x_idx = x_coord + (x_margin-self.patch_margin[0])*scale
                                    y_idx = y_coord + (y_margin-self.patch_margin[1])*scale
                                    # but check if not 'out of border'
                                    if 0 <= x_idx <= x_pixel_range[-1] and 0 <= y_idx <= y_pixel_range[-1]:
                                        image[x_idx][y_idx].append(out[:,scale_idx,x_margin, y_margin])
                        else:
                            x_idx = x_coord
                            y_idx = y_coord
                            image[x_idx][y_idx].append(out[:,scale_idx,self.patch_margin[0], self.patch_margin[1]])
                else:
                    x_idx = x_coord
                    y_idx = y_coord
                    image[x_idx][y_idx].append(out)

        # average contributions
        if (not center_pixel) or (len(self.patch_scales) > 1):
            for row in image:
                for idx in range(len(row)):
                    row[idx] = np.squeeze(np.mean(np.array(row[idx]), axis = 0))

        return np.squeeze(np.array(image))
    
    def generate(self, **kwargs):
        return self.generate_image(**kwargs).clip(0,1)
        
    def loss_function(self,
                      output,
                      truth,
                      mask = None,
                      weight = 1.0,
                      **kwargs) -> dict:
        """
            Default loss function of the model
        """    
        
        output = torch.squeeze(output)
        truth = torch.squeeze(truth)
        recons_loss = F.mse_loss(output, truth, reduction = 'none')
        
        if self.out_channels > 1:
            grey_loss = torch.sum(recons_loss, dim = 1)
        else:
            grey_loss = recons_loss
        
        if mask is not None:
            mask = torch.squeeze(mask) 
            rec_loss = grey_loss[mask].mean()
        else:
            rec_loss = grey_loss.mean()
        
        loss = rec_loss * weight
            
        return {'loss': loss, 'Reconstruction Loss': rec_loss}
    
    def compute_scale_limits(self):
        """
            Computes appropriate crop for each scaled/shifted image from patch representation
            (used for Cross-Patch Consistency Loss)
        """
        self.scrop = []
        
        for scale_idx in range(len(self.patch_scales)):
            scale_element = {}
            
            scale = self.patch_scales[scale_idx]
            
            for x_shift in range(-self.patch_margin[0], 1 + self.patch_margin[0], 1):
                x_element = {}                
                for y_shift in range(-self.patch_margin[1], 1 + self.patch_margin[1], 1):
                    
                    x_element[y_shift] = {
                        'Margin': (max(0, -1*x_shift*scale),
                                   min(-1, -1-1*x_shift*scale),
                                   max(0, -1*y_shift*scale),
                                   min(-1, -1-1*y_shift*scale)
                                  ),
                        'Center': (max(0, x_shift*scale),
                                   min(-1, -1+x_shift*scale),
                                   max(0, y_shift*scale),
                                   min(-1, -1+y_shift*scale)
                                  )
                    }
                    
                scale_element[x_shift] = x_element
                    
            self.scrop.append(scale_element)
    
    def x_patch_loss_per_image(self, out, scope, reference_scale = 0, mask = None, residual = False, visualize = False):
        # reference_scale: if -1 then marginal predictions are compared to central prediction from that specific scale, otherwise to the index of the specified scale (basic 0 scale by default)
        out_space = out.view(scope[3] - scope[2], scope[1] - scope[0], *out.shape[1:]).swapaxes(0,1)#.permute(1,0,2,3,4)
            
        # truth reference -> center pixel
        truth = out_space[...,self.patch_margin[0], self.patch_margin[1]]

        # compute differences at overlaps...
        total_loss = 0.0
        
        # unsqueeze for single scale
        if len(self.patch_scales) == 1:
            out_space = torch.unsqueeze(out_space, dim = 3)
            truth = torch.unsqueeze(truth, dim = 3)
        
        for scale_idx in range(len(self.patch_scales)):
            scale = self.patch_scales[scale_idx]
            
            # option for selecting specific scale for prediction reference
            if reference_scale == -1:
                ref_scale_idx = scale_idx
            else:
                ref_scale_idx = reference_scale
                
            for x_shift in range(-self.patch_margin[0], 1 + self.patch_margin[0], 1):
                for y_shift in range(-self.patch_margin[1], 1 + self.patch_margin[1], 1):

                    margin_x_lo, margin_x_hi, margin_y_lo, margin_y_hi = self.scrop[scale_idx][x_shift][y_shift]['Margin']
                    center_x_lo, center_x_hi, center_y_lo, center_y_hi = self.scrop[scale_idx][x_shift][y_shift]['Center']
            
                    margin_pred = out_space[margin_x_lo:margin_x_hi, margin_y_lo:margin_y_hi, :, scale_idx, x_shift+self.patch_margin[0], y_shift + self.patch_margin[1]]
                    margin_ref = truth[margin_x_lo:margin_x_hi, margin_y_lo:margin_y_hi, :, ref_scale_idx]
                    
                    center_pred = truth[center_x_lo:center_x_hi, center_y_lo:center_y_hi, :, ref_scale_idx]
                    if mask is not None:
                        mask_ref = mask[center_x_lo:center_x_hi, center_y_lo:center_y_hi]

                    test_pred = margin_pred + margin_ref if residual else margin_pred
                    
                    if visualize:
                        return test_pred, center_pred, mask_ref
                        
                    if mask is not None:
                        margin_loss = ((1-mask_ref)*F.mse_loss(test_pred, center_pred, reduction = 'none').mean(-1)).sum()
                    else:
                        margin_loss = F.mse_loss(test_pred, center_pred, reduction = 'mean')

                    total_loss += margin_loss
                    
        return total_loss
    
    def x_patch_loss(self, scope):
        
        total_loss = 0.0
        
        x_size = scope[1] - scope[0]
        y_size = scope[3] - scope[2]
        scope_mesh = torch.tensor([[[x_idx/x_size, y_idx/y_size] for x_idx in range(*scope[:2])] for y_idx in range(*scope[2:])]).float()

        out, _ = self.forward(torch.flatten(scope_mesh,start_dim=0, end_dim=-2).to(next(self.model.parameters()).device), 
                           transform = True)

        total_loss += self.x_patch_loss_per_image(out, scope)
            
        return total_loss  
    
class PatchDiscriminator(nn.Module):
    """
    PatchDiscriminator: MLP-based Patch Discrimnator
    (produces a confidence score for a patch stack representation)

    ...

    Parameters
    ----------
    input_size : list
        basic patch size (default : [3, 3] resulting from a margin of 1)
    channels : int
        number of input channels (default: 3 for RGB)
    scales : int,
        number of scales in the patch representation (default: 1)
    informing_channels : int,
        number of informing channels (default: 0) - this functionality is in beta, used for conditioning patch generation on another source
    hidden_layers : list
        contains a list of channels sizes for each layer
    sigmoid : bool
        setting for output sigmoid function

    Output
    -------
    patchDiscriminator[ patchMLP ( coordinate ) ]
    
    """
    def __init__(self,
                 input_size = [3,3],
                 channels = 3,
                 scales = 1,
                 informing_channels = 0,
                 hidden_layers = [256, 256, 128, 64, 32, 16],
                 sigmoid = True
                ):
        super().__init__()
        
        layers = []
        curr_width = int(scales*(channels + informing_channels)*np.prod(input_size))
        
        for ch in hidden_layers:
            layers.append(nn.Linear(curr_width, ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))    
            curr_width = ch
            
        layers.append(nn.Linear(curr_width, 1))
        
        if sigmoid:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        self.model.requires_grad_(True)

    def forward(self, input):
        patches = input.view(input.size(0), -1)
        score = self.model(patches)

        return score