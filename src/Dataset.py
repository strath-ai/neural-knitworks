from torch.utils.data import Dataset
from torchvision import io
import os

import torch
import torchvision
import numpy as np
import scipy.signal as sig

from tqdm.notebook import tqdm

class ShellDataset:
    def __init__(self, iterations = 100):
        self.iterations = iterations
    def __len__(self):
        return self.iterations
    def __getitem__(self,idx):
        return [None]
    
class SingleImageCoordinateDataset(Dataset):
    """Basic Dataset Class
    Args:
        dataframe:
        config: Global Configuration class
        validation: a flag to indicate validation set or not
    """

    def __init__(self,
                 image, 
                 mask = None,
                 coordinate_counts = None,
                 mapping = None,
                 embedding_length = 256,
                 kernel_margin = [0,0],
                 kernel_scales = [1],
                 residual_mode = False,
                 antialias = True,
                 resolution = 1,
                 repetition = 1):
        super().__init__()
        
        # basic params
        self.set_image(image)
        self.no_of_coordinates = len(self.image.shape) - 1
        self.coordinate_counts = torch.tensor(coordinate_counts) if coordinate_counts is not None else None
        self.set_mask(mask)
        self.kernel_margin = kernel_margin
        self.kernel_scales = kernel_scales
        self.residual = residual_mode
        self.resolution = resolution
        self.repetition = repetition
        
        # input mapping
        self.embedding_length = embedding_length
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
        # for scale of 1 the kernel is 1 x 1 resulting in no blurring
        if antialias and kernel_margin != [0,0]:
            self.scale_blurs = [torchvision.transforms.GaussianBlur([1+2*(scale-1),1+2*(scale-1)], sigma=scale) for scale in self.kernel_scales]       
            self.scale_images = []
            for blur in self.scale_blurs:
                self.scale_images.append(blur(self.image))
        else:
            self.scale_images = [self.image for scale in self.kernel_scales]
        
        
        self.inputs = []
        self.outputs = []
        
        if mapping is None:
            self.mapping = mapping
            self.input_length = self.no_of_coordinates
        else:
            if mapping == 'basic':
                self.mapping = torch.eye(self.no_of_coordinates)
            elif type(mapping) == int:
                self.mapping = torch.normal(0, 1,size = (self.embedding_length, self.no_of_coordinates))*mapping
            else:
                self.mapping = torch.eye(self.no_of_coordinates)
            self.input_length = self.mapping.shape[0] * 2
            
        self.precompute()

    def __getitem__(self, index: int):
        if self.mask is None:
            return [self.inputs, self.outputs]
        else:
            return [self.inputs, self.outputs, torch.squeeze(self.backprop_masks)]
            
    def precompute(self):
        self.missing_inputs = []
        self.output_masks = []
        
        pixel_count = self.image.shape[-2] * self.image.shape[-1]
        
        self.valid_patches = []        
        self.inputs = torch.zeros(pixel_count, self.no_of_coordinates)
        self.outputs = torch.zeros(pixel_count,
                                   self.image.shape[0],
                                   len(self.kernel_scales),
                                   1 + 2*self.kernel_margin[0],
                                   1 + 2*self.kernel_margin[1]
                                  )
        
        for index in tqdm(range(self.image.shape[-2] * self.image.shape[-1]), desc='Precomputing...'):
            if self.mask is None:
                x, y, v = self.__getval__(index)
                
                if v:
                    self.valid_patches.append(y)
            else:
                x, y, m = self.__getval__(index)
                self.output_masks.append(m)
                
                # entire patch contains the same mask
                if torch.all(m) or torch.all(m == 1):
                    self.valid_patches.append(y)

                # append to missing truth if center value unknown
                if self.kernel_margin != [0,0]:
                    m = m[0] if len(m.shape) > 2 else m
                    if not m[tuple(self.kernel_margin)]:
                        self.missing_inputs.append(x/(self.coordinate_counts*self.resolution))
                else:
                    if not m:
                        self.missing_inputs.append(x/(self.coordinate_counts*self.resolution))
                    
            # normalize idx to limit
            if self.coordinate_counts is not None:
                x = x / self.coordinate_counts
            self.inputs[index] = x/self.resolution
            self.outputs[index] = y
            
        if self.valid_patches != []:
            self.valid_patches = torch.stack(self.valid_patches)
        else:
            print('[WARNING] No valid patches found in the image.')
  
        if self.image.shape[0] == 1:
            self.outputs = self.outputs.unsqueeze(1)
            self.valid_patches = self.valid_patches.unsqueeze(1)

        if self.mask is not None:
            
            self.missing_inputs = torch.stack(self.missing_inputs)
            
            self.output_masks = torch.stack(self.output_masks)
            
            if len(self.output_masks.shape) == 3:
                self.output_masks = torch.unsqueeze(self.output_masks, dim = 1)
                
            if self.kernel_margin != [0,0]:
                self.core_factors = torch.stack([patch_mask[0][tuple(self.kernel_margin)] for patch_mask in self.output_masks])
                self.backprop_masks = torch.stack([self.output_masks[idx] == 1 for idx in range(len(self.core_factors))])
            else:
                self.core_factors = self.output_masks == 1
                self.backprop_masks = self.output_masks == 1
        
    def input_mapping(self, x, device = None):
        if self.mapping is None:
            return x
        else:
            if device is not None:
                self.mapping.to(device)
            
            if torch.is_tensor(x):
                x_proj = torch.mm(2.*self.pi*x, self.mapping.T)
                return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            else:
                x = np.array(x)
                x_proj = 2.*np.pi*x @ self.mapping.T.numpy()
                return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)

    def to_infer(self):
        return self.missing_inputs
            
    def __getindex__(self, index : int):
        s = self.image.shape[-2:]
        index_x = index % s[0]
        index_y = index // s[0]
        return [index_x, index_y]      

    def __getval__(self, index: int):
        idx = self.__getindex__(index)
        
        # initialize patch validity (false if some pixels are missing, like on the border)
        valid = True
        
        # get output with neighbourhood
        
        # a) initialize with zeros
        kernel_margin = self.kernel_margin
        output = np.zeros([self.image.shape[0], len(self.kernel_scales), 1+2*kernel_margin[0], 1 + 2*kernel_margin[1]])
        mask = np.zeros([len(self.kernel_scales), 1+2*kernel_margin[0], 1 + 2*kernel_margin[1]])
        
        for scale_idx in range(len(self.kernel_scales)):
            scale = self.kernel_scales[scale_idx]
            # set up single layer patch
            x_range = [idx[0] - kernel_margin[0]*scale, idx[0] + kernel_margin[0]*scale + 1]
            y_range = [idx[1] - kernel_margin[1]*scale, idx[1] + kernel_margin[1]*scale + 1]
            # for each pixel
            dead_pixels = []
            real_values = []
            for x_idx in range(*x_range, scale):
                for y_idx in range(*y_range, scale):
                    if 0 <= x_idx < self.image.shape[-2] and 0 <= y_idx < self.image.shape[-1]:
                        if self.residual and not (x_idx == idx[0] and y_idx == idx[1]):
                            # remove the central residual (predict on difference)
                            output[:, scale_idx, int((x_idx - x_range[0])/scale), int((y_idx - y_range[0])/scale)] = self.scale_images[scale_idx][:, x_idx, y_idx] - self.scale_images[scale_idx][:, idx[0], idx[1]]
                        else:
                            output[:, scale_idx, int((x_idx - x_range[0])/scale), int((y_idx - y_range[0])/scale)] = self.scale_images[scale_idx][:, x_idx, y_idx]

                        if self.mask is not None:
                            mask[scale_idx, int((x_idx - x_range[0])/scale), int((y_idx - y_range[0])/scale)] = self.mask[x_idx, y_idx]
                        real_values.append(self.image[:, x_idx, y_idx])
                    else:
                        dead_pixels.append([x_idx, y_idx])
            # fill dead pixels with random 'real' pixels
            real_tensor = torch.stack(real_values)
            real_tensor = real_tensor[torch.randperm(real_tensor.size(0))]
            for d_idx in range(len(dead_pixels)):
                x_idx, y_idx = dead_pixels[d_idx]
                output[:,scale_idx, int((x_idx - x_range[0])/scale), int((y_idx - y_range[0])/scale)] = real_tensor[d_idx % len(real_tensor)]
                if self.mask is not None:
                    mask[scale_idx, int((x_idx - x_range[0])/scale), int((y_idx - y_range[0])/scale)] = -1
                else:
                    valid = False
        
        #output = np.squeeze(output)
        #mask = np.squeeze(mask)
        
        if self.mask is None:
            return [torch.FloatTensor(idx),
                    torch.FloatTensor(output),
                    valid
                   ]
        else:
            return [torch.FloatTensor(idx),
                    torch.FloatTensor(output),
                    torch.FloatTensor(mask)
                   ]
        
    def get_patches(self, n = 1, rotation = False):
        n = int(n)

            
        if n != len(self.valid_patches):
            idxs = torch.randint(high = len(self.valid_patches), size = (n,))
            patches = self.valid_patches[idxs]
        else:
            patches = self.valid_patches
                
        if rotation:
            patches = torch.rot90(patches,
                                  int(torch.randint(4,(1,))),
                                  dims = [-2, -1])

        return patches
    
    def set_image(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)   
        if x.shape[-1] == 3:
            x = x.permute(2, 0, 1)
        x = x / x.max()
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        self.image = x
        
    def set_mask(self, x):
        if x is not None:
            if type(x) is not torch.Tensor:
                x = torch.Tensor(x)
        self.mask = x
        
    def __len__(self) -> int:
        return self.repetition
    