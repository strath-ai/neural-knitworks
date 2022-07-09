from .mlp import MLP, PatchDiscriminator
from ..Dataset import *

import torch
import torch.nn.functional as F
import torchvision

import omegaconf

import pytorch_lightning as pl
import numpy as np

import copy

def collate_fn(batch):
    return tuple(zip(*batch))

class LitMLP(pl.LightningModule):
    """
    LitMLP: Lightning Wrapper of Coordiante MLP Core of the Neural Knitwork (Patch MLP + Patch Reconstructor)
    
    * This module wraps the MLP object composed of two submodules:
        1. Patch MLP (coordinate to patch)
        2. Patch Reconstructor (patch to pixel)

    ...

    Parameters
    ----------
    image : numpy array or tensor
        source image
    mask : numpy array or tensor
        mask indicating which pixels are known
    depth : int
        number of stacked image frames (for multi-image tasks)
    kernel_scales : list of ints
        scales of the input patch
    kernel_margin : list of ints
        margin of the input patch
    antialias : bool
        option for antialiasing applied to higher scales
    patch_reconstructor : int
        Type [1-3] of the patch reconstructor
    parallel_reconstructor : bool
        Indicates if patch reconstruction is done separately for every channel
    patch_reconstructor_width : int
        number of neurons in a single layer of the patch reconstructor
    patch_reconstructor_depth : int
        number of layers of the MLP of the patch reconstructor
    num_layers : int
        number of layers in the patch MLP
    layer_width : int
        layer width of the patch MLP
    mapping_fn : callback
        callback to the mapping function of the input coordinate

    Output
    -------
    patchReconstructor[ patchMLP ( coordinate ) ]
    
    """

    def __init__(self,
                 image, 
                 mask = None,
                 kernel_margin = [0, 0], 
                 kernel_scales = [1],
                 antialias = True,                 
                 patch_mlp_width = 256,
                 patch_mlp_depth = 4,
                 patch_reconstructor_depth = 8,
                 patch_reconstructor_width = 8,
                 patch_reconstructor_weight = 1.0,
                 parallel_reconstructor = False,
                 mapping = 10,
                 embedding_length = 256,
                 reconstruction_loss_weight = 1.0,
                 x_patch_loss_weight = 0.0,
                 lr = 1e-3,
                 epoch_steps = 1000):
        super().__init__()
        
        if len(image.shape) > 2:
            self.in_channels = len(image.shape[:-1]) # x,y
            self.out_channels = image.shape[-1] # RGB  
            self.default_resolution = image.shape[:-1]
        else:
            self.in_channels = 2
            self.out_channels = 1
            self.default_resolution = image.shape
            
        self.coordinate_counts = image.shape[:-1]
        
        self.image = image        
        self.mask = mask
        
        # Default Configuration
            
        self.kernel_margin = kernel_margin
        self.kernel_scales = kernel_scales
        self.antialias = antialias
        self.mapping = mapping
        self.embedding_length = embedding_length

        self.patch_mlp_width = patch_mlp_width
        self.patch_mlp_depth = patch_mlp_depth

        if self.kernel_margin != [0,0]:
            self.patch_reconstructor = True
        else:
            self.patch_reconstructor = None

        self.patch_reconstructor_depth = patch_reconstructor_depth
        self.patch_reconstructor_width = patch_reconstructor_width

        if self.patch_reconstructor:
            self.pixel_restoration_weight = patch_reconstructor_weight

        self.reconstruction_loss_weight = reconstruction_loss_weight

        # x_patch_loss_weight should have the following format:
        # [l, a, b]
        # a - asymptotic value (weight goes to a in the limit)
        # b - amount of steps to go from 1% to 99% of a
        # c - amount of steps at which the weight is 50% of a

        self.x_patch_loss_weight = x_patch_loss_weight
        if not isinstance(self.x_patch_loss_weight, list):
            # sigmoid equation is still used with gradient 0.0, but multiplier has to be doubled to balance 1/2 from sigmoid
            self.x_patch_loss_weight = [self.x_patch_loss_weight*2, 0.0, 0.0]
        elif len(self.x_patch_loss_weight) == 1:
            self.x_patch_loss_weight = [self.x_patch_loss_weight[0]*2, 0.0, 0.0]
        elif len(self.x_patch_loss_weight) == 2:
            self.x_patch_loss_weight.append(0.0)            

        self.lr = lr
        self.epoch_steps = epoch_steps
            
        # Dataset & Model Instantiation
        if self.embedding_length is None:
            self.embedding_length = self.patch_mlp_width
            
        if self.mask is None:
            self.train_dataset = SingleImageCoordinateDataset(self.image,
                                                              mapping = self.mapping,
                                                              embedding_length = self.embedding_length,
                                                              coordinate_counts = self.coordinate_counts,
                                                              kernel_margin = self.kernel_margin,
                                                              kernel_scales = self.kernel_scales,
                                                              antialias = self.antialias,
                                                              repetition = self.epoch_steps
                                                             )                        
        else:
            self.train_dataset = SingleImageCoordinateDataset(self.image,
                                                              mask = self.mask,
                                                              mapping = self.mapping,
                                                              embedding_length = self.embedding_length,
                                                              coordinate_counts = self.coordinate_counts,
                                                              kernel_margin = self.kernel_margin,
                                                              kernel_scales = self.kernel_scales,
                                                              antialias = self.antialias,
                                                              repetition = self.epoch_steps
                                                         )
            
        self.valid_dataset = copy.deepcopy(self.train_dataset)
        self.valid_dataset.repetition = 1
            
        self.model = MLP(in_channels = self.train_dataset.input_length,
                         out_channels = self.out_channels,
                         default_resolution = self.default_resolution,
                         patch_margin = self.kernel_margin,
                         patch_mlp_depth = self.patch_mlp_depth,
                         patch_mlp_width = self.patch_mlp_width,                
                         patch_reconstructor = self.patch_reconstructor,
                         parallel_reconstructor = parallel_reconstructor,
                         patch_reconstructor_depth = self.patch_reconstructor_depth,
                         patch_reconstructor_width = self.patch_reconstructor_width,
                         patch_scales = self.kernel_scales,
                         mapping_fn = self.train_dataset.input_mapping
                        )
            
    def get_batch(self):
        if self.mask is None:
            return [self.train_dataset.inputs], [self.train_dataset.outputs]
        else:
            return [self.train_dataset.inputs], [self.train_dataset.outputs], [torch.squeeze(self.train_dataset.backprop_masks)]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(ShellDataset(self.epoch_steps),
                                           batch_size=1,
                                           num_workers=2,
                                           collate_fn = collate_fn,
                                           shuffle=False
                                          )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(ShellDataset(1),
                                           batch_size=1,
                                           num_workers=1,
                                           collate_fn = collate_fn,
                                           shuffle=False
                                          )
    
    def output(self):
        out = self.forward(self.input).detach().permute(0,2,3,1).numpy()
        if out.shape[-1] > 3:
            out_list = []
            for i in range(int(out.shape[-1]/3)):
                out_list.append(out[0,...,3*i:3*(i+1)])
            return out_list
        else:
            return out[0]
        
    def patch_mode(self):
         return self.kernel_margin != [0,0] and self.kernel_margin != 0 and self.kernel_margin != [0]

    def forward(self, input, **kwargs):
        return self.model.forward(input, **kwargs)
    
    def generate(self, **kwargs):
        if 'device' in kwargs.keys():
            if self.mapping is not None:
                device = kwargs['device']
                self.train_dataset.mapping = self.train_dataset.mapping.to(device)            
            self.model = self.model.to(device)
            return self.model.generate(**kwargs)
        else:
            if self.mapping is not None:
                self.train_dataset.mapping = self.train_dataset.mapping.to(self.device)
            return self.model.generate(device = self.device, **kwargs)
    
    def get_loss(self, batch, batch_idx):
        transform = True
        
        if self.mask is None:
            points, targets = batch

            if self.patch_mode() and self.patch_reconstructor:
                out, pixel = self.forward(points[0], transform = transform)
            else:
                out = self.forward(points[0], transform = transform)
             
            targets = targets[0]
            
            # reconstruction loss
            loss = self.model.loss_function(out, targets.unsqueeze(-1), weight = self.reconstruction_loss_weight)
        else:
            points, targets, backprop_mask = batch
            
            points = points[0]
            targets = targets[0]
            backprop_mask = backprop_mask[0]
            
            if self.patch_mode() and self.patch_reconstructor:
                out, pixel = self.forward(points, transform = transform)

            else:
                out = self.forward(points, transform = transform)                
                
            # reconstruction loss
            loss = self.model.loss_function(out, targets, mask = backprop_mask, weight = self.reconstruction_loss_weight)
            
            # cross-patch consistency
            if self.patch_mode() and self.x_patch_loss_weight[0] > 0.0:
                #x_patch_loss = self.model.x_patch_loss(scope = (0,self.image.shape[0],0,self.image.shape[1]))
                x_patch_loss = self.model.x_patch_loss_per_image(out,
                                                                 scope = (0,self.image.shape[0],
                                                                          0,self.image.shape[1]),
                                                                 #mask = self.train_dataset.mask,
                                                                )
                loss['Patch Consistency Loss'] = x_patch_loss
                
                progress = 1/(1 + np.exp(-(10/self.x_patch_loss_weight[1])*(self.global_step-self.x_patch_loss_weight[2])))
                
                loss['Patch Consistency Loss Weight'] = self.x_patch_loss_weight[0]*progress
                    
                loss['loss'] += x_patch_loss*self.x_patch_loss_weight[0]*progress
    
        # patch recombination loss
        if self.patch_reconstructor and self.patch_mode():
            pixel_target = targets[:,:,0, self.kernel_margin[0], self.kernel_margin[1]]            
            
            if self.mask is not None:
                if len(self.kernel_scales) > 1:
                    pixel_mask = backprop_mask[:, 0, self.kernel_margin[0], self.kernel_margin[1]]
                else:
                    pixel_mask = backprop_mask[:, self.kernel_margin[0], self.kernel_margin[1]]                  
            else:
                  pixel_mask = None             
                
            # compute pixel loss
            pixel_loss = self.model.loss_function(pixel, pixel_target, mask = pixel_mask)

            loss['Pixel Restoration Loss'] = pixel_loss['loss']*self.pixel_restoration_weight
            loss['loss'] += pixel_loss['loss']
            
            
        return loss

    def training_step(self, batch, batch_idx):
        batch = self.get_batch()        
        
        loss = self.get_loss(batch, batch_idx)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self.get_batch()        
        
        loss = self.get_loss(batch, batch_idx)                
        return {'val_loss': loss['loss'].detach().item()}
    
    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x['val_loss'] for x in outputs])
        
    def on_fit_start(self):
        self.train_dataset.inputs = self.train_dataset.inputs.to(self.device)
        self.train_dataset.outputs = self.train_dataset.outputs.to(self.device)
        if self.train_dataset.mask is not None:
            self.train_dataset.mask = self.train_dataset.mask.to(self.device)
            self.train_dataset.backprop_masks = self.train_dataset.backprop_masks.to(self.device)
        if self.train_dataset.mapping is not None:
            self.train_dataset.mapping = self.train_dataset.mapping.to(self.device)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=2,
                                                               threshold=1e-3,
                                                               threshold_mode='rel',
                                                               eps=1e-08
                                                              )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}