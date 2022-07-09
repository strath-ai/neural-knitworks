from .mlp import MLP, PatchDiscriminator
from ..Dataset import *
from .LitMLP import *

import torch
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
import numpy as np

import copy

def collate_fn(batch):
    return tuple(zip(*batch))

class NeuralKnitwork(LitMLP):
    """
    NeuralKnitwork: Lightning Implementation of a Neural Knitwork Module
    
    * This module wraps the MLP object composed of two submodules:
        1. Patch MLP (coordinate to patch)
        2. Patch Reconstructor (patch to pixel)
        3. Patch Discriminator

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
    patch_reconstructor_width : int
        number of neurons in a single layer of the patch reconstructor
    patch_reconstructor_depth : int
        number of layers of the MLP of the patch reconstructor
    layer_width : int
        layer width of the patch MLP
    mapping : int,
        mapping applied to the input, indicates the number of Fourier Features if not None
    x_patch_loss_weight : list -> [l, a, b]
        cross-patch consistency loss weight growth
        a - asymptotic value (weight goes to a in the limit)
        b - amount of steps to go from 1% to 99% of a
        c - amount of steps at which the weight is 50% of a        
    lr : float
        default learning rate
    d_lr : float
        discriminator learning rate    
    epoch_steps : int
        number of steps between validation routine calls
    generator_steps : int
        number of steps optimizing generator for every discriminator optimizing step
    gan_weight : float
        weight of the gan loss components
    nov_coef : float
        weight of novel (synthesized) region in the gan loss
    kno_coef : float
        weight of known (non-synthesized) region in the gan loss

    Output
    -------
    patchReconstructor[ patchMLP ( coordinate ) ]
    
    """

    def __init__(self,
                 image,
                 mask = None,
                 kernel_margin = [2, 2],
                 kernel_scales = [1, 8],
                 antialias = True,
                 patch_reconstructor_weight = 1.0,
                 patch_reconstructor_depth = 8,
                 layer_width = 256,
                 mapping = 10,
                 x_patch_loss_weight = [1e-5, 0.5e3, 1e3],
                 lr = 1e-3,
                 d_lr = None,
                 epoch_steps = 1000,
                 gan_weight = 1e-4,
                 nov_coef = 1.0,
                 kno_coef = 1.0):
        
        super().__init__(image,
                         mask = mask,
                         kernel_margin = kernel_margin,
                         kernel_scales = kernel_scales,
                         antialias = antialias,
                         patch_reconstructor_depth = patch_reconstructor_depth,
                         patch_reconstructor_weight = patch_reconstructor_weight,
                         patch_mlp_width = layer_width,
                         patch_mlp_depth = 4,
                         x_patch_loss_weight = x_patch_loss_weight,
                         mapping = mapping,
                         lr = lr,
                         epoch_steps = epoch_steps)
        
        self.discriminator = PatchDiscriminator(input_size = [1+2*kernel_margin[0],1+2*kernel_margin[1]],
                                                scales = len(kernel_scales),
                                                channels = self.out_channels
                                               )      
        self.gan_weight = gan_weight
        
        
        self.d_lr = d_lr if d_lr is not None else lr
        
        
        self.kno_coef = kno_coef
        self.nov_coef = nov_coef
        
    def on_fit_start(self):
        self.train_dataset.inputs = self.train_dataset.inputs.to(self.device)
        self.train_dataset.outputs = self.train_dataset.outputs.to(self.device)
        self.train_dataset.mapping = self.train_dataset.mapping.to(self.device)
        if self.train_dataset.mask is not None:
            self.train_dataset.backprop_masks = self.train_dataset.backprop_masks.to(self.device)
            self.train_dataset.missing_inputs = self.train_dataset.missing_inputs.to(self.device)
        
    def get_g_loss(self, batch, batch_idx):
        if self.mask is None:
            points, targets = batch
            
            points = points[0]
            targets = targets[0]
            
            if self.patch_mode() and self.patch_reconstructor:
                out, pixel = self.forward(points, transform = True)
            else:
                out = self.forward(points, transform = True)
            
            # 1A reconstruction loss
            loss = self.model.loss_function(out, targets)
            
            # 1B cross-patch consistency
            if self.patch_mode() and self.x_patch_loss_weight[0] > 0.0:
                
                x_patch_loss = self.model.x_patch_loss(scope = (0,self.image.shape[0],0,self.image.shape[1]))
                loss['Patch Consistency Loss'] = x_patch_loss
                
                progress = 1/(1 + np.exp(-(10/self.x_patch_loss_weight[1])*(self.global_step-self.x_patch_loss_weight[2])))
                
                loss['Patch Consistency Loss Weight'] = progress
                    
                loss['loss'] += x_patch_loss*self.x_patch_loss_weight[0]*progress
                
            
            # 1C patch recombination loss
            if self.patch_reconstructor and self.patch_mode():
                if len(self.kernel_scales) > 1:
                    pixel_target = targets[:,:,0, self.kernel_margin[0], self.kernel_margin[1]]
                else:
                    pixel_target = targets[:,:, self.kernel_margin[0], self.kernel_margin[1]]

                # compute pixel loss
                pixel_loss = self.model.loss_function(pixel, pixel_target)

                loss['Pixel Restoration Loss'] = pixel_loss['loss']*self.pixel_restoration_weight
                loss['loss'] += pixel_loss['loss']
            
            # 2 gan_loss
            #fake_patches = out[torch.randint(0, out.shape[0], (self.no_of_patches,)),:,:,:]
            fake_patches = out
            
            # extend single scale
            if len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = 2)
            
            gen_loss = F.binary_cross_entropy(self.discriminator(fake_patches), torch.ones(len(fake_patches),1).to(self.device))
            
            # update loss
            loss['Generator Loss'] = gen_loss
            loss['loss'] += self.gan_weight*gen_loss
            
        else:
            points, targets, backprop_mask = batch                        
            
            points = points[0]
            targets = targets[0]
            backprop_mask = backprop_mask[0]
            
            if self.patch_mode() and self.patch_reconstructor:
                out, pixel = self.forward(points, transform = True)
            else:
                out = self.forward(points, transform = True)
        
            # 1A reconstruction loss
            loss = self.model.loss_function(out, targets, mask = backprop_mask)
            
            # 1B cross-patch consistency
            if self.patch_mode() and self.x_patch_loss_weight[0] > 0.0:
                
                x_patch_loss = self.model.x_patch_loss(scope = (0,self.image.shape[0],0,self.image.shape[1]))
                loss['Patch Consistency Loss'] = x_patch_loss
                
                progress = 1/(1 + np.exp(-(10/self.x_patch_loss_weight[1])*(self.global_step-self.x_patch_loss_weight[2])))
                
                loss['Patch Consistency Loss Weight'] = progress
                
                loss['loss'] += x_patch_loss*self.x_patch_loss_weight[0]*progress
                
            # 1C patch recombination loss
            if self.patch_reconstructor and self.patch_mode():
                if len(self.kernel_scales) > 1:
                    pixel_target = targets[:,:,0, self.kernel_margin[0], self.kernel_margin[1]]
                    pixel_mask = backprop_mask[:,0, self.kernel_margin[0], self.kernel_margin[1]]
                else:
                    pixel_target = targets[:,:, self.kernel_margin[0], self.kernel_margin[1]]
                    pixel_mask = backprop_mask[:, self.kernel_margin[0], self.kernel_margin[1]]
                
                # compute pixel loss
                pixel_loss = self.model.loss_function(pixel, pixel_target, mask = pixel_mask)

                loss['Pixel Restoration Loss'] = pixel_loss['loss']*self.pixel_restoration_weight
                loss['loss'] += pixel_loss['loss']
            
            # 2 gan_loss (known region)
            fake_patches = out
            
            # extend single scale
            if len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = 2)
            
            kno_loss = F.binary_cross_entropy(self.discriminator(fake_patches), torch.ones(len(fake_patches),1).to(self.device))
            
            # 3. gan_loss (novel region)
            points = self.train_dataset.to_infer()
            
            if self.patch_mode() and self.patch_reconstructor:
                out, pixel = self.forward(points, transform = True)
            else:
                out = self.forward(points, transform = True)
                
            nov_loss = F.binary_cross_entropy(self.discriminator(out), torch.ones(out.shape[0],1).to(self.device))
            
            # update loss
            loss['Generator Known Region Loss'] = kno_loss
            loss['Generator Novel Region Loss'] = nov_loss
            gen_loss = self.kno_coef*kno_loss + self.nov_coef*nov_loss
            loss['Generator Loss'] = gen_loss
            loss['loss'] += self.gan_weight*gen_loss
            
        return loss
    
    def get_d_loss(self, batch, batch_idx):
        if self.mask is None:
            points, targets = batch
            
            points = points[0]
            targets = targets[0]
            
            if self.patch_mode() and self.patch_reconstructor:
                out, _ = self.forward(points, transform = True)
            else:
                out = self.forward(points, transform = True)
           
            #fake_patches = out[torch.randint(0, out.shape[0], (self.no_of_patches,)),:,:,:]
            fake_patches = out
            real_patches = self.train_dataset.get_patches(len(fake_patches)).to(self.device)

            # extend single scale or single color
            while len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = -3)
            while len(real_patches.shape) < 5:
                real_patches = real_patches.unsqueeze(dim = -3)
                
            mixed_patches = torch.cat([fake_patches, real_patches])
            
            mixed_truth = torch.cat([torch.zeros(len(fake_patches), 1),
                                     torch.ones(len(real_patches), 1)]).to(self.device)
                    
            disc_loss = F.binary_cross_entropy(self.discriminator(mixed_patches), mixed_truth)
            
            loss = {'loss': disc_loss*self.gan_weight, 'Discriminator Loss': disc_loss}  
        else:
            points, targets, backprop_mask = batch
            
            points = points[0]
            targets = targets[0]
            backprop_mask = backprop_mask[0]
            
            if self.patch_mode() and self.patch_reconstructor:
                out, _ = self.forward(points, transform = True)
            else:
                out = self.forward(points, transform = True)
        
            # 1. gan_loss
            fake_patches = out
            
            real_patches = self.train_dataset.get_patches(len(out)).to(self.device)
            
            # extend single scale
            if len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = 2)
                real_patches = real_patches.unsqueeze(dim = 2)
                
            mixed_patches = torch.cat([fake_patches, real_patches])            
            mixed_truth = torch.cat([torch.zeros(len(fake_patches), 1),
                                     torch.ones(len(real_patches), 1)]).to(self.device)
            
            kno_loss = F.binary_cross_entropy(self.discriminator(mixed_patches), mixed_truth)
            
            # 2. gan_loss (novel view)
            points = self.train_dataset.to_infer()
            
            if self.patch_mode() and self.patch_reconstructor:
                out, pixel = self.forward(points, transform = True)
            else:
                out = self.forward(points, transform = True)
                
            fake_patches = out
            real_patches = self.train_dataset.get_patches(len(out)).to(self.device)
            
            # extend single scale
            if len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = 2)
                real_patches = real_patches.unsqueeze(dim = 2)
                
            mixed_patches = torch.cat([fake_patches, real_patches])
            mixed_truth = torch.cat([torch.zeros(len(fake_patches), 1),
                                     torch.ones(len(real_patches), 1)]).to(self.device)
            
            nov_loss = F.binary_cross_entropy(self.discriminator(mixed_patches), mixed_truth)
            
            # update
            disc_loss = self.kno_coef*kno_loss + self.nov_coef*nov_loss
            loss = {'loss': disc_loss*self.gan_weight,
                    'Discriminator Loss': disc_loss,
                    'Discriminator Known Region Loss': kno_loss,
                    'Discriminator Novel Region Loss': nov_loss
                   }  
            
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx = 1):
        batch = self.get_batch()        
        # generator training
        if optimizer_idx == 1:
            loss = self.get_g_loss(batch, batch_idx)                            
        # discriminator
        elif optimizer_idx == 0:
            loss = self.get_d_loss(batch, batch_idx)         
        return loss
    
    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr)
        
        return ({"optimizer": optimizer_d
                },
                {"optimizer": optimizer_g
                })