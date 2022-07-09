from ..LitMLP import *

from ...methods import *
from ...Dataset import *

import torch.nn.functional as F
from torch import nn
import numpy as np

# TEMPORARY
from pytorch_lightning.profiler import PassThroughProfiler

import torchvision.transforms as T

class BicubicDownsampler(nn.Module):
    # this module is roughly based on KernelGAN code
    # https://github.com/sefibk/KernelGAN/blob/master/util.py
    
    def __init__(self, downscale_factor = 2):
        super().__init__()
        
        self.scale_factor = downscale_factor
        self.bicubic_k = nn.parameter.Parameter(torch.tensor([[0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125,-0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [-.0013275146484375, -0.0039825439453130, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0050811767578125, -0.0152435302734375, 0.0491180419921875, 0.1880035400390630, 0.1880035400390630, 0.0491180419921875, -0.0152435302734375, -0.0050811767578125],
                     [-.0013275146484380, -0.0039825439453125, 0.0128326416015625, 0.0491180419921875, 0.0491180419921875, 0.0128326416015625, -0.0039825439453125, -0.0013275146484375],
                     [0.0004119873046875, 0.0012359619140625, -0.0039825439453125, -0.0152435302734375, -0.0152435302734375, -0.0039825439453125, 0.0012359619140625, 0.0004119873046875],
                     [0.0001373291015625, 0.0004119873046875, -0.0013275146484375, -0.0050811767578125, -0.0050811767578125, -0.0013275146484375, 0.0004119873046875, 0.0001373291015625]]),requires_grad = False)

    def forward(self, image):
        k = self.bicubic_k.expand(image.shape[1], image.shape[1], self.bicubic_k.shape[0], self.bicubic_k.shape[1])
        # Calculate padding
        padding = (k.shape[-1] - 1) // 2
        return F.conv2d(image, k, stride=self.scale_factor, padding=padding)

class LearnedDownsampler(nn.Module):
    
    # Downsampler based on KernelGAN implementation
    # https://github.com/sefibk/KernelGAN
    
    # from:
    # https://arxiv.org/abs/1909.06581
    
    def __init__(self,
                 channels = 3,
                 downscale_factor = 2,
                 kernel_sizes = [7, 5, 3, 1, 1, 1],
                 width = 64,
                 parallel = True,
                 sigmoid_out = False
                ):
        
        super().__init__()
        
        self.channels = channels
        self.downscale_factor = downscale_factor
        self.kernel_sizes = kernel_sizes
        self.depth = len(self.kernel_sizes)
        self.width = width
        self.parallel = parallel
        self.sigmoid_out = sigmoid_out
        
        self.model =  self.make_model(1 if self.parallel else self.channels)
                    
    def make_model(self, channels):
        layers = []
        
        if self.depth > 1:
            layers.append(nn.Conv2d(channels,
                                    self.width,
                                    kernel_size = self.kernel_sizes[0],
                                    stride=1,
                                    padding = int(0.5*(self.kernel_sizes[0]-1)),
                                    dilation=1,
                                    bias=False,
                                    padding_mode='replicate')
                         )
            
            for d in range(self.depth-1):
                out_ch = self.width if d < self.depth-2 else channels
                stride = 1 if d < self.depth-2 else self.downscale_factor
                
                layers.append(nn.Conv2d(self.width,
                                        out_ch,
                                        kernel_size = self.kernel_sizes[d+1],
                                        stride=stride,
                                        padding = int(0.5*(self.kernel_sizes[d+1]-1)),
                                        dilation= 1,
                                        bias = False,
                                        padding_mode = 'replicate')
                             )
        else:
            layers.append(nn.Conv2d(channels,
                                    channels,
                                    kernel_size = int(0.5*(self.kernel_sizes[0]-1)),
                                    stride=self.downscale_factor,
                                    padding = 'same',
                                    dilation=1,
                                    bias=False,
                                    padding_mode='replicate')
                         )
            
        if self.sigmoid_out:
            layers.append(nn.Sigmoid())
            
        self.delta = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        return nn.Sequential(*layers)

    def forward(self, input):
        if not self.parallel:
            return self.model(input)
        else:
            return torch.cat([self.model(input[:,ch,...].unsqueeze(0)) for ch in range(input.shape[1])], dim = 1)
        
    def net_kernel(self):
        
        for ind, w in enumerate(self.parameters()):
            curr_k = F.conv2d(self.delta, w, padding=self.kernel_sizes[0]-1) if ind == 0 else F.conv2d(curr_k, w, padding=1)
            
        # flip is an expensive operation, and avoiding it can be still give equivalent solution
        curr_k = curr_k.squeeze()#.flip([0, 1])
        
        return curr_k

class SuperResolver(LitMLP):

    def __init__(self,
                 image,
                 ground_truth = None,
                 upscale_factor = 2,
                 downsampling_mode = 'learned',
                 central_patch_downsampling = False,
                 kernel_margin = [1, 1],
                 kernel_scales = [1, 2, 4],
                 antialias = True,
                 patch_reconstructor_depth = 8,
                 sr_mean_match = False,
                 scale_shuffling = False,
                 patch_mlp_width = 256,
                 patch_mlp_depth = 4,
                 mapping = 10,
                 x_patch_loss_weight = [1e-2, 1e3, 0],
                 lr = 1e-3,
                 downsampler_lr = 1e-3,
                 epoch_steps = 1000,
                 label_smoothing = 0.1,
                 gan_weight = 1e-2,                
                 gan_lr_ratio = 1.0,
                 no_of_patches = None,
                 generator_steps = 1,
                 profiler = None):
        self.profiler = profiler or PassThroughProfiler()
        
        self.upscale_factor = upscale_factor
        self.downsampling_mode = downsampling_mode
        self.central_patch_downsampling = central_patch_downsampling
        self.ground_truth = ground_truth
        
        # force basic scale and combine (macro used for original pixels
        self.kernel_scales = kernel_scales

        super().__init__(image,
                         kernel_margin = kernel_margin,
                         kernel_scales = kernel_scales,
                         patch_mlp_width = patch_mlp_width,
                         patch_mlp_depth = patch_mlp_depth,
                         antialias = antialias,
                         patch_reconstructor_depth = patch_reconstructor_depth,
                         x_patch_loss_weight = x_patch_loss_weight,
                         mapping = mapping,
                         lr = lr,
                         epoch_steps = epoch_steps)        
        
        self.downsampler_lr = downsampler_lr

        self.softplus_discrimination = True
        self.disc_hidden_layers = [256, 256, 128, 64, 32, 16]
        self.gan_weight = gan_weight
        self.gan_lr_ratio = gan_lr_ratio    

        self.generator_steps = generator_steps        
        self.label_smoothing = label_smoothing
        self.rotate_patches = False                                           
                        
        self.no_of_patches = no_of_patches
        #self.bicubic = BicubicDownsampler(downscale_factor = self.upscale_factor)
        if self.downsampling_mode == 'learned':
            self.downsampler = LearnedDownsampler(downscale_factor = self.upscale_factor,
                                       width = 64,
                                       parallel = True,
                                       sigmoid_out = False
                                      )
        if self.downsampling_mode == 'dip-like':
            self.downsampler = Downsampler(n_planes=3,
                                           factor = self.upscale_factor,
                                           kernel_type = 'lanczos2',
                                           phase=0.5,
                                           preserve_size=True)
        
        
        self.discriminator = PatchDiscriminator(input_size = [1+2*self.kernel_margin[0],1+2*self.kernel_margin[1]],
                                                scales = len(self.kernel_scales),
                                                channels = self.out_channels,
                                                hidden_layers = self.disc_hidden_layers,
                                                sigmoid = not self.softplus_discrimination
                                               )                                       
                
        # sr-mesh
        s = self.image.shape[-2:]
        mesh_x, mesh_y = torch.meshgrid(torch.arange(0,(self.image.shape[0]*self.upscale_factor))/(self.image.shape[0]*self.upscale_factor),
                                        torch.arange(0,                                      (self.image.shape[1]*self.upscale_factor))/(self.image.shape[1]*self.upscale_factor)
                                       )
        self.sr_points = torch.stack([mesh_x, mesh_y], dim = 2)        
        
    def on_fit_start(self):

        self.train_dataset.inputs = self.train_dataset.inputs.to(self.device)
        self.sr_points = self.sr_points.to(self.device)
        self.train_dataset.outputs = self.train_dataset.outputs.to(self.device)
        self.train_dataset.valid_patches = self.train_dataset.valid_patches.to(self.device)
        if self.train_dataset.mapping is not None:
            self.train_dataset.mapping = self.train_dataset.mapping.to(self.device)
        self.train_dataset.image = self.train_dataset.image.to(self.device)
        
        if self.downsampling_mode == 'learned':
            self.downsampler.delta = self.downsampler.delta.to(self.device)
        
        if self.train_dataset.mask is not None:
            self.train_dataset.backprop_masks = self.train_dataset.backprop_masks.to(self.device)
        
    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([x['val_loss'] for x in outputs])
        self.log('val_loss', torch.tensor(avg_loss)) 
        
    def get_g_loss(self, batch, batch_idx):
        
        with self.profiler.profile("G Loss Computation"):
            points, targets = batch

            points = points[0]
            targets = targets[0]
            loss = {'loss' : 0.0}

            sr_out, sr_pixel = self.model.forward(torch.flatten(self.sr_points, 0, 1), transform = True)                            

            # patch recombination loss (LR image)
            if self.patch_reconstructor and self.patch_mode():
                pixel_target = targets[:,:,0, self.kernel_margin[0], self.kernel_margin[1]]


            # Adjusted for Super Resolution
            if self.x_patch_loss_weight[0] != 0.0:
                x_patch_loss_sr = self.model.x_patch_loss_per_image(sr_out,
                                                                    scope = (0,self.upscale_factor*self.image.shape[0],
                                                                             0,self.upscale_factor*self.image.shape[1])
                                                                   )

                x_patch_loss = x_patch_loss_sr #+ x_patch_loss_lr
                loss['Patch Consistency Loss'] = x_patch_loss

                progress = 1/(1 + np.exp(-(10/self.x_patch_loss_weight[1])*(self.global_step-self.x_patch_loss_weight[2])))

                loss['Patch Consistency Loss Weight'] = self.x_patch_loss_weight[0]*progress

                loss['loss'] += x_patch_loss*self.x_patch_loss_weight[0]*progress

            # 2 gan loss
            if self.no_of_patches is None:
                fake_patches = sr_out
                no_of_patches = len(fake_patches)
            else:
                no_of_patches = self.no_of_patches
                fake_patches = sr_out[torch.randint(high = len(sr_out), size = (no_of_patches,))]

            # extend single scale
            if len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = 2)

            truth = torch.ones(len(fake_patches),1)

            if not self.softplus_discrimination:
                 gen_loss = F.binary_cross_entropy(self.discriminator(fake_patches), truth)
            else:
                gen_loss = F.softplus(-self.discriminator(fake_patches)).mean()     

            # update loss
            loss['Generator Loss'] = gen_loss
            loss['loss'] += self.gan_weight*gen_loss   

            # downsampling

            # pad output sr_pixel with self.downsampling_margin
            sr_reshaped = sr_pixel.view(1,
                                        self.image.shape[0]*self.upscale_factor,
                                        self.image.shape[1]*self.upscale_factor,
                                        3).permute(0,3,1,2)
            
            # sr from central patch pixel
            if self.central_patch_downsampling:
                if len(self.kernel_scales) == 1:
                    sr_central = sr_out.view(self.upscale_factor*self.image.shape[1], self.upscale_factor*self.image.shape[0], *sr_out.shape[1:])[..., self.kernel_margin[0], self.kernel_margin[1]].permute(2,0,1).unsqueeze(0)
                else:
                    sr_central = sr_out.view(self.upscale_factor*self.image.shape[1], self.upscale_factor*self.image.shape[0], *sr_out.shape[1:])[..., 0, self.kernel_margin[0], self.kernel_margin[1]].permute(2,0,1).unsqueeze(0)
            
            #print(sr_reshaped.shape)
            #print(sr_central.shape)

            if self.downsampling_mode == 'learned':
                curr_k = self.downsampler.net_kernel()        
                #reverse = self.downsampler(sr_reshaped) 
                reverse = torch.nn.functional.conv2d(sr_reshaped.transpose(-2,-1),
                                                     weight = torch.stack(3*[curr_k.unsqueeze(0)]),
                                                     bias=None,
                                                     stride=self.upscale_factor,
                                                     padding=int((curr_k.shape[-1]-1)/2),
                                                     groups=3)[0].permute(0,2,1)                                

                # kernel regularization
                norm_loss = (1 - curr_k.sum()).abs()
                loss['Downsampling Kernel Regularization Loss'] = norm_loss
                loss['loss'] += norm_loss

            elif self.downsampling_mode == 'delta':
                reverse = sr_reshaped[0,:,::int(self.upscale_factor),::int(self.upscale_factor)]
            elif self.downsampling_mode == 'bilinear':
                reverse = F.interpolate(sr_reshaped.transpose(-2,-1), scale_factor = 1/self.upscale_factor, mode = 'bilinear')[0].permute(0,2,1)
            elif self.downsampling_mode == 'dip-like':
                reverse = self.downsampler(sr_reshaped)
                if self.central_patch_downsampling:
                    reverse_central = self.downsampler(sr_central)
            else:
                print('Downsampling Mode "" not recognized.'.format(self.downsampling_mode))

            downsampling_loss = self.model.loss_function(reverse, self.train_dataset.image)['loss']       
            
            if self.central_patch_downsampling:
                downsampling_loss += self.model.loss_function(reverse_central, self.train_dataset.image)['loss']  

            loss['Downsampling Loss'] = downsampling_loss

            loss['loss'] += downsampling_loss
            
            return loss
    
    def get_d_loss(self, batch, batch_idx):
        
        with self.profiler.profile("D Loss Computation"):
        
            points, targets = batch

            points = points[0]
            targets = targets[0]

            sr_out, sr_pixel = self.model.forward(torch.flatten(self.sr_points, 0, 1), transform = True)        

            if self.no_of_patches is None:
                fake_patches = sr_out
                no_of_patches = len(fake_patches)
            else:
                no_of_patches = self.no_of_patches
                fake_patches = sr_out[torch.randint(high = len(sr_out), size = (no_of_patches,))]

            real_patches = self.train_dataset.get_patches(no_of_patches, rotation = self.rotate_patches)

            # extend single scale
            while len(fake_patches.shape) < 5:
                fake_patches = fake_patches.unsqueeze(dim = -3)
            while len(real_patches.shape) < 5:    
                real_patches = real_patches.unsqueeze(dim = -3)        

            if not self.softplus_discrimination:
                sr_loss = F.binary_cross_entropy(self.discriminator(torch.cat([fake_patches, real_patches])),
                                                 torch.cat([torch.zeros(no_of_patches, 1),
                                                            (1-self.label_smoothing)*torch.ones(no_of_patches, 1)])
                                                )
            else:
                sr_loss = F.softplus(-self.discriminator(real_patches)).mean() + F.softplus(self.discriminator(fake_patches)).mean()

            # total
            disc_loss = sr_loss

            loss = {'loss': disc_loss*self.gan_weight,
                    'Discriminator Loss': disc_loss
                   }  
            
            return loss
    
    def get_batch(self):
        return [self.train_dataset.inputs], [self.train_dataset.outputs]

    def training_step(self, batch, batch_idx, optimizer_idx = 1):
        batch = self.get_batch()
        # generator training
        if optimizer_idx == 1:
            loss = self.get_g_loss(batch, batch_idx)    

        # discriminator
        elif optimizer_idx == 0:
            loss = self.get_d_loss(batch, batch_idx)

        return loss
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(ShellDataset(self.epoch_steps),
                                                   batch_size=1,
                                                   num_workers=2,
                                                   collate_fn = collate_fn,
                                                   shuffle=False
                                                  )
        
        return train_loader
    
    def configure_optimizers(self):
        g_params = [{'params': self.model.parameters(), 'lr' : self.lr}]
        
        if self.downsampling_mode == 'learned':
            g_params.append({'params': self.downsampler.parameters(), 'lr' : self.downsampler_lr})
        
        optimizer_g = torch.optim.Adam(g_params)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        return ({"optimizer": optimizer_d,
                 "frequency": 1
                },
                {"optimizer": optimizer_g,
                 "frequency": self.generator_steps
                })
    def generate(self, **kwargs):
        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = (self.image.shape[0]*self.upscale_factor, self.image.shape[1]*self.upscale_factor)
            
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