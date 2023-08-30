import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl
from torchvision.utils import make_grid
from torchvision import transforms

from src.domain_adaptation.cyclegan.models import patch_discriminator
from src.domain_adaptation.cyclegan.models import resnet_generator
from src.domain_adaptation.cyclegan.utils import ImagePool, init_weights, set_requires_grad

import wandb

from src.utils.timer import Timer

class CycleGAN(pl.LightningModule):
    def __init__(self, input_nc_genX=4, output_nc_genX=3, input_nc_genY=3, output_nc_genY=4, log_interval=5):
        super().__init__()
        # generator pair
        self.genX = resnet_generator.get_generator(input_nc=input_nc_genX, output_nc=output_nc_genX)
        self.genY = resnet_generator.get_generator(input_nc=input_nc_genY, output_nc=output_nc_genY)

        # discriminator pair
        self.disX = patch_discriminator.get_model(input_nc=3)
        self.disY = patch_discriminator.get_model(input_nc=3)

        # image pools
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()

        # hyperparameters
        self.lm = 10.0
        self.log_interval = log_interval

        # losses
        self.genLoss = None
        self.disLoss = None

        # Predefined tensors for efficiency
        self.ones = None  # to be defined in the training step where we know the shape
        self.zeros = None  # same as above
        self.dummy_depth = None

        for m in [self.genX, self.genY, self.disX, self.disY]:
            init_weights(m)

        self.automatic_optimization = False

    def configure_optimizers(self):
        """
            Configure optimizers and schedulers
            
            Returns:
                list: list of optimizers
                list: list of schedulers
        """

        optG = Adam(
            itertools.chain(self.genX.parameters(), self.genY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        optD = Adam(
            itertools.chain(self.disX.parameters(), self.disY.parameters()),
            lr=2e-4, betas=(0.5, 0.999))
        
        gamma = lambda epoch: 1 - max(0, epoch + 1 - 100) / 101
        schG = LambdaLR(optG, lr_lambda=gamma)
        schD = LambdaLR(optD, lr_lambda=gamma)
        return [optG, optD], [schG, schD]

    def get_mse_loss(self, predictions, label):
        """
            According to the CycleGan paper, label for
            real is one and fake is zero.

            Args:
                predictions (torch.Tensor): predictions from discriminator
                label (str): label for real or fake

            Returns:
                torch.Tensor: loss
        """
        if self.ones is None or self.ones.shape != predictions.shape:
            self.ones = torch.ones_like(predictions)
            self.zeros = torch.zeros_like(predictions)

        if label.lower() == 'real':
            target = self.ones
        else:
            target = self.zeros
        
        return F.mse_loss(predictions, target)

    def generator_training_step(self, imgA, imgB, opt_gen):
        """ Update Generator 
        
            Args:
                imgA (torch.Tensor): images from domain A
                imgB (torch.Tensor): images from domain B
                opt_gen (torch.optim): optimizer for generator
                
            Returns:
                torch.Tensor: loss
        """
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)

        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)

        if self.dummy_depth is None or self.dummy_depth.shape != imgA[:, 3:4, :, :].shape:
            self.dummy_depth = torch.zeros_like(imgA[:, 3:4, :, :])
        img_B_depth = torch.cat([imgB, self.dummy_depth], dim=1)

        sameB = self.genX(img_B_depth)
        sameA = self.genY(imgA[:, :3, :, :])

        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')

        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA[:, :3, :, :])
        mseGenA = self.get_mse_loss(predFakeA, 'real')

        # compute extra losses
        identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)

        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)

        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss

        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()

        self.manual_backward(self.genLoss)
        opt_gen.step()
        opt_gen.zero_grad()

        if self.global_step % self.log_interval == 0:
            self.log('gen_loss', self.genLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            # Convert images to PyTorch tensors and add a batch dimension
            images = [imgA[0].unsqueeze(0)[:, :3, :, :], fakeB[0].unsqueeze(0)[:, :3, :, :], cycledA[0].unsqueeze(0)[:, :3, :, :], sameA[0].unsqueeze(0)[:, :3, :, :], 
                    imgB[0].unsqueeze(0)[:, :3, :, :], fakeA[0].unsqueeze(0)[:, :3, :, :], cycledB[0].unsqueeze(0)[:, :3, :, :], sameB[0].unsqueeze(0)[:, :3, :, :]]
            
            # Create a grid of images
            grid = make_grid(torch.vstack(images), nrow=4)  # Adjust 'nrow' as needed
            grid = transforms.ToPILImage()((grid + 1) / 2)

            # Log the grid of images to W&B
            wandb.log({"Images": wandb.Image(grid, caption="Images")})

        return self.genLoss

    def discriminator_training_step(self, imgA, imgB, opt_dis):
        """ Update Discriminator

            Args:
                imgA (torch.Tensor): images from domain A
                imgB (torch.Tensor): images from domain B
                opt_dis (torch.optim): optimizer for discriminator

            Returns:
                torch.Tensor: loss
        """

        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)

        # disX checks for domain A photos
        predRealA = self.disX(imgA[:, :3, :, :])
        mseRealA = self.get_mse_loss(predRealA, 'real')

        predFakeA = self.disX(fakeA[:, :3, :, :])
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')

        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')

        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')

        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)

        self.manual_backward(self.disLoss)
        opt_dis.step()
        opt_dis.zero_grad()

        if self.global_step % self.log_interval == 0:
            self.log('dis_loss', self.disLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return self.disLoss

    def training_step(self, batch, batch_idx):
        ''' Training step for CycleGAN
        
            Args:
                batch (dict): batch of images from both domains
                batch_idx (int): index of batch
                
            Returns:
                dict: loss
        '''
        imgA, imgB = batch['A'], batch['B']

        # Get the optimizers
        opt_gen, opt_dis = self.optimizers()

        # Generator training step
        set_requires_grad([self.disX, self.disY], False)
        with Timer() as t:
            loss_gen = self.generator_training_step(imgA, imgB, opt_gen)

        if self.global_step % self.log_interval == 0:
            wandb.log({"Generator step time": t.interval})

        # Discriminator training step
        set_requires_grad([self.disX, self.disY], True)
        with Timer() as t:
            loss_dis = self.discriminator_training_step(imgA, imgB, opt_dis)

        if self.global_step % self.log_interval == 0:
            wandb.log({"Discriminator step time": t.interval})

        if self.global_step % self.log_interval == 0:
            wandb.log({"Generator loss": self.genLoss.item()}, step=batch_idx)
            wandb.log({"Discriminator loss": self.disLoss.item()}, step=batch_idx)

        return {'loss': loss_gen + loss_dis}

if __name__ == '__main__':
    model = CycleGAN()
