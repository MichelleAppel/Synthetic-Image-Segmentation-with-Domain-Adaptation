import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl

from src.domain_adaptation.models import patch_discriminator
from src.domain_adaptation.models import resnet_generator
from src.domain_adaptation.utils import ImagePool, init_weights, set_requires_grad

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import itertools
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.utils import make_grid

from src.domain_adaptation.models import patch_discriminator
from src.domain_adaptation.models import resnet_generator
from src.domain_adaptation.utils import ImagePool, init_weights, set_requires_grad

import wandb

class CycleGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # generator pair
        self.genX = resnet_generator.get_generator()
        self.genY = resnet_generator.get_generator()
        
        # discriminator pair
        self.disX = patch_discriminator.get_model()
        self.disY = patch_discriminator.get_model()
        
        self.lm = 10.0
        self.fakePoolA = ImagePool()
        self.fakePoolB = ImagePool()
        self.genLoss = None
        self.disLoss = None

        for m in [self.genX, self.genY, self.disX, self.disY]:
            init_weights(m)

        self.automatic_optimization = False

    def configure_optimizers(self):
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
        """
        if label.lower() == 'real':
            target = torch.ones_like(predictions)
        else:
            target = torch.zeros_like(predictions)
        
        return F.mse_loss(predictions, target)

    def generator_training_step(self, imgA, imgB, opt_gen):

        """cycle images - using only generator nets"""
        fakeB = self.genX(imgA)
        cycledA = self.genY(fakeB)

        fakeA = self.genY(imgB)
        cycledB = self.genX(fakeA)

        sameB = self.genX(imgB)
        sameA = self.genY(imgA)

        # generator genX must fool discrim disY so label is real = 1
        predFakeB = self.disY(fakeB)
        mseGenB = self.get_mse_loss(predFakeB, 'real')

        # generator genY must fool discrim disX so label is real
        predFakeA = self.disX(fakeA)
        mseGenA = self.get_mse_loss(predFakeA, 'real')

        # compute extra losses
        identityLoss = F.l1_loss(sameA, imgA) + F.l1_loss(sameB, imgB)

        # compute cycleLosses
        cycleLoss = F.l1_loss(cycledA, imgA) + F.l1_loss(cycledB, imgB)

        # gather all losses
        extraLoss = cycleLoss + 0.5 * identityLoss
        self.genLoss = mseGenA + mseGenB + self.lm * extraLoss
        self.log('gen_loss', self.genLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # store detached generated images
        self.fakeA = fakeA.detach()
        self.fakeB = fakeB.detach()

        self.manual_backward(self.genLoss)
        opt_gen.step()
        opt_gen.zero_grad()


        # Convert images to PyTorch tensors and add a batch dimension
        images = [imgA[0].unsqueeze(0), fakeB[0].unsqueeze(0), cycledA[0].unsqueeze(0), sameA[0].unsqueeze(0), 
                  imgB[0].unsqueeze(0), fakeA[0].unsqueeze(0),cycledB[0].unsqueeze(0), sameB[0].unsqueeze(0)]
        

        # Create a grid of images
        grid = make_grid(torch.vstack(images), nrow=4)  # Adjust 'nrow' as needed
        grid = transforms.ToPILImage()(grid)

        # Log the grid of images to W&B
        wandb.log({"Images": wandb.Image(grid, caption="Images")})

        return self.genLoss

    def discriminator_training_step(self, imgA, imgB, opt_dis):

        """Update Discriminator"""        
        fakeA = self.fakePoolA.query(self.fakeA)
        fakeB = self.fakePoolB.query(self.fakeB)

        # disX checks for domain A photos
        predRealA = self.disX(imgA)
        mseRealA = self.get_mse_loss(predRealA, 'real')

        predFakeA = self.disX(fakeA)
        mseFakeA = self.get_mse_loss(predFakeA, 'fake')

        # disY checks for domain B photos
        predRealB = self.disY(imgB)
        mseRealB = self.get_mse_loss(predRealB, 'real')

        predFakeB = self.disY(fakeB)
        mseFakeB = self.get_mse_loss(predFakeB, 'fake')

        # gather all losses
        self.disLoss = 0.5 * (mseFakeA + mseRealA + mseFakeB + mseRealB)
        self.log('dis_loss', self.disLoss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.manual_backward(self.disLoss)
        opt_dis.step()
        opt_dis.zero_grad()

        return self.disLoss

    def training_step(self, batch, batch_idx):
        imgA, imgB = batch['A'], batch['B']

        # Get the optimizers
        opt_gen, opt_dis = self.optimizers()

        # Generator training step
        set_requires_grad([self.disX, self.disY], False)
        loss_gen = self.generator_training_step(imgA, imgB, opt_gen)

        # Discriminator training step
        set_requires_grad([self.disX, self.disY], True)
        loss_dis = self.discriminator_training_step(imgA, imgB, opt_dis)

        wandb.log({"Generator loss": self.genLoss.item()})
        wandb.log({"Discriminator loss": self.disLoss.item()})

        return {'loss': loss_gen + loss_dis}

if __name__ == '__main__':
    model = CycleGAN()
