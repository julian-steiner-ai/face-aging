"""
Generative Adversarial Network.
"""

from torch import nn
from torch import full
from torch import randn
from torch import device as tdevice
from torch import float as tfloat
from torch.optim import Adam
from torch.utils.data import DataLoader

import torch
import torchvision.utils as vutils

class GAN:
    """
    Generative Adversarial Network.
    """
    def __init__(self, device, dataloader : DataLoader, n_gpus=0):
        self.device = device
        self.dataloader = dataloader
        self.n_gpus = n_gpus
        self._init_model()

    def _init_model(self):
        self.discriminator = Discriminator(3, 64).to(self.device)
        self.discriminator.apply(self._weighs_init)

        if (self.device.type == 'cuda') and (self.n_gpus > 1):
            self.discriminator = nn.DataParallel(self.discriminator, list(range(self.n_gpus)))

        self.generator = Generator(100, 64, 3).to(self.device)
        self.generator.apply(self._weighs_init)

        if (self.device.type == 'cuda') and (self.n_gpus > 1):
            self.generator = nn.DataParallel(self.generator, list(range(self.n_gpus)))

        self.criterion= nn.BCELoss()

        self.fixed_noise = randn(64, 100, 1, 1, device=self.device)

        self.real_label = 1.
        self.fake_label = 0.

        self.optimizer_discriminator = Adam(
            self.discriminator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )

        self.optimizer_generator = Adam(
            self.generator.parameters(),
            lr=0.0002,
            betas=(0.5, 0.999)
        )

    def train(self, n_epochs):
        """
        Train the model.
        """
        img_list = []
        generator_losses = []
        discriminator_losses = []
        iters = 0
        for epoch in range(n_epochs):
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.discriminator.zero_grad()

                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = full(
                    (b_size,),
                    self.real_label,
                    dtype=tfloat,
                    device=self.device
                )
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                discriminator_error_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                discriminator_error_real.backward()
                discriminator_x = output.mean().item()

                 ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = randn(b_size, 100, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                error_discriminator_fake = self.criterion(output, label)
                # Calculate the gradients for this batch,
                # accumulated (summed) with previous gradients
                error_discriminator_fake.backward()
                discriminator_gradients_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                discriminator_error = discriminator_error_real + error_discriminator_fake
                # Update D
                self.optimizer_discriminator.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                generator_error = self.criterion(output, label)
                # Calculate gradients for G
                generator_error.backward()
                discriminator_gradients_z2 = output.mean().item()
                # Update G
                self.optimizer_generator.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, n_epochs, i, len(self.dataloader),
                            discriminator_error.item(), generator_error.item(), discriminator_x, discriminator_gradients_z1, discriminator_gradients_z2))

                # Save Losses for plotting later
                generator_losses.append(generator_error.item())
                discriminator_losses.append(discriminator_error.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.generator(self.fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1

    def _weighs_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Generator.
    """
    def __init__(self, z_dim, n_feature_maps, n_channels):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.n_feature_maps = n_feature_maps
        self.n_channels = n_channels

        self._init_network()

    def _init_network(self):
        layer = []

        # input is Z, going into a convolution
        layer.append(nn.ConvTranspose2d(
            self.z_dim,
            self.n_feature_maps * 8,
            4,
            1,
            0,
            bias=False
        ))

        layer.append(nn.BatchNorm2d(self.n_feature_maps * 8))
        layer.append(nn.ReLU(True))

        # state size. ``(ngf*8) x 4 x 4``
        layer.append(nn.ConvTranspose2d(
            self.n_feature_maps * 8,
            self.n_feature_maps * 4,
            4,
            2,
            1,
            bias=False
        ))

        layer.append(nn.BatchNorm2d(self.n_feature_maps * 4))
        layer.append(nn.ReLU(True))

        # state size. ``(ngf*4) x 8 x 8``
        layer.append(nn.ConvTranspose2d(
            self.n_feature_maps * 4,
            self.n_feature_maps * 2,
            4,
            2,
            1,
            bias=False
        ))

        layer.append(nn.BatchNorm2d(self.n_feature_maps * 2))
        layer.append(nn.ReLU(True))

        # state size. ``(ngf*2) x 16 x 16``
        layer.append(nn.ConvTranspose2d(
            self.n_feature_maps * 2,
            self.n_feature_maps,
            4,
            2,
            1,
            bias=False
        ))

        layer.append(nn.BatchNorm2d(self.n_feature_maps))
        layer.append(nn.ReLU(True))

        # state size. ``(ngf) x 32 x 32``
        layer.append(nn.ConvTranspose2d(
            self.n_feature_maps,
            self.n_channels,
            4,
            2,
            1,
            bias=False
        ))

        layer.append(nn.Tanh())

        self.main = nn.Sequential(*layer)

    def forward(self, x):
        """
        Forward operation for the generator network.
        """
        return self.main(x)

class Discriminator(nn.Module):
    """
    Discriminator.
    """
    def __init__(self, n_channels, n_feature_maps):
        super(Discriminator, self).__init__()

        self.n_channels = n_channels
        self.n_feature_maps = n_feature_maps

        self._init_network()

    def _init_network(self):
        layers = []

        # input is ``(nc) x 64 x 64``
        layers.append(nn.Conv2d(
            self.n_channels,
            self.n_feature_maps,
            4,
            2,
            1,
            bias=False
        ))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # state size. ``(ndf) x 32 x 32``
        layers.append(nn.Conv2d(
            self.n_feature_maps,
            self.n_feature_maps * 2,
            4,
            2,
            1,
            bias=False
        ))
        layers.append(nn.BatchNorm2d(self.n_feature_maps * 2))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # state size. ``(ndf*2) x 16 x 16``
        layers.append(nn.Conv2d(
            self.n_feature_maps * 2,
            self.n_feature_maps * 4,
            4,
            2,
            1,
            bias=False
        ))
        layers.append(nn.BatchNorm2d(self.n_feature_maps * 4))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # state size. ``(ndf*4) x 8 x 8``
        layers.append(nn.Conv2d(
            self.n_feature_maps * 4,
            self.n_feature_maps * 8,
            4,
            2,
            1,
            bias=False
        ))
        layers.append(nn.BatchNorm2d(self.n_feature_maps * 8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # state size. ``(ndf*8) x 4 x 4``
        layers.append(nn.Conv2d(
            self.n_feature_maps * 8,
            1,
            4,
            1,
            0,
            bias=False
        ))
        layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward Operation.
        """
        return self.main(x)
