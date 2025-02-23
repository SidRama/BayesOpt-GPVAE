from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from dataset_def import RotatedMNISTDataset


class ConvVAE(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(40 * 13 * 13, 500)
        self.fc21 = nn.Linear(500, 50)
        self.fc211 = nn.Linear(50, self.latent_dim)
        self.fc221 = nn.Linear(50, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(self.latent_dim, 50)
        self.fc31 = nn.Linear(50, 500)
        self.fc4 = nn.Linear(500, 40 * 13 * 13)

        # first transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=40, out_channels=20, kernel_size=4, stride=2, padding=1)

        # second transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """

        # convolution
        z = F.relu(self.conv1(x))
        z = self.pool1(z)
        z = F.relu(self.conv2(z))
        z = self.pool2(z)
        # MLP
        z = z.view(-1, 40 * 13 * 13)

        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc21(h1))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # MLP
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = x.view(-1, 40, 13, 13)
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        mu, log_var = self.encode(x)

        z = self.sample_latent(mu, log_var)

        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.mean(nll, dim=1)


class ConvVAE_partial(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False, X_dim=0):
        super(ConvVAE_partial, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim
        self.X_dim = X_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        self._cov_log_vy = nn.Parameter(torch.Tensor(X_dim * [log_vy_init]))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(60 * 6 * 6 + self.X_dim, 500)
        self.fc21 = nn.Linear(500, 50)
        self.fc211 = nn.Linear(50, self.latent_dim)
        self.fc221 = nn.Linear(50, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(self.latent_dim, 50)
        self.fc31 = nn.Linear(50, 500)
        self.fc4 = nn.Linear(500, 60 * 6 * 6)

        # first transposed convolution
        self.deconv1_n = nn.ConvTranspose2d(in_channels=60, out_channels=40, kernel_size=4, stride=2, padding=2,
                                            dilation=2)

        # second transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=40, out_channels=20, kernel_size=4, stride=2, padding=1)

        # third transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

        self.fc1_X = nn.Linear(self.X_dim, 40)
        self.fc2_X = nn.Linear(40, 20)
        self.fc21_X = nn.Linear(20, self.X_dim)
        self.fc22_X = nn.Linear(20, self.X_dim)

        self.log_scale = nn.Parameter(torch.zeros(self.num_dim))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def gaussian_likelihood(self, x_hat, x, mask):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_pxz = torch.mul(dist.log_prob(x), mask)
        return log_pxz.sum(-1)

    def encode(self, x, X_cov):
        """
        Encode the passed parameter

        :param x: input data
        :param X_cov: covariates
        :return: variational mean and variance
        """
        # convolution
        z = F.relu(self.conv1(x))
        z = self.pool1(z)
        z = F.relu(self.conv2(z))
        z = self.pool2(z)
        z = F.relu(self.conv3(z))
        z = self.pool3(z)
        # MLP
        z = z.view(-1, 60 * 6 * 6)
        X_cov = X_cov.view(-1, self.X_dim)

        h1 = F.relu(self.fc1(torch.cat((z, X_cov), 1)))
        h2 = F.relu(self.fc21(h1))

        # covolution for X
        h1_X = F.relu(self.fc1_X(X_cov))
        h2_X = F.relu(self.fc2_X(h1_X))

        return self.fc211(h2), self.fc221(h2), torch.sigmoid(self.fc21_X(h2_X)), torch.sigmoid(self.fc22_X(h2_X))

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        z = z.view(-1, self.latent_dim)
#        X_cov = X_cov.view(-1, self.X_dim)
        # MLP
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = x.view(-1, 60, 6, 6)
        x = F.relu(self.deconv1_n(x))
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, X_cov, X_cov_mask):
        mu, log_var, mu_X, log_var_X = self.encode(x, X_cov)
        z = self.sample_latent(mu, log_var)
        X_tilde = self.sample_latent(mu_X, log_var_X)
#        z_X = X_tilde * (1 - X_cov_mask) + X_cov * X_cov_mask
        return self.decode(z), mu, log_var, mu_X, log_var_X, X_tilde

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)

    def cov_loss_function(self, X_tilde_norm, covariates_norm, label_mask):
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(X_tilde_norm.view(-1, covariates_norm.shape[1]), covariates_norm.view(-1, covariates_norm.shape[1])), label_mask.view(-1, covariates_norm.shape[1]))
        mask_sum = torch.sum(label_mask.view(-1, covariates_norm.shape[1]), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse_X = torch.sum(torch.sum(se, dim=1) / mask_sum)

        nll_X = se / (2 * torch.exp(self._cov_log_vy))
        nll_X += 0.5 * (np.log(2 * math.pi) + self._cov_log_vy)
        return mse_X, torch.sum(nll_X, dim=1)


class ConvVAE_mod1(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False):
        super(ConvVAE_mod1, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv3 = nn.Conv2d(40, 60, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(60 * 6 * 6, 500)
        self.fc21 = nn.Linear(500, 50)
        self.fc211 = nn.Linear(50, self.latent_dim)
        self.fc221 = nn.Linear(50, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(self.latent_dim, 50)
        self.fc31 = nn.Linear(50, 500)
        self.fc4 = nn.Linear(500, 60 * 6 * 6)

        # first transposed convolution
        self.deconv1_n = nn.ConvTranspose2d(in_channels=60, out_channels=40, kernel_size=4, stride=2, padding=2,
                                            dilation=2)

        # second transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=40, out_channels=20, kernel_size=4, stride=2, padding=1)

        # third transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """

        # convolution
        z = F.relu(self.conv1(x))
        z = self.pool1(z)
        z = F.relu(self.conv2(z))
        z = self.pool2(z)
        z = F.relu(self.conv3(z))
        z = self.pool3(z)
        # MLP
        z = z.view(-1, 60 * 6 * 6)

        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc21(h1))
        return self.fc211(h2), self.fc221(h2)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # MLP
        x = F.relu(self.fc3(z))
        x = F.relu(self.fc31(x))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = x.view(-1, 60, 6, 6)
        x = F.relu(self.deconv1_n(x))
        x = F.relu(self.deconv1(x))
        return torch.sigmoid(self.deconv2(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        mu, log_var = self.encode(x)

        z = self.sample_latent(mu, log_var)

        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.sum(nll, dim=1)


class ConvVAE_mod2(nn.Module):
    """
    Encoder and decoder for variational autoencoder with convolution and transposed convolution layers.
    Modify according to dataset.

    For pre-training, run: python VAE.py --f=path_to_pretraining-config-file.txt
    """

    def __init__(self, latent_dim, num_dim, vy_init=1, vy_fixed=False):
        super(ConvVAE_mod2, self).__init__()

        self.latent_dim = latent_dim
        self.num_dim = num_dim

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        # encoder network
        # first convolution layer
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)
#        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # second convolution layer
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
#        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)


        self.fc1 = nn.Linear(128 * 1 * 1, 64)
#        self.fc21 = nn.Linear(120, 84)
        self.fc211 = nn.Linear(64, self.latent_dim)
        self.fc221 = nn.Linear(64, self.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(self.latent_dim, 64)
#        self.fc31 = nn.Linear(84, 120)
        self.fc4 = nn.Linear(64, 128 * 1 * 1)

        # first transposed convolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=1)

        # second transposed convolution
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.register_buffer('min_log_vy', min_log_vy * torch.ones(1))

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert torch.min(torch.tensor(vy)) >= 0.0005, "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x):
        """
        Encode the passed parameter

        :param x: input data
        :return: variational mean and variance
        """

        # convolution
        z = F.relu(self.conv1(x))
#        z = self.pool1(z)
        z = F.relu(self.conv2(z))
#        z = self.pool2(z)
        z = F.relu(self.conv3(z))
        z = F.relu(self.conv4(z))
        z = F.relu(self.conv5(z))
        # MLP
        z = z.view(-1, 128 * 1 * 1)

        h1 = F.relu(self.fc1(z))
#        h2 = F.relu(self.fc21(h1))
        return self.fc211(h1), self.fc221(h1)

    def decode(self, z):
        """
        Decode a latent sample

        :param z:  latent sample
        :return: reconstructed data
        """
        # MLP
        x = F.relu(self.fc3(z))
#        x = F.relu(self.fc31(x))
        x = F.relu(self.fc4(x))

        # transposed convolution
        x = x.view(-1, 128, 1, 1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        return torch.sigmoid(self.deconv5(x))

    def sample_latent(self, mu, log_var):
        """
        Sample from the latent space

        :param mu: variational mean
        :param log_var: variational variance
        :return: latent sample
        """
        # generate samples
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        mu, log_var = self.encode(x)

        z = self.sample_latent(mu, log_var)

        return self.decode(z), mu, log_var

    def loss_function(self, recon_x, x, mask):
        """
        Reconstruction loss

        :param recon_x: reconstruction of latent sample
        :param x:  true data
        :param mask:  mask of missing data samples
        :return:  mean squared error (mse) and negative log likelihood (nll)
        """
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        loss = nn.MSELoss(reduction='none')
        se = torch.mul(loss(recon_x.view(-1, self.num_dim), x.view(-1, self.num_dim)), mask.view(-1, self.num_dim))
        mask_sum = torch.sum(mask.view(-1, self.num_dim), dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        nll = se / (2 * torch.exp(self._log_vy))
        nll += 0.5 * (np.log(2 * math.pi) + self._log_vy)
        return mse, torch.mean(nll, dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device: {}'.format(device))
    csv_file_data = 'dataset_2000/train_data.csv'
    csv_file_label = 'dataset_2000/train_labels.csv'
    mask_file = 'dataset_2000/train_mask.csv'
    data_source_path = './data'
    latent_dim = 32
    num_dim = 2704
    vy_fixed = False
    vy_init = 1
    batch_size = 200
    loss_function = 'nll'
    weight = 0.01
    epochs = 20
    save_path = './results'
    dataset = RotatedMNISTDataset(csv_file_data, csv_file_label, mask_file, root_dir=data_source_path,
                                  transform=transforms.ToTensor())

    print('Length of dataset:  {}'.format(len(dataset)))
    N = len(dataset)

    print('Using convolutional neural network')
    nnet_model = ConvVAE(latent_dim, num_dim, vy_init=vy_init, vy_fixed=vy_fixed).double().to(device)
    nnet_model = nnet_model.double().to(device)

    dataloader_BO = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    n_batches = (N + batch_size - 1) // (batch_size)

    adam_param_list = []
    adam_param_list.append({'params': nnet_model.parameters()})
    optimiser = torch.optim.Adam(adam_param_list, lr=1e-3)
    for epoch in range(1, epochs + 1):
        recon_loss_sum = 0
        nll_loss_sum = 0
        kld_loss_sum = 0
        net_loss_sum = 0
        for batch_idx, sample_batched in enumerate(dataloader_BO):

            optimiser.zero_grad()
            nnet_model.train()
            indices = sample_batched['idx']
            data = sample_batched['digit'].double().to(device)
            train_x = sample_batched['label'].double().to(device)
            mask = sample_batched['mask'].double().to(device)
            N_batch = data.shape[0]

            recon_batch, mu, log_var = nnet_model(data)
            [recon_loss, nll] = nnet_model.loss_function(recon_batch, data, mask)
            recon_loss = torch.sum(recon_loss)
            nll_loss = torch.sum(nll)

            recon_loss = recon_loss * N / N_batch
            nll_loss = nll_loss * N / N_batch
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            if loss_function == 'nll':
                net_loss = nll_loss + kld_loss
    #            net_loss = nll_loss
            elif loss_function == 'mse':
                kld_loss = kld_loss / latent_dim
                net_loss = recon_loss + weight * kld_loss

            net_loss.backward()
            optimiser.step()

            net_loss_sum += net_loss.item() / n_batches
            recon_loss_sum += recon_loss.item() / n_batches
            nll_loss_sum += nll_loss.item() / n_batches
            kld_loss_sum += kld_loss.item() / n_batches

        print('Iter %d/%d - Loss: %.3f  - GP loss: %.3f  - NLL Loss: %.3f  - Recon Loss: %.3f' % (
             epoch, epochs, net_loss_sum, kld_loss_sum, nll_loss_sum, recon_loss_sum), flush=True)
    recon_arr = np.empty((0, 2704))
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(dataloader_BO):
            data = sample_batched['digit'].double().to(device)
            recon_batch, mu, log_var = nnet_model(data)
            recon_arr = np.append(recon_arr, nnet_model.decode(mu).reshape(-1, 2704).detach().cpu().numpy(), axis=0)

        fig, ax = plt.subplots(10, 10)
        for ax_ in ax:
            for ax__ in ax_:
                ax__.set_xticks([])
                ax__.set_yticks([])
        plt.axis('off')

        for i in range(0, 10):
            for j in range(0, 10):
                idx = i * 10 + j
                ax[i, j].imshow(np.reshape(recon_arr[idx], [52, 52]), cmap='gray')
        plt.savefig(os.path.join(save_path, 'reconstructions.pdf'), bbox_inches='tight')
        plt.close('all')
        pd.to_pickle([recon_arr],
                     os.path.join(save_path, 'diagnostics_plot.pkl'))





