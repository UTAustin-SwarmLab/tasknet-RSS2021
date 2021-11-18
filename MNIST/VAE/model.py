# this code was modified from online, need citation!!!

from collections import OrderedDict
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, label, image_size, channel_num, kernel_num, z_size, enc_layer_num=3):
        assert enc_layer_num > 1
        # configurations
        super().__init__()
        self.label = label
        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        print('self.image_size: ', self.image_size)
        print('self.channel_num: ', self.channel_num)
        print('self.kernel_num: ', self.kernel_num)

        # encoder
        enc_layers = OrderedDict()
        for i in range(enc_layer_num):
            if i == 0:
                enc_layers['conv_'+str(i+1)] = self._conv(channel_num, kernel_num // (2**(enc_layer_num-1)))
            elif i == enc_layer_num - 1:
                enc_layers['conv_'+str(i+1)] = self._conv(kernel_num // 2, kernel_num, last=True)
            else:
                enc_layers['conv_'+str(i+1)] = self._conv(kernel_num // (2**(enc_layer_num-i)), kernel_num // (2**(enc_layer_num-i-1)))
        self.encoder = nn.Sequential(enc_layers)

        # encoded feature's size and volume
        self.feature_size = image_size // 2**enc_layer_num
        self.feature_volume = kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, z_size, relu=False)

        # projection
        self.project = self._linear(z_size, self.feature_volume, relu=False)

        # decoder
        dec_layers = OrderedDict()
        for i in range(enc_layer_num):
            if i == 0:
                dec_layers['deconv_'+str(i+1)] = self._deconv(kernel_num, kernel_num // 2)
            elif i == enc_layer_num - 1:
                dec_layers['deconv_'+str(i+1)] = self._deconv(kernel_num // (2**(enc_layer_num-1)), channel_num, last=True)
            else:
                dec_layers['deconv_'+str(i+1)] = self._deconv(kernel_num // (2**i), kernel_num // (2**(i+1)))
        dec_layers['sigmoid'] = nn.Sigmoid()
        self.decoder = nn.Sequential(dec_layers)

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        #eps = (
        #    Variable(torch.randn(std.size())).cuda() if self.cuda else
        #    Variable(torch.randn(std.size()))
        #)
        #eps = torch.randn(std.size())
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{label}'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            label=self.label,
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        #z = Variable(
        #    torch.randn(size, self.z_size).cuda() if self.cuda else
        #    torch.randn(size, self.z_size)
        #)
        #z = torch.randn(size, self.z_size)
        z = torch.randn(size, self.z_size)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data

    #def _is_on_cuda(self):
    #    return self.cuda
        #return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num, last=False):
        conv = nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
        )
        return conv if last else nn.Sequential(
            conv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num, last=False):
        deconv = nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        )
        return deconv if last else nn.Sequential(
            deconv,
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)
