# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_gan.learner.ipynb (unless otherwise specified).

__all__ = ['GANLearner']

# Cell

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from fastai.basics import set_seed
from ..synthetic_data import *
from .model import *
from ..tabular.model import EmbeddingModule

import torch.nn.functional as F

plt.style.use('seaborn-colorblind')

#import glob

# Cell

class GANLearner():
    def __init__(self, gan, n_gen=1, n_dis=1):
        super(GANLearner, self).__init__()
        # gan should contain a class which itself contains a generator and discriminator/critic class and combines them
        self.gan = gan
        self.n_gen = n_gen
        self.n_dis = n_dis

    def generate_samples(self, x_cat, x_cont):
        with torch.no_grad():
            z = self.gan.noise(x_cont)
            fake_samples = self.gan.generator(x_cat, z)
        return fake_samples

    def fit(self, dl, epochs=10, lr=1e-3, plot_epochs=10, save_model=False):

        self.gan.to_device(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.gan.gen_optim.param_groups[0]['lr'] = lr
        self.gan.dis_optim.param_groups[0]['lr'] = lr

        self.gan.train()

        for e in tqdm(range(epochs)):
            for x_cat, x_cont, y in dl:
                #x_cat[:] = 0
                x_cat = x_cat.to(self.gan.device).long()
                x_cont = x_cont.to(self.gan.device)
                y = y.to(self.gan.device)

                for _ in range(self.n_dis):
                    self.gan.train_discriminator(x_cat, x_cont, y)

                for _ in range(self.n_gen):
                    self.gan.train_generator(x_cat, x_cont, y)

            if (e+1)%plot_epochs==0:
                plt.figure()
                plt.plot(self.gan.real_loss, label='Real Loss')
                plt.plot(self.gan.fake_loss, label='Fake Loss')
                if len(self.gan.aux_loss) > 0:
                    plt.plot(self.gan.aux_loss, label='Aux Loss')
                plt.legend()
                plt.show()

        self.gan.eval()

        if save_model:
            self.gan.to_device('cpu')

        return