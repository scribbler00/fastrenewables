# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_gan.learner.ipynb (unless otherwise specified).

__all__ = ['GANLearner', 'evaluate_gan']

# Cell

import pandas as pd
import numpy as np
import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from fastai.basics import set_seed
from ..synthetic_data import *
from .model import *
from ..tabular.model import EmbeddingModule

from sklearn.model_selection import train_test_split

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

    def fit(self, dl, epochs=10, lr=1e-3, plot_epochs=10, save_model=False, save_dir='models/', save_file='tmp', figsize=(16, 9)):

        self.gan.to_device(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.gan.gen_optim.param_groups[0]['lr'] = lr
        self.gan.dis_optim.param_groups[0]['lr'] = lr

        self.gan.train()

        for e in tqdm(range(epochs)):
            for x_cat, x_cont, y in dl:
                x_cat = x_cat.to(self.gan.device).long()
                x_cont = x_cont.to(self.gan.device)
                y = y.to(self.gan.device)

                for _ in range(self.n_dis):
                    self.gan.train_discriminator(x_cat, x_cont, y)

                for _ in range(self.n_gen):
                    self.gan.train_generator(x_cat, x_cont, y)

            #if (e+1)%plot_epochs==0:
                #plt.figure(figsize=figsize)
                #plt.plot(self.gan.real_loss, label='Real Loss')
                #plt.plot(self.gan.fake_loss, label='Fake Loss')
                #if len(self.gan.aux_loss) > 0:
                #    plt.plot(self.gan.aux_loss, label='Aux Loss')
                #plt.legend()
                #plt.show()

                #fig, ax1 = plt.subplots(figsize=figsize)
                #ax1.set_xlabel('iterations')
                #ax1.set_ylabel('bce loss')
                #ax1.plot(self.gan.real_loss, label='real', color='red')
                #ax1.plot(self.gan.fake_loss, label='fake', color='blue')
                #ax1.tick_params(axis='y')
                #ax1.legend(loc='upper right')
                #
                #ax2 = ax1.twinx()
                #ax2.set_ylabel('aux loss')
                #ax2.plot(self.gan.aux_loss, label='aux', color='green')
                #ax2.tick_params(axis='y')
                #ax2.legend(loc='lower right')
                #
                #fig.tight_layout()
                #plt.show()

        self.gan.eval()

        if save_model:
            self.gan.to_device('cpu')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(self.gan.state_dict(), save_dir+save_file+'.pt')

        return

# Cell

def evaluate_gan(gan_type='bce', aux_factor=1, epochs=10):

    print(gan_type, aux_factor)
    set_seed(1337)
    emb_module = EmbeddingModule(categorical_dimensions=[n_classes+1])
    model = get_gan_model(structure=[n_z, n_hidden, n_hidden, n_in], n_classes=n_classes, emb_module=emb_module, gan_type=gan_type, aux_factor=aux_factor, label_noise=0.1, label_bias=0.25)
    learner = GANLearner(gan=model, n_gen=n_gen, n_dis=n_dis)
    learner.fit(train_dl, epochs=epochs, lr=lr, plot_epochs=epochs, save_model=True)
    for x_cat, x_cont, y in test_dl:
        x_cat = x_cat.long()
        print('distribution of real data:')
        d_real = fit_kde(x_cont, bandwidth=1/25, show_plot=True)
        x_fake = learner.generate_samples(x_cat, x_cont)
        print('distribution of generated data:')
        d_fake = fit_kde(x_fake, bandwidth=1/25, show_plot=True)
        break
    kld = calculate_kld(d_real, d_fake)

    return kld