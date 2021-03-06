# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_gan.model.ipynb (unless otherwise specified).

__all__ = ['flatten_ts', 'LinBnAct', 'GANMLP', 'GAN', 'WGAN', 'AuxiliaryDiscriminator', 'get_gan_model']

# Cell

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from functools import partial
from torch.nn import BCELoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm

from ..synthetic_data import GaussianDataset, plot_class_hists
from ..timeseries.model import TemporalCNN
from ..tabular.model import EmbeddingModule

# Cell

def flatten_ts(x):
    """assumes matrix of shape (n_samples, n_features, ts_length)"""
    if len(x.shape) in [1,2]:
        return x

    n_samples, n_features, ts_length = x.shape

    if isinstance(x, np.ndarray):
        x = x.swapaxes(1,2)
    else:
        x = x.permute(0,2,1)
    x = x.reshape(n_samples*ts_length, n_features)
    return x

# Cell

def LinBnAct(si, so, use_bn, act_cls):
    layers = [nn.Linear(si, so)]
    if use_bn:
        layers += [nn.BatchNorm1d(so)]
    if act_cls is not None:
        layers += [act_cls]

    return nn.Sequential(*layers)

# Cell

class GANMLP(torch.nn.Module):
    def __init__(self, ann_structure, bn_cont=False, act_fct=torch.nn.ReLU, final_act_fct=nn.Sigmoid, embedding_module=None, transpose=False):
        super(GANMLP, self).__init__()

        n_cont = ann_structure[0]
        if embedding_module is not None:
            emb_sz = []
            ann_structure[0] = ann_structure[0] + embedding_module.no_of_embeddings

        self.embedding_module = embedding_module

        layers = []
        for idx in range(1, len(ann_structure)):
            cur_use_bn = bn_cont
            cur_act_fct = act_fct()
            if idx == 1 and not bn_cont:
                cur_use_bn = False
            if idx == len(ann_structure)-1:
                cur_act_fct = None
                #cur_use_bn = False

            layer = LinBnAct(ann_structure[idx-1], ann_structure[idx], cur_use_bn, cur_act_fct)
            layers.append(layer)
        if final_act_fct is not None:
            layers.append(final_act_fct())

        self.model = nn.Sequential(*layers)

    def forward(self, x_cat, x_cont):
        if self.embedding_module is not None:
            x_cat = self.embedding_module(x_cat)
            x_cont = torch.cat([x_cat, x_cont], 1)

        return self.model(x_cont)

# Cell

class GAN(nn.Module):

    def __init__(self, generator, discriminator, gen_optim, dis_optim, n_z=100, auxiliary=False, auxiliary_weighting_factor=1):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optim = gen_optim
        self.dis_optim = dis_optim
        self.n_z = n_z
        self.real_loss = []
        self.fake_loss = []
        self.aux_loss = []
        self.auxiliary = auxiliary
        self.bce_loss = BCELoss()
        self.auxiliary_loss_function = CrossEntropyLoss()
        self.auxiliary_weighting_factor=auxiliary_weighting_factor
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to_device(self.device)

    def noise(self, x):
        z = torch.randn(x.shape[0], self.n_z).to(self.device)
        return z

    def to_device(self, device):
        self.device = device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        self.bce_loss = self.bce_loss.to(device)
        self.auxiliary_loss_function = self.auxiliary_loss_function.to(device)

    def _split_pred(self, y):
        if self.auxiliary:
            y, class_probs = y
        else:
            y, class_probs = y, None
        return y, class_probs

    def auxiliary_loss(self, class_probs, y):
        return self.auxiliary_loss_function(class_probs, y.long().squeeze())*self.auxiliary_weighting_factor

    def train_generator(self, x_cat, x_cont, y):
        z = self.noise(x_cont)
        self.generator.zero_grad()
        x_cont_fake = self.generator(x_cat, z)
        y_fake = self.discriminator(None, x_cont_fake)
        y_fake, class_probs = self._split_pred(y_fake)
        loss = self.bce_loss(y_fake, torch.ones_like(y_fake))
        if self.auxiliary:
            aux_loss = self.auxiliary_loss(class_probs, y)
            loss = (loss + aux_loss)#2
        loss.backward()
        self.gen_optim.step()
        return

    def train_discriminator(self, x_cat, x_cont, y):
        z = self.noise(x_cont)
        self.discriminator.zero_grad()
        y_real = self.discriminator(None, x_cont)
        y_real, class_probs = self._split_pred(y_real)
        real_loss = self.bce_loss(y_real, torch.ones_like(y_real))
        if self.auxiliary:
            aux_loss = self.auxiliary_loss(class_probs, y)
            self.aux_loss.append(aux_loss)
            real_loss = (real_loss + aux_loss)#/2

        real_loss.backward()
        self.dis_optim.step()
        self.real_loss.append(real_loss.item())

        z = self.noise(x_cont)
        self.discriminator.zero_grad()
        x_cont_fake = self.generator(x_cat, z).detach()
        y_fake = self.discriminator(None, x_cont_fake)
        y_fake, class_probs = self._split_pred(y_fake)

        fake_loss =  self.bce_loss(y_fake, torch.zeros_like(y_fake))
        if self.auxiliary:
            aux_loss = self.auxiliary_loss(class_probs, y)
            fake_loss = (fake_loss + aux_loss)#/2

        fake_loss.backward()
        self.dis_optim.step()
        self.fake_loss.append(fake_loss.item())
        return

    def forward(self, x_cat, x_cont):
        z = self.noise(x_cont)
        x_gen = self.generator(x_cat, z)
        assert(x_gen.shape == x_cont.shape)
        y = self.discriminator(None, x_gen)
        out = self._split_pred(y)
        return out

# Cell

class WGAN(GAN):
    def __init__(self, generator, discriminator, gen_optim, dis_optim, n_z=100, clip=0.01, auxiliary=False):
        super(WGAN, self).__init__(generator, discriminator, gen_optim, dis_optim, n_z, clip, auxiliary)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optim = gen_optim
        self.dis_optim = dis_optim
        self.n_z = n_z
        self.clip = clip
        self.auxiliary = auxiliary
        self.real_loss = []
        self.fake_loss = []

    def train_generator(self, x_cat, x_cont, y):
        z = self.noise(x_cont)
        self.generator.zero_grad()
        x_cont_fake = self.generator(x_cat, z)
        y_fake = self.discriminator(None, x_cont_fake)
        loss = - y_fake.mean()
        loss.backward()
        self.gen_optim.step()
        return

    def train_discriminator(self, x_cat, x_cont, y):
        z = self.noise(x_cont)
        self.discriminator.zero_grad()
        y_real = self.discriminator(None, x_cont)
        real_loss = - y_real.mean()
        real_loss.backward()
        self.dis_optim.step()
        self.real_loss.append(real_loss.item())

        z = self.noise(x_cont)
        self.discriminator.zero_grad()
        x_cont_fake = self.generator(x_cat, z).detach()
        y_fake = self.discriminator(None, x_cont_fake)
        fake_loss = y_fake.mean()
        fake_loss.backward()
        self.dis_optim.step()
        self.fake_loss.append(fake_loss.item())

        for p in self.discriminator.parameters():
            p = torch.clamp(p, -self.clip, self.clip)
        return

# Cell

class AuxiliaryDiscriminator(torch.nn.Module):
    def __init__(self, basic_discriminator, n_classes, final_input_size, len_ts=1, bn=False):
        super(AuxiliaryDiscriminator, self).__init__()
        self.basic_discriminator = basic_discriminator
        self.n_classes = n_classes
        self.final_input_size = final_input_size
        self.len_ts = len_ts

        self.adv_layer = nn.Sequential(nn.Linear(self.final_input_size*len_ts, 1), nn.BatchNorm1d(1) ,nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(self.final_input_size*len_ts, n_classes), nn.BatchNorm1d(n_classes), nn.Softmax(dim=1))

    def forward(self, cats, conts):
        out = self.basic_discriminator(cats, conts)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return (validity, label)

# Cell

def get_gan_model(gan_type, structure, n_classes=2, emb_module=None, bn=False):
    gen_structure = structure.copy()
    structure.reverse()
    dis_structure = structure
    dis_structure[-1] = 1
    n_z = gen_structure[0]

    if gan_type == 'bce' or gan_type == 'aux':
        final_act_dis = nn.Sigmoid
        opt_fct = torch.optim.Adam
        gan_class = GAN
    elif gan_type == 'wgan':
        final_act_dis = None
        opt_fct = torch.optim.RMSprop
        gan_class = WGAN

    generator = GANMLP(ann_structure=gen_structure, act_fct=nn.ReLU, final_act_fct=nn.Sigmoid, transpose=True, embedding_module=emb_module, bn_cont=bn)
    if gan_type == 'aux':
        auxiliary = True
        dis_structure = dis_structure[:-1]
        final_input_size = dis_structure[-1]
        discriminator = GANMLP(ann_structure=dis_structure, act_fct=nn.LeakyReLU, final_act_fct=final_act_dis, bn_cont=bn)
        discriminator = AuxiliaryDiscriminator(basic_discriminator=discriminator, n_classes=n_classes, final_input_size=final_input_size)
    else:
        auxiliary = False
        discriminator = GANMLP(ann_structure=dis_structure, act_fct=nn.LeakyReLU, final_act_fct=final_act_dis, bn_cont=bn)

    gen_opt = opt_fct(params=generator.parameters())
    dis_opt = opt_fct(params=discriminator.parameters())
    model = gan_class(generator=generator, discriminator=discriminator, gen_optim=gen_opt, dis_optim=dis_opt, n_z=n_z, auxiliary=auxiliary)

    return model

# Cell

#class GANCNN(torch.nn.Module):
#    def __init__(self, ann_structure, n_z=100, len_ts=1, bn_cont=False, act_fct=nn.ReLU, final_act_fct=nn.Sigmoid, embedding_module=None, transpose=False):
#        super(GANCNN, self).__init__()
#
#        self.conv_net = TemporalCNN(cnn_structure=ann_structure, batch_norm_cont=bn_cont,
#                                    cnn_type='cnn', act_func=act_fct,
#                                    # TODO: this is not the final layer
#                                    final_activation=act_fct, transpose=transpose)
#        self.transpose = transpose
#        if self.transpose:
#            if final_act_fct is not None:
#                self.model_in = nn.Sequential(nn.Linear(n_z, ann_structure[0]*len_ts), final_act_fct(),
#                                          nn.Unflatten(1, (ann_structure[0], len_ts))
#                                         )
#            else:
#                self.model_in = nn.Sequential(nn.Linear(n_z, ann_structure[0]*len_ts),
#                                          nn.Unflatten(1, (ann_structure[0], len_ts))
#                                         )
#        if not self.transpose:
#            # TODO ugly
#            if final_act_fct is not None:
#                self.model_out = nn.Sequential(nn.Flatten(), nn.Linear(len_ts, len_ts), final_act_fct())
#            else:
#                self.model_out = nn.Sequential(nn.Flatten(), nn.Linear(len_ts, len_ts))
#
#    def forward(self, x_cat, x_cont):
#        if self.transpose:
#            x = self.model_in(x_cont)
#            x = self.conv_net(x_cat, x)
#        else:
#            x = self.conv_net(x_cat, x_cont)
#            x = self.model_out(x)
#        return x