# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_gan.model.ipynb (unless otherwise specified).

__all__ = ['flatten_ts', 'LinBnAct', 'GANMLP', 'AuxiliaryDiscriminator']

# Cell
# export

import numpy as np
import torch
import torch.nn as nn

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
    layers = [nn.Linear(si,so)]
    if use_bn:
        layers += [nn.BatchNorm1d(so)]
    if act_cls is not None:
        layers += [act_cls]

    return nn.Sequential(*layers)

class GANMLP(torch.nn.Module):
    def __init__(self, ann_structure, use_bn=True, bn_cont=False, act_cls=torch.nn.ReLU(), embedding_module=None, final_activation=None):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(GANMLP, self).__init__()

        n_cont = ann_structure[0]
        if embedding_module is not None:
            emb_sz = []
            ann_structure[0] = ann_structure[0] + embedding_module.no_of_embeddings

        self.embedding_module = embedding_module

        layers = []
        for idx in range(1, len(ann_structure)):
            cur_use_bn = use_bn
            cur_act_cls = act_cls
            if idx == 1 and not bn_cont:
                cur_use_bn = False
            if idx == len(ann_structure)-1:
                cur_act_cls=None
                cur_use_bn = False

            layer = LinBnAct(ann_structure[idx-1], ann_structure[idx], cur_use_bn , cur_act_cls )
            layers.append(layer)
        if final_activation is not None:
            layers.append(final_activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, cat, continuous_data):
        if self.embedding_module is not None:
            cat = self.embedding_module(cat)
            continuous_data = torch.cat([cat, continuous_data], 1)

        return self.layers(continuous_data)


class AuxiliaryDiscriminator(torch.nn.Module):
    def __init__(self, basic_discriminator, n_classes, input_size, model_type='mlp'):
        super(AuxiliaryDiscriminator, self).__init__()
        self.basic_discriminator = basic_discriminator
        self.n_classes = n_classes
        self.input_size = input_size
        self.model_type = model_type

        if self.model_type == 'mlp':
            self.adv_layer = nn.Sequential(nn.Linear(self.input_size, 1), nn.Sigmoid())
            self.aux_layer = nn.Sequential(nn.Linear(self.input_size, self.n_classes), nn.Softmax(dim=1))
        elif self.model_type == 'tcn':
            self.adv_layer = nn.Sequential(nn.Linear(self.input_size, 1), nn.Sigmoid())
            self.aux_layer = nn.Sequential(nn.Linear(self.input_size, self.n_classes), nn.Softmax(dim=1))

    def forward(self, cats, conts):
        out = self.basic_discriminator(cats, conts)
        if self.model_type == 'tcn':
            out = out.flatten(1, 2)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return (validity, label)