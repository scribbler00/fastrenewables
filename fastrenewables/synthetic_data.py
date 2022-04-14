# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00g_synthetic_data.ipynb (unless otherwise specified).

__all__ = ['DummyDataset', 'SineDataset']

# Cell
# export

import torch
import matplotlib.pyplot as plt

# Cell

class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, n_samples=100, n_cat_feats=2, n_cont_feats=2, n_targets=1, len_ts=24, n_dim=2):

        self.n_samples = n_samples
        if n_dim == 2:
            self.cat = torch.ones(n_samples, n_cat_feats)
            self.cont = torch.ones(n_samples, n_cont_feats)
        elif n_dim == 3:
            self.cat = torch.ones(n_samples, n_cat_feats, len_ts)
            self.cont = torch.ones(n_samples, n_cont_feats, len_ts)
            #self.y = torch.ones(n_samples, n_targets, len_ts)
        elif n_dim == 4:
            self.cat = torch.ones(n_samples, n_cat_feats, len_ts, len_ts)
            self.cont = torch.ones(n_samples, n_cont_feats, len_ts, len_ts)
            #self.y = torch.ones(n_samples, n_targets, len_ts, len_ts)
        self.y = torch.ones(n_samples, n_targets)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_cat = self.cat[idx, :]
        x_cont = self.cont[idx, :]
        y = self.y[idx, :]
        return x_cat, x_cont, y

# Cell

class SineDataset(torch.utils.data.Dataset):

    def __init__(self, n_samples=100, n_classes=2, n_features=2, len_ts=24, noise=0.05):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_features = n_features
        self.len_ts = len_ts
        self.amp_min = 0.2
        self.amp_max = 0.8
        self.freq_min = 0.0
        self.freq_max = 3.0
        self.phase_min = 0.0
        self.phase_max = 2.0*torch.pi
        self.noise = noise
        self.cat_data = torch.zeros(self.n_samples*self.n_classes, self.n_features, self.len_ts)
        self.cont_data = torch.zeros(self.n_samples*self.n_classes, self.n_features, self.len_ts)
        self.target = torch.zeros(self.n_samples*self.n_classes, 1)

        self.generate_series()
        self.cat_data = (self.cat_data-self.cat_data.min())/(self.cat_data.max()-self.cat_data.min())
        self.cont_data = (self.cont_data-self.cont_data.min())/(self.cont_data.max()-self.cont_data.min())

    def generate_series(self):
        for c in range(self.n_classes):
            t = torch.linspace(0, 1, self.len_ts)
            amplitudes = torch.distributions.Uniform(self.amp_min, self.amp_max).sample((1, self.n_features))
            frequencies = torch.distributions.Uniform(self.freq_min, self.freq_max).sample((1, self.n_features))
            phases = torch.distributions.Uniform(self.phase_min, self.phase_max).sample((1, self.n_features))
            for idx in range(self.n_samples):
                self.target[c*self.n_samples + idx, :] = c
                for f_idx in range(self.n_features):
                    a = amplitudes[:, f_idx]
                    f = frequencies[:, f_idx]
                    p = phases[:, f_idx]
                    self.cat_data[c*self.n_samples + idx, f_idx, :] = p + self.noise*torch.randn_like(t)
                    self.cont_data[c*self.n_samples + idx, f_idx, :] = a*torch.sin(2*torch.pi*f*t + p) + self.noise*torch.randn_like(t)

    def __len__(self):
        return self.n_samples*self.n_classes

    def __getitem__(self, idx):
        x_cat = self.cat_data[idx, :]
        x_cont = self.cont_data[idx, :]
        y = self.target[idx, :]
        return x_cat, x_cont, y