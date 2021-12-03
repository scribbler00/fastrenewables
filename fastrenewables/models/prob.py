# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/13_probabilistic_models.ipynb (unless otherwise specified).

__all__ = ['MeanStdWrapper']

# Cell
import numpy as np
import torch
from torch import nn
from ..tabular.model import *
from fastai.tabular.data import *
from ..timeseries.model import *
from fastai.tabular.all import *
from torch.autograd import Variable
import pandas as pd
from ..losses import *
from fastai.losses import MSELossFlat
from blitz.utils import variational_estimator
from ..utils_blitz import set_train_mode
from ..metrics import *

# Cell
@variational_estimator
class MeanStdWrapper(nn.Module):
    def __init__(self, model, last_layer_size=None, nll_output_layer=None):
        super().__init__()
        self.model = model
        if nll_output_layer is None and last_layer_size is not None:
            self.nll_output_layer = nn.Linear(last_layer_size, 2)
        elif nll_output_layer is None and last_layer_size is None:
            self.nll_output_layer = nll_output_layer
        else:
            raise ValueError("Either provide and output layer or the lasy layer size.")

    def forward(self, categorical_data, continuous_data):
        x = self.model(categorical_data, continuous_data)
        x = self.nll_output_layer(x)

        return x


    def train(self, mode: bool = True):
        super().train(mode)
        set_train_mode(self, mode)