# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/16_weak_supvervision_models.ipynb (unless otherwise specified).

__all__ = ['CrowdLayer']

# Cell
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch import nn
from ..tabular.core import *
from ..tabular.model import *
from ..tabular.learner import *
from ..timeseries.model import *
from fastai.tabular.all import *
from torch.autograd import Variable
from fastai.learner import *
from ..utils_pytorch import *
from ..utils import filter_preds

import copy

from ..baselines import BayesLinReg, MCLeanPowerCurve
from ..tabular.learner import convert_to_tensor
from ..losses import *



from ..timeseries.core import *
from ..timeseries.model import *
from ..timeseries.learner import *

# Cell
class CrowdLayer(nn.Module):
    def __init__(
        self,
        source_model,
        num_weak_labels=1,
        num_timesteps=1,
        layer_type="bias"
    ):
        super().__init__()

        self.num_weak_labels=num_weak_labels
        self.num_timesteps=num_timesteps

        # the "1" broadcast through the samples
        # we expect for both cases a univariate output
        if self.num_timesteps == 1:
            self.bias = nn.Parameter(torch.zeros(1, num_weak_labels))
        else:
            self.bias = nn.Parameter(torch.zeros(1, num_weak_labels, self.num_timesteps))
        self.bias.requires_grad=True
        self.source_model=source_model
        if layer_type != "bias":
            raise NotImplementedError
        self.layer_type = layer_type
    def forward(self, cats, conts):
        x = conts
        if self.source_model is not None:
            x = self.source_model(cats, conts)

        if x.shape[1] != 1:
            raise ValueError("Only univariate outputs are supported.")

        return x + self.bias