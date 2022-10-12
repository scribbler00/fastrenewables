# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_timeseries.model.ipynb (unless otherwise specified).

__all__ = ['Chomp1d', 'BasicTemporalBlock', 'ResidualBlock', 'TemporalConvNet', 'TemporalCNN']

# Cell
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.tabular.data import *
from fastai.tabular.core import *
from .data import *

# Cell
#hide
import numpy as np
import warnings
from collections import OrderedDict, defaultdict
from torch import nn
import torch
from torch.nn.utils import weight_norm
from fastcore.foundation import defaults
from fastai.tabular.model import *
from fastai.layers import *
from ..tabular.model import *
from ..utils_blitz import set_train_mode
from torch.nn import BatchNorm1d
from enum import Enum

from blitz.utils import variational_estimator
import torch.nn.functional as F
from fastcore.foundation import L
from fastai.torch_core import params

# Cell
class Chomp1d(nn.Module):
    """Removes excess padding."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

# Cell
class BasicTemporalBlock(nn.Module):
    """
        Extends fastai `ConvLayer` (CNN+normalization) to include results from an embedding layer,
        as proposed in Task-TCN (https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/main.pdf) for
        MTL-Architectures.
    """
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size=3,
        act_func=nn.ReLU,
        embedding_size=None,
        transpose=False,
    ):
        """[summary]

        Args:
            n_inputs ([type]): [description]
            n_outputs ([type]): [description]
            kernel_size (int, optional): [description]. Defaults to 3.
            act_func ([type], optional): [description]. Defaults to nn.ReLU.
            embedding_size ([type], optional): [description]. Defaults to None.
            transpose (bool, optional): [description]. Defaults to False.
        """
        super(BasicTemporalBlock, self).__init__()
        self.embedding_size = embedding_size
        self.act_func = act_func

        self.conv = ConvLayer(
            n_inputs,
            n_outputs,
            ks=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            norm_type=NormType.Weight,
            bias=False,
            ndim=1,
            act_cls=self.act_func,
            transpose=transpose,
        )

        if self.embedding_size is not None:
            self.embedding_transform = (
                nn.Conv1d(self.embedding_size, n_outputs, 1)
                if (self.embedding_size != n_outputs)
                and (self.embedding_size is not None)
                else None
            )
            self.emb_act_func = self.act_func()
        else:
            self.embedding_transform = None

        self.init_weights()

    def init_weights(self):
        # ConvLayer uses fastai init strategy
        # use fastais init strategy
        if self.embedding_transform is not None:
            init_linear(self.embedding_transform, act_func=self.act_func, init="auto")

    def forward(self, categorical, continous=None):
        res = self.conv(continous)

        res = (
            res
            if self.embedding_transform is None
            else self.emb_act_func(self.embedding_transform(categorical) + res)
        )

        return res

# Cell
class ResidualBlock(nn.Module):
    # todo: doesn't support transposedconv-layer
    """
        (Single) Residual block of a TCN.
    """
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        act_func=nn.ReLU,
        embedding_size=None,
        dropout=0.2,
        transpose=False #todo if needed
    ):
        super(ResidualBlock, self).__init__()
        self.embedding_size = embedding_size
        self.act_func = Identity if act_func is None else act_func

        if transpose:
            conv_fct = nn.ConvTranspose1d
        else:
            conv_fct = nn.Conv1d

        self.conv1 = weight_norm(
            conv_fct(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.act_func1 = self.act_func()
        # equivalent to keras spatial dropout if input is of form (batch_size, channels, time_series_length)
        self.dropout1 = nn.Dropout2d(dropout)

        self.conv2 = weight_norm(
            conv_fct(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.act_func2 = self.act_func()
        # equivalent to keras spatial dropout if input is of form (batch_size, channels, time_series_length)
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.act_func1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.act_func2,
            self.dropout2,
        )

        self.downsample = (
            conv_fct(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

        self.act_func3 = self.act_func()

        if self.embedding_size is not None:
            self.embedding_transform = (
                conv_fct(self.embedding_size, n_outputs, 1)
                if (self.embedding_size != n_outputs)
                and (self.embedding_size is not None)
                else None
            )

        else:
            self.embedding_transform = None

        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the convolution layers
        Returns
        -------

        """
        # use fastais init strategy
        init_linear(self.conv1, act_func=self.act_func, init="auto")
        init_linear(self.conv2, act_func=self.act_func, init="auto")
        if self.downsample is not None:
            init_linear(self.downsample, act_func=self.act_func, init="auto")
        if self.embedding_transform is not None:
            init_linear(self.embedding_transform, act_func=self.act_func, init="auto")

    def forward(self, categorical, continous=None):
        out = self.net(continous)
        res = continous if self.downsample is None else self.downsample(continous)
        res = out + res

        res = (
            res
            if self.embedding_transform is None
            else self.embedding_transform(categorical) + res
        )

        return self.act_func3(res)

# Cell
class TemporalConvNet(nn.Module):
    """Wrapper module that is capable of creating a simple CNN or a TCN."""

    def __init__(
        self,
        num_inputs,
        num_channels,
        kernel_size=3,
        dropout=0.0,
        cnn_type="tcn",
        embedding_size=None,
        final_activation=Identity,
        act_func=nn.ReLU,
        add_embedding_at_layer=L(),
        transpose=False,
    ):
        """[summary]

        Args:
            num_inputs ([type]): [description]
            num_channels ([type]): [description]
            kernel_size (int, optional): [description]. Defaults to 3.
            dropout (float, optional): [description]. Defaults to 0.0.
            cnn_type (str, optional): [description]. Defaults to "tcn".
            embedding_size ([type], optional): [description]. Defaults to None.
            final_activation ([type], optional): [description]. Defaults to Identity.
            act_func ([type], optional): [description]. Defaults to nn.ReLU.
            add_embedding_at_layer ([type], optional): [description]. Defaults to L().
            transpose (bool, optional): [description]. Defaults to False.

        Raises:
            ValueError: [description]
        """
        super(TemporalConvNet, self).__init__()

        self.embedding_size = embedding_size
        self.cnn_type = cnn_type
        layers = []

        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            if self.cnn_type == "tcn":
                dilation_size = 2 ** i
                cur_layer = ResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    embedding_size=self.embedding_size
                    if i in add_embedding_at_layer
                    else None,
                    act_func=act_func if i < num_levels - 1 else final_activation,
                    transpose=transpose
                )
            elif self.cnn_type == "cnn":
                cur_layer = BasicTemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    embedding_size=self.embedding_size
                    if i in add_embedding_at_layer
                    else None,
                    act_func=act_func if i < num_levels - 1 else final_activation,
                    transpose=transpose,
                )
            else:
                raise ValueError("Expected cnn/tcn as input for cnn_type.")

            layers += [cur_layer]

        self.temporal_blocks = nn.Sequential(*layers)

    def forward(self, categorical, continous=None):
        """The categorical data can either be encoded via an embedding layer or directly applied."""
        x = continous
        for layer in self.temporal_blocks:
            x = layer(categorical, x)

        return x

# Cell
@variational_estimator
class TemporalCNN(nn.Module):
    """Module to create a CNN based architecture for timerseries such as a simple CNN, a TCN, or a Task-TCN."""
    def __init__(
        self,
        cnn_structure,
        kernel_size=3,
        dropout=0.0,
        embedding_module=None,
        batch_norm_cont=True,
        cnn_type="tcn",
        y_ranges=None,
        final_activation=Identity,
        act_func=nn.ReLU,
        add_embedding_at_layer=[],
        input_sequence_length=None,
        output_sequence_length=None,
        transpose=False,
        sequence_transform = None
    ):
        """[summary]

        Args:
            cnn_structure ([type]): [description]
            kernel_size (int, optional): [description]. Defaults to 3.
            dropout (float, optional): [description]. Defaults to 0.0.
            embedding_module ([type], optional): [description]. Defaults to None.
            batch_norm_cont (bool, optional): [description]. Defaults to True.
            cnn_type (str, optional): [description]. Defaults to "tcn".
            y_ranges ([type], optional): [description]. Defaults to None.
            final_activation ([type], optional): [description]. Defaults to Identity.
            act_func ([type], optional): [description]. Defaults to nn.ReLU.
            add_embedding_at_layer (list, optional): [description]. Defaults to [].
            input_sequence_length ([type], optional): [description]. Defaults to None.
            output_sequence_length ([type], optional): [description]. Defaults to None.
            transpose (bool, optional): [description]. Defaults to False.
        """
        super(TemporalCNN, self).__init__()

        self.embedding_module = embedding_module
        self.cnn_structure = cnn_structure
        self.cnn_type = cnn_type.lower()
        self.batch_norm_cont = batch_norm_cont
        self.y_ranges = y_ranges
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.transpose = transpose

        self.bn_cont = (
            BatchNorm1d(self.cnn_structure[0]) if self.batch_norm_cont else None
        )
        self.embedding_size = (
            self.embedding_module.no_of_embeddings
            if self.embedding_module is not None
            else None
        )

        num_channels = self.cnn_structure[1:]
        num_inputs = self.cnn_structure[0]

        self.layers = TemporalConvNet(
            num_inputs,
            num_channels=num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            cnn_type=self.cnn_type,
            embedding_size=self.embedding_size,
            final_activation=final_activation,
            act_func=act_func,
            add_embedding_at_layer=add_embedding_at_layer,
            transpose=self.transpose,
        )

        self.custom_output_sequence=False
        if output_sequence_length != None and input_sequence_length != None and sequence_transform is None:
            # TODO: can this replace with a 1D-CNN with kernel size 1?
            self.sequence_transform = nn.Linear(
                self.cnn_structure[-1] * self.input_sequence_length,
                self.cnn_structure[-1] * self.output_sequence_length,
            )
        elif sequence_transform is not None:
            self.sequence_transform = sequence_transform
            self.custom_output_sequence=True
        else:
            self.sequence_transform = None
            self.custom_output_sequence=False



    def forward(
        self,
        categorical_data,
        continuous_data,
    ):
        """

        Parameters
        ----------
        categorical_data : pytorch.Tensor
            categorical input data. only used when an embedding module is available.
        continuous_data : pytorch.Tensor
            continuous input data.

        Returns
        -------
        pytorch.Tensor
            concatenated outputs of all separate subnetworks.
        """
        if self.batch_norm_cont:
            # expecting (batch_size, n_features, timeseries_length)
            continuous_data = self.bn_cont(continuous_data)

        if self.embedding_module is not None:
            categorical_data = self._forward_embedding_module(
                categorical_data,
            )

        x = self.layers(categorical_data, continuous_data)

        if self.custom_output_sequence:
            x = x.reshape(-1, self.cnn_structure[-1] * self.input_sequence_length)
            x = self.sequence_transform(categorical_data, x)
            x = x.unsqueeze(1)

        elif self.sequence_transform is not None:
            x = x.reshape(-1, self.cnn_structure[-1] * self.input_sequence_length)
            x = self.sequence_transform(x)
            x = x.unsqueeze(1)

        if self.y_ranges is not None:
            y_range = self.y_ranges[0]
            x = (y_range[1] - y_range[0]) * torch.sigmoid(x) + y_range[0]

        return x

    def _forward_embedding_module(self, categorical_data):
        """
        Apply the embedding layer and return result.
        ----------
        categorical_data : pytorch.Tensor
            categorical input data. only used when an embedding module is available.
        continuous_data : pytorch.Tensor
            continuous input data.

        Returns
        -------
        pytorch.Tensor
            combined tensors of continuous data and the output of the embedding layer.
        """
        # check if all columns have the same value
        if (
            categorical_data[:, :, 0].reshape((-1, categorical_data.shape[1], 1))
            - categorical_data
        ).sum() == 0:
            return self._forward_embedding_module_same(categorical_data)
        else:
            if self.embedding_module.embedding_type == EmbeddingType.Bayes:
                warnings.warn(
                    "Mixed types not supported for bayesian embedding. Fallback to sampling per time step."
                )

            return self._forward_embedding_module_different(categorical_data)

    def _forward_embedding_module_same(self, categorical_data):
        """
        If all columns have the same value, apply the embedding method only to the first column
        Parameters
        ----------
        categorical_data : pytorch.Tensor
            categorical input data.

        Returns
        -------
        pytorch.Tensor
            resulting tensor of the embedding module.
        """
        timesteps = categorical_data.shape[2]
        batch_size = categorical_data.shape[0]

        # Assume that all columns have the same value
        # In case of an bayes embedding all should have the same value
        categorical_data = categorical_data[:, :, 0]
        categorical_data = self.embedding_module(categorical_data)
        emb_dim = categorical_data.shape[1]
        ones = torch.ones((batch_size, emb_dim, timesteps))

        device_type = self.embedding_module.embeddings[0].weight.device.type
        if torch.cuda.is_available() and device_type != "cpu":
            ones = ones.cuda()

        return ones * categorical_data.reshape(batch_size, emb_dim, 1)

    def _forward_embedding_module_different(self, categorical_data):
        """
        If all columns do not have the same value, apply the embedding method to the whole input tensor
        Parameters
        ----------
        categorical_data : pytorch.Tensor
            categorical input data.

        Returns
        -------
        pytorch.Tensor
            resulting tensor of the embedding module.
        """
        features = categorical_data.shape[1]
        timesteps = categorical_data.shape[2]

        categorical_data = self.embedding_module(
            categorical_data.permute(0, 2, 1).reshape(-1, features)
        )
        categorical_data = categorical_data.reshape(
            -1, timesteps, categorical_data.shape[1]
        ).permute(0, 2, 1)

        return categorical_data

    def train(self, mode: bool = True):
        super().train(mode)
        set_train_mode(self, mode)

    def network_split(self):
        "Default split of the between body and head"

        if self.embedding_module is not None:
            splitter = lambda m: L(
                m.layers.temporal_blocks[0],
                m.embedding_module,
                m.layers.temporal_blocks[1:-1],
                m.layers.temporal_blocks[-1:],
            ).map(params)

            lr = L(1e-6, 1e-6, 1e-6, 1e-4)
        else:
            splitter = lambda m: L(
                m.layers.temporal_blocks[0],
                m.layers.temporal_blocks[1:-1],
                m.layers.temporal_blocks[-1:],
            ).map(params)

            lr = L(1e-6, 1e-6, 1e-4)

        return splitter, lr