# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_tabular.model.ipynb (unless otherwise specified).

__all__ = ['EmbeddingType', 'get_emb_sz_list', 'EmbeddingModule', 'MultiLayerPerceptron', 'get_structure']

# Cell
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.tabular.data import *
from fastai.tabular.core import *
from .data import *

# Cell

from enum import Enum

import torch
import torch.nn as nn

from fastai.tabular.model import emb_sz_rule
from fastai.basics import *
from fastai.layers import *


from blitz.modules.embedding_bayesian_layer import BayesianEmbedding
from blitz.utils import variational_estimator


class EmbeddingType(Enum):
    Normal = 0
    Bayes = 1


def get_emb_sz_list(dims: list):
    """
    For all elements in the given list, find a size for the respective embedding through trial and error
    Each element denotes the amount of unique values for one categorical feature
    Parameters
    ----------
    dims : list
        a list containing a number of integers.
    Returns
    -------
    list of tupels
        a list containing an the amount of unique values and respective embedding size for all elements.
    """
    return [(d, emb_sz_rule(d)) for d in dims]


@variational_estimator
class EmbeddingModule(nn.Module):
    def __init__(
        self,
        categorical_dimensions,
        embedding_dropout=0.0,
        embedding_dimensions=None,
        embedding_type=EmbeddingType.Normal,
        names=None,
        **kwargs
    ):
        super().__init__()
        """
        Parameters
        ----------
        categorical_dimensions : list of integers
            List with number of categorical values for each feature.
            Output size is calculated based on
            fastais `emb_sz_rule`. In case explicit dimensions
            are required use `embedding_dimensions`.

        embedding_dropout : Float
            the dropout to be used after the embedding layers.

        embedding_dimensions : list of tupels
            This list will contain a two element tuple for each
            categorical feature. The first element of a tuple will
            denote the number of unique values of the categorical
            feature. The second element will denote the embedding
            dimension to be used for that feature. If None, `categorical_dimensions`
            is used to determine the dimensions.
        embedding_type : EmbeddingType
            the type of embedding that is to be used.
        """

        self.embedding_type = embedding_type

        # Set the embedding dimension for all features
        if embedding_dimensions is None:
            self.embedding_dimensions = get_emb_sz_list(categorical_dimensions)
        else:
            self.embedding_dimensions = embedding_dimensions

        self.embeddings = nn.ModuleList(
            [self._embedding(ni, nf, **kwargs) for ni, nf in self.embedding_dimensions]
        )
        self.emb_drop = nn.Dropout(embedding_dropout) if embedding_dropout > 0 else None
        self._set_names(names)

    def _set_names(self, names):
        if names is not None and len(names) == len(self.embeddings):
            self.embeddings_by_names = {n:self.embeddings[i] for i,n in enumerate(names)}
        else:
            self.embeddings_by_names = None

    @property
    def _embedding(self):
        if self.embedding_type == EmbeddingType.Normal:
            emb_type = Embedding
        elif self.embedding_type == EmbeddingType.Bayes:
            emb_type = BayesianEmbedding
        else:
            raise ValueError(f"Unknown embedding type {self.embedding_type}.")

        return emb_type

    @property
    def no_of_embeddings(self):
        return sum(e.embedding_dim for e in self.embeddings)


    @property
    def categorical_dimensions(self):
        return L(e.num_embeddings for e in self.embeddings)

    def forward(self, categorical_data):
        """
        Parameters
        ----------
        categorical_data : pytorch.Tensor
            categorical input data.

        Returns
        -------
        pytorch.Tensor
            concatenated outputs of the network for all categorical features.
        """
        x = torch.cat(
            [
                emb_layer(categorical_data[:, i])
                for i, emb_layer in enumerate(self.embeddings)
            ],
            1,
        )

        x = self.emb_drop(x) if self.emb_drop is not None else x

        return x

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters of the embeddings.
        Parameters
        ----------
        device (:class:`torch.device`) : the desired device of the parameters
                and buffers in this module
        dtype (:class:`torch.dtype`) : the desired floating point type of
            the floating point parameters and buffers in this module
        tensor (torch.Tensor) : Tensor whose dtype and device are the desired
            dtype and device for all parameters and buffers in this module
        memory_format (:class:`torch.memory_format`) : the desired memory
            format for 4D parameters and buffers in this module (keyword
            only argument)

        Returns
        -------
        self
        """
        self = super().to(*args, **kwargs)
        for idx, emb in enumerate(self.embeddings):
            self.embeddings[idx] = emb.to(*args, **kwargs)

        return self

    @typedispatch
    def __getitem__(self, idx: int) -> Module:
        return self.embeddings[idx]
    @typedispatch
    def __getitem__(self, key: str) -> Module:
        return self.embeddings_by_names[key]

    def extra_repr(self):
        s = ""
        if self.embeddings_by_names is not None:
            s += f"Embedding Names: {list(self.embeddings_by_names.keys())}\n"

        return s

    def reset_cat_embedding(self, emb_id: int, cat_ids: list):
        cat_ids = listify(cat_ids)
        with torch.no_grad():
            emb = self.embeddings[emb_id]
            reset_data = self._embedding(emb.num_embeddings, emb.embedding_dim)
            if self.embedding_type == EmbeddingType.Normal:
                emb.weight[cat_ids, :] = reset_data.weight[cat_ids, :]
            else:
                emb.weight_sampler.mu[cat_ids, :] = reset_data.weight_sampler.mu[
                    cat_ids, :
                ]
                emb.weight_sampler.rho[cat_ids, :] = reset_data.weight_sampler.rho[
                    cat_ids, :
                ]

    def copy_cat_embedding(self, emb_id: int, from_cat_ids: list, to_cat_ids: list):
        from_cat_ids, to_cat_ids = listify(from_cat_ids), listify(to_cat_ids)
        with torch.no_grad():
            emb = self.embeddings[emb_id]
            for idx in range(len(from_cat_ids)):
                if isinstance(emb, Embedding):
                    emb.weight[to_cat_ids[idx], :] = emb.weight[from_cat_ids[idx], :]
                elif isinstance(emb, BayesianEmbedding):
                    emb.weight_sampler.mu[to_cat_ids[idx], :] = emb.weight_sampler.mu[
                        from_cat_ids[idx], :
                    ]
                    emb.weight_sampler.rho[to_cat_ids[idx], :] = emb.weight_sampler.rho[
                        from_cat_ids[idx], :
                    ]
                else:
                    raise ValueError("Unexpected embedding type.")

    def increase_embedding_by_one(self, emb_id: int, device="cpu"):
        with torch.no_grad():
            emb = self.embeddings[emb_id]

            if isinstance(emb, Embedding):
                emb_new = Embedding(emb.num_embeddings + 1, emb.embedding_dim).to(device)
                elements_to_copy = list(range(emb.weight.shape[0]))
                emb_new.weight[elements_to_copy, :] = emb.weight[elements_to_copy, :]
            elif isinstance(emb, BayesianEmbedding):
                emb_new = BayesianEmbedding(emb.num_embeddings + 1, emb.embedding_dim).to(
                    device
                )
                elements_to_copy = list(range(emb.weight_sampler.mu.shape[0]))
                emb_new.weight_sampler.mu[elements_to_copy, :] = emb.weight_sampler.mu[
                    elements_to_copy, :
                ]
                emb_new.weight_sampler.rho[elements_to_copy, :] = emb.weight_sampler.rho[
                    elements_to_copy, :
                ]
            else:
                raise ValueError("Unexpected embedding type.")

            self.embeddings[emb_id] = emb_new


@patch
def extra_repr(self:BayesianEmbedding): return f"Shape: {list(self.weight_sampler.mu.shape)}"


# Cell

import copy

import torch
import torch.nn as nn

from torch.nn import ReLU

from fastai.tabular.model import *
from fastai.vision.all import *

from blitz.utils import variational_estimator

# from dies.utils_pytorch import freeze, unfreeze
# from dies.abstracts import Transfer
# from dies.utils_blitz import set_train_mode


@variational_estimator
class MultiLayerPerceptron(TabularModel):
#     class MultiLayerPerceptron(TabularModel, Transfer):
    @use_kwargs_dict(
        ps=None,
        embed_p=0.0,
        y_range=None,
        use_bn=True,
        bn_final=False,
        act_cls=ReLU(inplace=True),
        embedding_module=None,
    )
    def __init__(
        self,
        ann_structure,
        emb_sz=[],
        embedding_module=None,
        final_activation=Identity,
        bn_cont=True,
        **kwargs
    ):
        """

        Parameters
        ----------
        ann_structure : list of integers
            amount of features for each layer.
        emb_sz : list
            currently not used.
        embedding_module : dies.embedding
            if not 'None', use the given embedding module and adjust the network accordingly.
        final_activation : class
            activation function for the last layer.
        bn_cont : bool
            decide whether a batch norm is to be used for the continuous data.
        kwargs :

        """
        n_cont = ann_structure[0]
        if embedding_module is not None:
            emb_sz = []
            ann_structure[0] = ann_structure[0] + embedding_module.no_of_embeddings

        self.embedding_module = embedding_module
        self.final_activation = final_activation()

        super(MultiLayerPerceptron, self).__init__(
            emb_sz,
            ann_structure[0],
            ann_structure[-1],
            ann_structure[1:-1],
            bn_cont=bn_cont,
            **kwargs
        )

        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None

    def forward(self, categorical_data, continuous_data):
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
            concatenated outputs of the network for continuous and categorical data.
        """
        if self.embedding_module is None:
            x = super().forward(categorical_data, continuous_data)
        else:
            categorical_data = self.embedding_module(categorical_data)
            if self.bn_cont is not None:
                continuous_data = self.bn_cont(continuous_data)

            x = torch.cat([categorical_data, continuous_data], 1)
            x = self.layers(x)

        return self.final_activation(x)

    def network_split(self):
        "Default split of the body and head"

        if self.embedding_module is not None:
            splitter = lambda m: L(
                m.layers[0],
                m.embedding_module,
                m.layers[1:-1],
                m.layers[-1:],
            ).map(params)

            lr = L(1e-6, 1e-6, 1e-6, 1e-4)
        else:
            splitter = lambda m: L(
                m.layers[0],
                m.layers[1:-1],
                m.layers[-1:],
            ).map(params)

            lr = L(1e-6, 1e-6, 1e-4)

        return splitter, lr
# TODO:
#     def train(self, mode: bool = True):
#         super().train(mode)
#         set_train_mode(self, mode)


# Cell
def get_structure(
    initial_size,
    percental_reduce,
    min_value,
    input_size=None,
    final_outputs=1,
    reverse_structure=False,
):
    """
    Turn the given parameters into the structure of an ann model.

    The 'initial size' acts as the first layer, and each following layer i is of the size
    'initial_size' * (1 - percental_reduce) ^ i. This is repeated until 'min_value' is reached. Finally, 'final_outputs'
    is appended as the last layer.

    Parameters
    ----------
    initial_size : integer
        size of the first layer, and baseline for all following layers.
    percental_reduce : float
        percentage of the size reduction of each subsequent layer.
    min_value : integer
        the minimum layer size up to which the 'initial_size' is used to create new layers.
    input_size : integer
        if not None, a layer of the given size will be prepended to the actual structure.
    final_outputs : integer
        the size of the final layer.

    Returns
    -------
    list
        The finished structure of the ann model.
    """
    ann_structure = [initial_size]
    final_outputs = listify(final_outputs)

    if 0 in final_outputs or (None in final_outputs):
        raise ValueError(
            "Invalid parameters: final_outputs should not contain 0 or None"
        )

    if percental_reduce >= 1.0:
        percental_reduce = percental_reduce / 100.0

    while True:
        new_size = int(ann_structure[-1] - ann_structure[-1] * percental_reduce)

        if new_size <= min_value:
            new_size = min_value
            ann_structure.append(new_size)
            break
        else:
            ann_structure.append(new_size)

    if reverse_structure:
        ann_structure = list(reversed(ann_structure))

    if input_size != None:
        input_size = listify(input_size)
        return input_size + ann_structure + final_outputs

    else:
        return ann_structure + final_outputs