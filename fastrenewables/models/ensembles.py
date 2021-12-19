# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_ensemble_models.ipynb (unless otherwise specified).

__all__ = ['rank_by_evidence', 'get_posterioirs', 'normalise', 'get_predictive_uncertainty', 'BayesModelAveraing']

# Cell
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch import nn
from ..tabular.model import *
from ..timeseries.model import *
from fastai.tabular.all import *
from torch.autograd import Variable
from sklearn.datasets import make_regression
from fastai.learner import *
from ..utils_pytorch import *
import copy
from ..timeseries.model import *
from ..baselines import BayesLinReg, ELM
from .transfermodels import *
from ..tabular.learner import *
from ..timeseries.learner import *
from sklearn.base import BaseEstimator

# Cell
def rank_by_evidence(cats, conts, targets, models):
    evidences = np.zeros(len(models))
    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            evidences[idx] = model.log_evidence(to_np(conts), to_np(targets), logme=True)
        else:
            evidences[idx] = model.log_evidence(cats, conts, targets, logme=True)
    sort_ids = evidences.argsort()[::-1]
    return evidences, sort_ids

# Cell
def get_posterioirs(cats, conts, targets, models):
    posteriors = []
    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            posterior = np.exp(model.log_posterior(to_np(conts), to_np(targets)))
        else:
            posterior = np.exp(model.log_posterior(cats, conts, targets))
        posteriors.append(posterior.reshape(-1,1))
    posteriors = np.concatenate(posteriors, axis=0)
    return posteriors

# Cell
def normalise(weight_matrix):
#     pytorch: weight_matrix / torch.sum(weight_matrix, dim=1).reshape(weight_matrix.shape[0], 1)
    return weight_matrix / np.sum(weight_matrix, axis=1).reshape(weight_matrix.shape[0], 1)

# Cell
def get_predictive_uncertainty(cats, conts, models):
    yhats,ystds = [], []
    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            yhat, ystd = model.predict_proba(to_np(conts))
            yhats.append(yhat)
            ystds.append(ystd)
        else:
            yhat, ystd = model.predict_proba(cats, conts)
            yhats.append(to_np(yhat))
            ystds.append(to_np(ystd))
#         if yhat.shape[1]>1:
#             raise ValueError("Check for timerseries")


    #  TODO if this working for timersiers?
    yhats, ystds = np.concatenate(yhats, axis=1), np.concatenate(ystds, axis=-1)
    y_uncertainty = 1/ystds

    return yhats, y_uncertainty

# Cell
class BayesModelAveraing(nn.Module):
    def __init__(self, source_models,
                 rank_measure="evidence",
                 weighting_strategy="evidence",
                 n_best_models=-1,
                is_timeseries=False):
        """
            rank_measure [str]: either rmse or logevidence
            n_best_models [int]: the n best models w.r.t. rank_measure are taken into account.
                                default -1 takes all models

        """
        super().__init__()
        self.source_models = np.array(source_models)
        self.rank_measure=rank_measure

        self.weighting_strategy = weighting_strategy
        self.n_best_models = n_best_models
        self.is_timeseries = is_timeseries

        # fake param so that it can be used with pytorch trainers
        self.fake_param=nn.Parameter(torch.zeros((1,1), dtype=torch.float))
        self.fake_param.requires_grad =True

        self.conversion_to_tensor = convert_to_tensor_ts if self.is_timeseries else convert_to_tensor

        self.ensemble_weights = None

    def fit(self, dls):
        cats, conts, targets = self.conversion_to_tensor(dls.train_ds)

        if self.training:
            if self.rank_measure=="evidence":
                self.rank_measure_values, self.sord_ids = rank_by_evidence(cats, conts, targets, self.source_models)
            else:
                raise NotImplemented

            if self.n_best_models != -1:
                self.source_models = self.source_models[self.sord_ids[0:self.n_best_models]]
                self.rank_measure_values = self.rank_measure_values[self.sord_ids[0:self.n_best_models]]

        if self.weighting_strategy=="evidence":
            self.ensemble_weights, _ = rank_by_evidence(cats, conts, targets, self.source_models)
            self.ensemble_weights = normalise(self.ensemble_weights.reshape(-1, len(self.source_models)))
        elif self.weighting_strategy=="posterior":
            self.ensemble_weights = get_posterioirs(cats, conts, targets, self.source_models)
            self.ensemble_weights = normalise(self.ensemble_weights.reshape(-1, len(self.source_models)))


    def forward(self, cats, conts):
        yhat = get_preds(cats, conts, self.source_models)
        yhat = (yhat*self.ensemble_weights).sum(1)
        return yhat

    def predict(self, dls, ds_idx=0):
        ds = dls.train_ds
        if ds_idx==1:
            ds = dls.valid_ds

        self.conversion_to_tensor(ds)

        if self.weighting_strategy=="uncertainty":
            yhat, y_uncertainty = get_predictive_uncertainty(cats, conts, models)
            self.ensemble_weights = normalise(y_uncertainty)
        else:
            yhat = get_preds(cats, conts, self.source_models)

        if self.is_timeseries:
            pass
        else:
            yhat = (yhat*self.ensemble_weights).sum(1)

        return yhat
