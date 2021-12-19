# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_ensemble_models.ipynb (unless otherwise specified).

__all__ = ['normalize_weight', 'weight_preds', 'update_single_model', 'rank_by_evidence', 'get_posterioirs',
           'get_predictive_uncertainty', 'get_preds', 'BayesModelAveraing', 'squared_error', 'soft_gating',
           'create_error_matrix', 'get_global_weight', 'get_timedependent_weight', 'simple_local_error_estimator',
           'get_local_weight', 'CSGE']

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
import numpy as np

# Cell
def normalize_weight(ensemble_weight):
    """
    Let N be the number of samples, k the number of ensembles, and
    t be the forecast horizon.
    In case of an input array of dimension k, it is reshaped to k x 1, assuming a single forecast horizon.
    In case of a k x t array we normalize for each forecast horizon t acrross all ensmeble members.
    In case of Nxkxt we also normalise accross all members for each horizon.
    """

    if len(ensemble_weight.shape) == 1:
        ensemble_weight = ensemble_weight.reshape(len(ensemble_weight), 1)

    if len(ensemble_weight.shape) == 2:
        ensemble_weight = ensemble_weight/ensemble_weight.sum(0)
    elif len(ensemble_weight.shape) == 3:
        ensemble_weight = (ensemble_weight/ensemble_weight.sum(1)[:,np.newaxis,:])

    return ensemble_weight

# Cell
def weight_preds(preds, ensemble_weight):
    """
        Weight the predictions by the ensemble weights, depeneding on the shape.
    """
    ensemble_weight = ensemble_weight
    if len(ensemble_weight.shape) == 1:
        ensemble_weight = ensemble_weight.reshape(len(ensemble_weight), 1)
    if len(preds.shape) == 1:
        preds = preds.reshape(len(preds), 1)
    if len(preds.shape) == 2:
        preds = preds[:,:,np.newaxis]

    if len(ensemble_weight.shape) == 2:
        preds = ((preds*ensemble_weight).sum(1))
    elif len(ensemble_weight.shape) == 3:
        preds = (preds*ensemble_weight).sum(1)
    return preds

# Cell
def update_single_model(model, dls):
    target_model = LinearTransferModel(
                    model, num_layers_to_remove=1,
                    prediction_model=BayesLinReg(1, 1, use_fixed_point=True))

    target_learner = RenewableLearner(dls, target_model, loss_func=target_model.loss_func, metrics=rmse,)
    target_learner.dls[0].bs=len(target_learner.dls.train_ds)
    target_learner.fit(1)

    return target_learner.model

# Cell
def rank_by_evidence(cats, conts, targets, models):
    evidences = np.zeros(len(models))
    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            evidences[idx] = model.log_evidence(to_np(conts), to_np(targets), logme=True)
        else:
            evidences[idx] = model.log_evidence(cats, conts, targets, logme=True)
    sort_ids = evidences.argsort()[::-1]
    return evidences.reshape(len(models), 1), sort_ids

# Cell
def get_posterioirs(cats, conts, targets, models):
    posteriors = []
    ts_length = 1
    if len(conts.shape)==3:
        ts_length = conts.shape[2]

    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            posterior = np.exp(model.log_posterior(to_np(conts), to_np(targets)))
        else:
            posterior = np.exp(model.log_posterior(cats, conts, targets))
        posteriors.append(posterior.reshape(-1,1))
    posteriors = np.concatenate(posteriors, axis=1)
    return posteriors.reshape(len(models),ts_length)

# Cell
def get_predictive_uncertainty(cats, conts, models):
    yhats,ystds = [], []
    ts_length = 1
    if len(conts.shape)==3:
        ts_length = conts.shape[2]

    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            yhat, ystd = model.predict_proba(to_np(conts))
            yhats.append(yhat.reshape(-1,ts_length))
            ystds.append(ystd.reshape(-1,ts_length))
        else:
            yhat, ystd = model.predict_proba(cats, conts)
            yhats.append(to_np(yhat).reshape(-1,ts_length))
            ystds.append(to_np(ystd).reshape(-1,ts_length))

    #  TODO if this working for timersiers?
    yhats, ystds = np.concatenate(yhats, axis=1), np.concatenate(ystds, axis=1)
    yhats, ystds = yhats.reshape(len(conts),len(models),ts_length), ystds.reshape(len(conts),len(models),ts_length)
    y_uncertainty = 1/ystds

    return yhats, y_uncertainty

# Cell
def get_preds(cats, conts, models):
    yhats,ystds = [], []
    ts_length = 1
    if len(conts.shape)==3:
        ts_length = conts.shape[2]
    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            yhat = model.predict(to_np(conts)).reshape(-1,ts_length)
            yhats.append(yhat)
        else:
            yhat = model.predict(cats, conts).reshape(-1,ts_length)
            yhats.append(to_np(yhat))

    yhats = np.concatenate(yhats, axis=1).reshape(len(conts),len(models),ts_length)

    return yhats

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

        elif self.weighting_strategy=="posterior":
            self.ensemble_weights = get_posterioirs(cats, conts, targets, self.source_models)

        if self.weighting_strategy != "uncertainty":
            self.ensemble_weights = normalize_weight(self.ensemble_weights)


    def _predict(self, cats, conts):
        if self.weighting_strategy=="uncertainty":
            yhat, y_uncertainty = get_predictive_uncertainty(cats, conts, self.source_models)
            if len(y_uncertainty.shape) == 2:
                y_uncertainty = y_uncertainty[:,:, np.newaxis]

            self.ensemble_weights = normalize_weight(y_uncertainty)
        else:
            yhat = get_preds(cats, conts, self.source_models)


        yhat = weight_preds(yhat, self.ensemble_weights)

        return yhat

    def forward(self, cats, conts):
        yhat = self._predict(cats, conts)

        return yhat

    def predict(self, dls, ds_idx=0):
        ds = dls.train_ds
        if ds_idx==1:
            ds = dls.valid_ds

        cats, conts, _ = self.conversion_to_tensor(ds)

        yhat = self._predict(cats, conts)

        return yhat


# Cell
def squared_error(y,yhat):
    return (y-yhat)**2

# Cell
def soft_gating(errors, eta, eps=1e-9):
    errors_sum = errors.sum(0)+eps
    res = (errors_sum/(errors**eta))
    return res/res.sum()

# Cell
# hide
def _flatten_ts(x):
    n_samples, n_features, ts_length = x.shape

    if isinstance(x, np.ndarray):
        x = x.swapaxes(1,2)
    else:
        x = x.permute(0,2,1)
#     tn = to_np(t)
#     tn.swapaxes(1,2).reshape(2*3,2)
    x = x.reshape(n_samples*ts_length, n_features)
    return x

def _unflatten_to_ts(x, ts_length, n_features):
    x = x.reshape(-1, ts_length, n_features)
    if isinstance(x, np.ndarray):
        x = x.swapaxes(1,2)
    else:
        x = x.permute(0,2,1)

    return x

# Cell
def create_error_matrix(targets, preds, error_function=squared_error):
    """
        N=#samples, k=#ensembles, t=forecast horizon
        targets needs to be of size Nxt or Nx1xt
        yhat needs to be of shape Nxkxt
    """
    # shape N --> Nxkxt
    if len(targets.shape)==1:
        targets = targets.reshape(-1,1,1)

    # shape Nxt --> Nxkxt
    if len(targets.shape)==2:
        # 1 is so that we can brodcast accross k ensembles
        targets = targets.reshape(len(targets), 1, -1)

    # shape N --> Nxkxt
    if len(preds.shape)==1:
        preds = preds.reshape(-1,1,1)

    # shape Nxk --> Nxkxt
    if len(preds.shape)==2:
        # 1 is for a single forecast horizon
        preds = preds.reshape(len(preds), preds.shape[1],1)
    error_matrix = error_function(targets, preds)

    if error_function == squared_error:
        error_matrix = error_matrix**0.5

    if not isinstance(error_matrix, np.ndarray):
        error_matrix = to_np(error_matrix)

    return error_matrix

# Cell
def get_global_weight(error_matrix, eta=1):
    """expected to be of shape n_samples x n_ensembles x forecast_horizon """
    # average accross all forecast horizon and samples
    # either via error_matrix.mean(0).mean(1) or error_matrix.mean(2).mean(0)

    global_weight =  error_matrix.mean(0).mean(1)

    return soft_gating(global_weight.reshape(-1,1), eta)

# Cell
def get_timedependent_weight(error_matrix, eta=1):
    """expected to be of shape n_samples x n_ensembles x forecast_horizon """
    # average accross all samples
    N,k,t =error_matrix.shape
    time_depentent_weight =  error_matrix.mean(0)

    if not isinstance(time_depentent_weight, np.ndarray):
            time_depentent_weight = to_np(time_depentent_weight)

    for t_i in range(t):
        time_depentent_weight[:,t_i] = soft_gating(time_depentent_weight[:,t_i], eta)


    return time_depentent_weight

# Cell
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
def simple_local_error_estimator(param_dict = {"n_components":5, "n_neighbors": 5}):
    pipe = Pipeline([('pca', PCA(n_components=param_dict["n_components"])),
                     ('knn', KNeighborsRegressor(n_neighbors=param_dict["n_neighbors"]))])
    return pipe

def get_local_weight(conts, error_matrix, error_expectation_regressor, do_fit=False, eta=1):
    N,k,t = error_matrix.shape

    if len(conts.shape)==3:
        conts=_flatten_ts(conts)

    if not isinstance(conts,np.ndarray):
        conts = to_np(conts)
    if do_fit:
        error_expectation_regressor = error_expectation_regressor.fit(conts, _flatten_ts(error_matrix))

    # todo check timeseries that is the same for each timestep
    local_error_expectation = error_expectation_regressor.predict(conts)

    local_error_expectation = _unflatten_to_ts(local_error_expectation,ts_length=t, n_features=k)

#     local_error_weight = normalize_weight(local_error_weight)
    if not isinstance(local_error_expectation, np.ndarray):
                local_error_expectation = to_np(local_error_expectation)
    # we should have an error matrix of shape Nxk
    for n in range(len(local_error_expectation)):
        local_error_expectation[n,:] = soft_gating(local_error_expectation[n,], eta)

    return local_error_expectation

# Cell
class CSGE(nn.Module):
    def __init__(self, source_models,
                 error_expectation_regressor=\
                     simple_local_error_estimator(param_dict = {"n_components":1, "n_neighbors": 5}),
                 eta_global=10, eta_time=10, eta_local=10,
                 is_timeseries_data=False, is_timeseries_model=False):
        """
        """
        super().__init__()
        self.source_models = np.array(source_models)
        self.is_timeseries_data = is_timeseries_data
        self.is_timeseries_model = is_timeseries_model
        self.error_expectation_regressor = error_expectation_regressor
        # fake param so that it can be used with pytorch trainers
        self.fake_param=nn.Parameter(torch.zeros((1,1), dtype=torch.float))
        self.fake_param.requires_grad =True
        self.eta_global, self.eta_time, self.eta_local = eta_global, eta_time, eta_local

        self.conversion_to_tensor = convert_to_tensor_ts if self.is_timeseries_data else convert_to_tensor

        self.ts_length = 1
        self.error_matrix = None
        self.global_weights = None
        self.timedependent_weights = None
        self.local_weights = None

    def fit(self, dls, ds_idx=0):
        if ds_idx==0:
            ds = dls.train_ds
        else:
            ds = dls.valid_ds
        self.ts_length = 1


        cats, conts, targets = self.conversion_to_tensor(ds)

        if self.is_timeseries_data:
            self.ts_length = conts.shape[2]

            conts_non_ts = _flatten_ts(conts)
            cats_non_ts = _flatten_ts(cats)
            targets_non_ts = _flatten_ts(targets)
        else:
            conts_non_ts = conts
            cats_non_ts = cats
            targets_non_ts = targets

        if self.is_timeseries_data and not self.is_timeseries_model:
            preds = get_preds(cats, conts_non_ts, self.source_models)
            preds = _unflatten_to_ts(preds, self.ts_length, 1) # assume univariate output
        else:
            preds = get_preds(cats, conts, self.source_models)

        self.error_matrix = create_error_matrix(targets, preds)

        self.global_weights = get_global_weight(self.error_matrix, self.eta_global)

        self.time_dependent_weights = get_timedependent_weight(self.error_matrix, self.eta_time)

        self.local_weight = get_local_weight(conts, self.error_matrix,
                                             self.error_expectation_regressor,
                                             do_fit=True,
                                             eta=self.eta_local)

    def calc_final_weights(self):
        final_weights = self.global_weights*self.time_dependent_weights
        final_weights = normalize_weight(final_weights)

        if self.local_weights is not None:
            local_weights = self.local_weights
            final_weights = final_weights*self.local_weights

        self.final_weights = final_weights
        self.final_weights = normalize_weight(final_weights)

        return self.final_weights

    def _predict(self, cats, conts):
        yhat = get_preds(cats, conts, self.source_models)

        self.local_weights = get_local_weight(conts, self.error_matrix,
                                              self.error_expectation_regressor,
                                              do_fit=False)


        final_weights = self.calc_final_weights()

        yhat = weight_preds(yhat, final_weights)

        return yhat

    def forward(self, cats, conts):
        yhat = self._predict(cats, conts)

        return yhat

    def predict(self, dls, ds_idx=0):
        ds = dls.train_ds
        if ds_idx==1:
            ds = dls.valid_ds

        cats, conts, _ = self.conversion_to_tensor(ds)

        yhat = self._predict(cats, conts)

        return yhat
