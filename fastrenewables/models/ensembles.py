# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/15_ensemble_models.ipynb (unless otherwise specified).

__all__ = ['normalize_weight', 'weight_preds', 'update_single_model', 'rank_by_evidence', 'get_posterioirs',
           'get_predictive_uncertainty', 'get_preds', 'BayesModelAveraing', 'squared_error', 'soft_gating',
           'create_error_matrix', 'get_global_weight', 'get_timedependent_weight', 'simple_local_error_estimator',
           'get_local_weight', 'LocalErrorPredictor', 'turnOffTrackingStats', 'CSGE']

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
def rank_by_evidence(cats, conts, targets, models, logme=True):
    evidences = np.zeros(len(models))
    for idx,model in enumerate(models):
        if isinstance(model, BaseEstimator):
            evidences[idx] = model.log_evidence(to_np(conts), to_np(targets), logme=logme)
        else:
            evidences[idx] = model.log_evidence(cats, conts, targets, logme=logme)
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
def get_preds(cats, conts, models, convert_to_np=True):
    yhats, ystds = [], []
    ts_length = 1

    if len(conts.shape) == 3:
        ts_length = conts.shape[2]

    for idx, model in enumerate(models):
        if isinstance(model, BaseEstimator):
            yhat = model.predict(to_np(conts)).reshape(-1, ts_length)
            yhats.append(yhat)
        elif hasattr(model, "predict"):
            yhat = model.predict(cats, conts).reshape(-1, ts_length)
            if convert_to_np:
                yhat = to_np(yhat)
            yhats.append(yhat)
        elif hasattr(model, "forward"):
            yhat = model.forward(cats, conts).reshape(-1, ts_length)
            if convert_to_np:
                yhat = to_np(yhat)
            yhats.append(yhat)
        else:
            raise ValueError("Unknown prediction function.")
    if convert_to_np:
        yhats = np.concatenate(yhats, axis=1)
    else:
        yhats = torch.cat(yhats, axis=1)

    yhats = yhats.reshape(len(conts), len(models), ts_length)

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

    def fit_tensors(self, cats, conts, targets):
        if self.training:
            if self.rank_measure == "evidence":
                self.rank_measure_values, self.sord_ids = rank_by_evidence(
                    cats, conts, targets, self.source_models
                )


            if self.n_best_models != -1:
                self.source_models = self.source_models[
                    self.sord_ids[0 : self.n_best_models]
                ]
                self.rank_measure_values = self.rank_measure_values[
                    self.sord_ids[0 : self.n_best_models]
                ]

        if self.weighting_strategy == "evidence":
            self.ensemble_weights, _ = rank_by_evidence(
                cats, conts, targets, self.source_models
            )

        elif self.weighting_strategy == "posterior":
            self.ensemble_weights = get_posterioirs(
                cats, conts, targets, self.source_models
            )

        if self.weighting_strategy != "uncertainty":
            self.ensemble_weights = normalize_weight(self.ensemble_weights)

    def fit(self, dls):
        cats, conts, targets = self.conversion_to_tensor(dls.train_ds)
        self.fit_tensors(cats, conts, targets)


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

    def log_evidence(self, dls, ds_idx=0,logme=False):
        ds = dls.train_ds
        if ds_idx==1:
            ds = dls.valid_ds

        cats, conts, targets = self.conversion_to_tensor(ds)
        evidences, _ = rank_by_evidence(cats, conts, targets, self.source_models, logme=logme)

        return evidences.mean()


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
    if len(x.shape) == 2:
        return x

    n_samples, n_features, ts_length = x.shape

    if isinstance(x, np.ndarray):
        x = x.swapaxes(1,2)
    else:
        x = x.permute(0,2,1)
    x = x.reshape(n_samples*ts_length, n_features)
    return x

def _unflatten_to_ts(x, ts_length, n_features):
    if len(x) == 0 or n_features == 0:
        return x

    x = x.reshape(-1, ts_length, n_features)
    if isinstance(x, np.ndarray):
        x = x.swapaxes(1,2)
    else:
        x = x.permute(0,2,1)

    return x

# Cell
# hide
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

    return error_matrix

# Cell
# hide
def get_global_weight(error_matrix, eta=1):
    """expected to be of shape n_samples x n_ensembles x forecast_horizon """

    global_weight =  error_matrix.mean(0).mean(1)

    global_weight = Variable(global_weight)
    global_weight.requires_grad = True

    return soft_gating(global_weight.reshape(-1,1), eta)

# Cell
# hide
def get_timedependent_weight(error_matrix, eta=1):
    """
        Caclulates timedepentend input.
        Input expected to be of shape n_samples x n_ensembles x forecast_horizon
        Output is of size n_ensembles x forecast_horizon
    """
    if len(error_matrix.shape) != 3:
        raise ValueError("Error matrix is not of dimension n_samples x n_ensembles x forecast_horizon.")

    N,k,t =error_matrix.shape
    # average accross all samples
    time_depentent_weight =  error_matrix.mean(0)

    time_depentent_weight = Variable(time_depentent_weight)
    time_depentent_weight.requires_grad = False

    new_time_depentent_weight = []
    for t_i in range(t):
        new_time_depentent_weight += [soft_gating(time_depentent_weight[:,t_i], eta).reshape((k,1))]

    return torch.cat(new_time_depentent_weight, axis=1)

# Cell
# hide
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
def simple_local_error_estimator(param_dict = {"n_components":5, "n_neighbors": 5}):
    pipe = Pipeline([('pca', PCA(n_components=param_dict["n_components"])),
                     ('knn', KNeighborsRegressor(n_neighbors=param_dict["n_neighbors"]))])
    return pipe


# Cell
# hide

def get_local_weight(conts, error_expectation_regressor, eta=1):
    """
      Calculates the expected local error based on an trained error expectation reggressor.
      Based on this matrix with shape n_samples x n_ensembles the weights are calculated.
    """
    with torch.no_grad():
        if len(conts.shape)==3:
            conts=_flatten_ts(conts)

        if not isinstance(conts,np.ndarray):
            conts = to_np(conts)

        # todo check timeseries that is the same for each timestep
        local_error_expectation = error_expectation_regressor.predict(conts)

        if type(local_error_expectation) == np.ndarray:
            local_error_expectation = torch.tensor(local_error_expectation)

        if len(local_error_expectation.shape) != 2:
            raise ValueError(f"Local error matrix is not of dimension n_samples x n_ensembles x forecast_horizon. It is of size {local_error_expectation.shape}")
        N,k = local_error_expectation.shape

#     local_error_expectation = Variable(local_error_expectation)
    local_error_expectation.requires_grad = False


    # we should have an error matrix of shape Nxk (n_samples x n_ensembles)
    local_weight = []
    for n in range(len(local_error_expectation)):
        local_weight += [soft_gating(local_error_expectation[n,], eta).reshape(1,k)]
    local_weight = torch.cat(local_weight, axis=0)

    return local_weight

# Cell
# hide
class LocalErrorPredictor(BaseEstimator):
    """
       This is a wrapper for a multivariate Bayesian linear regression/extreme learning machine
       to provide local error forecasts for the CSGE.
    """
    def __init__(self, n_models=1, use_elm=True, n_hidden=200):
        self.n_models = n_models
        self.use_elm = use_elm
        self.n_hidden = n_hidden

    def fit(self, conts, errors):
        if not isinstance(conts,np.ndarray):
            conts = to_np(conts)

        if not isinstance(errors,np.ndarray):
            errors = to_np(errors)

        self.models = []
        for idx in range(self.n_models):
            model = BayesLinReg(use_fixed_point=True)
            if self.use_elm:
                model = ELM(n_hidden=self.n_hidden, prediction_model=model)
            model = model.fit(conts, errors[:,idx])
            self.models.append(model)

    def predict(self, conts):
        if len(conts.shape)==3:
            conts=_flatten_ts(conts)

        if not isinstance(conts,np.ndarray):
            conts = to_np(conts)

        preds = []
        for model in self.models:
            pred = model.predict(conts)
            preds.append(torch.tensor(pred).reshape(-1,1))

        preds = torch.cat(preds, axis=1)
        preds[preds<0] = 1
        preds[preds>1] = 1


        return preds

# Cell
def turnOffTrackingStats(module):
        if hasattr(module, "track_running_stats"):
            module.track_running_stats = False
        for childMod in module.children():
            turnOffTrackingStats(childMod)


class CSGE(nn.Module):
    def __init__(self, source_models,
                       local_error_estimator,
                       eta_global=10, eta_time=10, eta_local=10,
                 is_timeseries_model=False,
                 is_timeseries_data=False,
                 ts_length=1):
        """
        """
        super().__init__()

        self.source_models = torch.nn.Sequential(*source_models)
#         self.source_models = np.array(source_models)
        self.is_timeseries_data = is_timeseries_data
        self.is_timeseries_model = is_timeseries_model
        self.local_error_estimator = local_error_estimator
        self.conversion_to_tensor = convert_to_tensor_ts if self.is_timeseries_data else convert_to_tensor

        self.ts_length = ts_length
        self.n_ensembles = len(self.source_models)
        self.error_matrix = None
        self.global_weights = None
        self.timedependent_weights = None
        self.local_weights = None
        self.n_features = None

        self.eta_global = nn.Parameter(torch.Tensor([eta_global]))
        self.eta_local = nn.Parameter(torch.Tensor([eta_local]))
        self.eta_time = nn.Parameter(torch.Tensor([eta_time]))

    def create_preds(self, cats, conts):
        if self.is_timeseries_model:
            preds = get_preds(cats, conts, self.source_models, convert_to_np=False)
        else:
            preds = get_preds(_flatten_ts(cats), _flatten_ts(conts),
                              self.source_models, convert_to_np=False)

            preds = _unflatten_to_ts(preds, self.ts_length, self.n_ensembles)
        return preds

    def create_error_matrix(self, preds, targets):
        self.error_matrix = create_error_matrix(targets, preds)
        if not isinstance(self.error_matrix, type(torch.tensor(1))):
            raise ValueError(f"Unexpected data type: {type(self.error_matrix)}")

    def _single_data_as_ts(self, data, n_features):
        if data is None:
            return data
        elif len(data.shape) == 3:
            raise ValueError()
        else:
            return _unflatten_to_ts(data, self.ts_length, n_features)

    def _n_features(self,data):
        if data is None:
            return 0
        elif len(data) == 0:
            return 0
        else:
            return data.shape[1]

    def all_data_as_ts(self, cats, conts, targets=None):
        if self.is_timeseries_model:
            return cats,  conts, targets
        else:
            return self._single_data_as_ts(cats, n_features=self._n_features(cats)), \
                   self._single_data_as_ts(conts, n_features=self.n_features), \
                   self._single_data_as_ts(targets, n_features=self._n_features(targets))


    def fit(self, dls, ds_idx=0, n_epochs=1):
        turnOffTrackingStats(self)
        if ds_idx == 0:
            ds = dls.train_ds
        else:
            ds = dls.valid_ds

        cats, conts, targets = self.conversion_to_tensor(ds)
        self.n_features = conts.shape[1]
        cats, conts, targets = self.all_data_as_ts(cats, conts, targets)

        # we do an initial fit to setup everything and check dimensions
        with torch.no_grad():
            preds = self.create_preds(cats, conts)
            self.create_error_matrix(preds, targets)

            self.global_weights = get_global_weight(self.error_matrix, self.eta_global)

            self.local_error_estimator.fit(_flatten_ts(conts), _flatten_ts(self.error_matrix))
            self.local_weights = self.get_local_weight(conts)

            self.timedependent_weights = get_timedependent_weight(
                self.error_matrix, self.eta_time
            )

            assert len(conts) == self.error_matrix.shape[0]
            assert self.n_ensembles == self.error_matrix.shape[1]
            assert self.ts_length == self.error_matrix.shape[2]

            assert self.n_ensembles == self.global_weights.shape[0]
            assert 1 == self.global_weights.shape[1]
            assert 2==len(self.global_weights.shape)

            assert len(conts) == self.local_weights.shape[0]
            assert self.n_ensembles == self.local_weights.shape[1]
            assert 3==len(self.local_weights.shape)

            assert self.n_ensembles == self.timedependent_weights.shape[0]
            assert self.ts_length == self.timedependent_weights.shape[1]
            assert 2==len(self.timedependent_weights.shape)

    def get_local_weight(self, conts):
        local_weights = get_local_weight(_flatten_ts(conts), self.local_error_estimator, self.eta_local)

        local_weights = _unflatten_to_ts(local_weights, self.ts_length, self.n_ensembles)

        return local_weights

    def calc_final_weights(self, ):

        final_weights = self.global_weights*self.timedependent_weights
        final_weights = normalize_weight(final_weights)

        final_weights = self.local_weights*final_weights
        final_weights = normalize_weight(final_weights)

        return final_weights

    def forward(self, cats, conts):
        cats, conts, targets = self.all_data_as_ts(cats, conts, None)

        self.global_weights = get_global_weight(self.error_matrix, self.eta_global)

        self.timedependent_weights = get_timedependent_weight(
                self.error_matrix, self.eta_time
            )

        yhat = self._predict(cats, conts)

        return yhat


    def _predict(self, cats, conts):
        yhat = self.create_preds(cats, conts)

        self.local_weights = self.get_local_weight(conts)

        final_weights = self.calc_final_weights()

        yhat = weight_preds(yhat, final_weights)

        return yhat


    def predict(self, dls, ds_idx=0):
        ds = dls.train_ds
        if ds_idx==1:
            ds = dls.valid_ds
        with torch.no_grad():
            cats, conts, targets = self.conversion_to_tensor(ds)
            cats, conts, targets = self.all_data_as_ts(cats, conts, targets)

            yhat = self._predict(cats, conts)

            return yhat, targets.reshape(yhat.shape[0],yhat.shape[1])

