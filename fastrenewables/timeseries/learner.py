# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/08_timeseries.learner.ipynb (unless otherwise specified).

__all__ = ['RenewableTimeseriesLearner', 'renewable_timeseries_learner']

# Cell
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.tabular.data import *
from fastai.tabular.core import *
from fastai.tabular.model import *
from fastai.basics import *
from ..tabular.core import *
from ..tabular.model import *
from .core import *
from .data import *
from .model import *
from ..losses import VILoss
from ..utils import *
import pandas as pd

# Cell
class RenewableTimeseriesLearner(Learner):
    "`Learner` for renewable timerseries data."
    def predict(self, ds_idx=1, test_dl=None, filter=True, as_df=False):
        device = next(self.model.parameters()).device
        preds, targets = None, None
        if test_dl is not None:
            to = test_dl.train_ds
        elif ds_idx == 0:
            to = self.dls.train_ds
        elif ds_idx == 1:
            to = self.dls.valid_ds

        # to increase speed we direclty predict on all tensors
        if isinstance(to, (TimeseriesDataset)):
            with torch.no_grad():
                preds = self.model(to.cats.to(device), to.conts.to(device))

            preds, targets = to_np(preds).reshape(-1), to_np(to.ys).reshape(-1)
            if filter:
                targets, preds = filter_preds(targets, preds)
        else:
            raise NotImplementedError("Unknown type")

        if as_df:
            return pd.DataFrame({"Prediction": preds, "Target":targets}, index=to.indexes.reshape(-1))
        else:
            return preds, targets

# Cell
@delegates(Learner.__init__)
def renewable_timeseries_learner(dls, layers=None, emb_szs=None, config=None,
                                 n_out=None, y_range=None,
                                 embedding_type=EmbeddingType.Normal,
                                 input_sequence_length=None,
                                 output_sequence_length=None,
                                 sequence_transform=None,
                                 **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()

    if n_out is None:
        n_out = get_c(dls)
#     n_out = dls.train_ds.ys.shape[1]

    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"

    if layers is None: layers = [len(dls.cont_names), 200, 100, n_out]
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')

    embed_p = kwargs["embed_p"].pop() if "embed_p" in kwargs.keys() else 0.1

    emb_module = None
    if len(dls.train_ds.cat_names) > 0:
        emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
        emb_module = EmbeddingModule(None, embedding_dropout=embed_p, embedding_dimensions=emb_szs)

    model = TemporalCNN(layers, embedding_module=emb_module,
                        input_sequence_length=input_sequence_length,
                        output_sequence_length=output_sequence_length,
                        sequence_transform=sequence_transform,
                        **config)

    if embedding_type==EmbeddingType.Bayes and "loss_func" not in kwargs.keys():
        base_loss = getattr(dls.train_ds, 'loss_func', None)
        assert base_loss is not None, "Could not infer loss function from the data, please pass a loss function."
        loss_func=VILoss(model=model, base_loss=base_loss, kl_weight=0.1)
        kwargs["loss_func"] = loss_func

    return RenewableTimeseriesLearner(dls, model, **kwargs)

# Cell
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.tabular.data import *
from fastai.tabular.core import *
from fastai.tabular.model import *
from fastai.basics import *
from ..tabular.core import *
from ..tabular.data import *
from ..tabular.model import *
from ..losses import VILoss