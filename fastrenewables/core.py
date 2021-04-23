# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_core.ipynb (unless otherwise specified).

__all__ = ['str_to_path', 'read_hdf', 'read_csv', 'read_files', 'RenewablesTabularProc', 'CreateTimeStampIndex',
           'AddSeasonalFeatures', 'FilterYear', 'DropCols', 'FilterByCol', 'FilterMonths', 'BinFeatures',
           'TabularRenewables', 'ReadTabBatchRenewables', 'TabDataLoaderRenewables', 'NormalizePerTask']

# Cell
#export
import pandas as pd
from nbdev.showdoc import *
from fastai.data.external import *
from fastcore.all import *
from pathlib import PosixPath
from fastcore.test import *
from fastai.tabular.all import *
import fastai
from fastai.tabular.core import _maybe_expand

# Cell
#export
def str_to_path(file: str):
    "Convers a string to a Posixpath."
    if isinstance(file, str) and "~" in file:
        file = os.path.expanduser(file)

    file = Path(file)

    return file

# Cell
def read_hdf(file:PosixPath, key: str = "/powerdata", key_metadata=None):
    "Reads a hdf5 table based on the given key."
    file = str_to_path(file)
    if "/" not in key: key = "/" + key
    with pd.HDFStore(file, "r") as store:
        if key in store.keys():
            df = store[key]
            if key_metadata is not None:
                df_meta = store[key_metadata]
                for c in df_meta: df[c] = df_meta[c].values[0]
        else:
            df = pd.DataFrame()
    return df

# Cell
def read_csv(file:PosixPath, sep:str =";"):
    "Reads a csv file."
    file = str_to_path(file)
    df = pd.read_csv(str(file), sep=sep)
    df.drop(["Unnamed: 0"], inplace=True, axis=1, errors="ignore")
    return df

# Cell
def read_files(
    files:PosixPath,
    key:str ="/powerdata",
    key_metadata=None,
    sep:str=";",
    add_task_id=True
) -> pd.DataFrame:
    "Reads a number of CSV or HDF5 files depending on file ending."

    files = listify(files)
    dfs=L()
    for task_id,file in enumerate(files):
        if isinstance(file, str):
            file = str_to_path(file)

        if file.suffix == ".h5":
            df = read_hdf(file, key, key_metadata=key_metadata)
        elif file.suffix == ".csv":
            df = read_csv(file, sep=";")
        else:
            raise f"File ending of file {file} not supported."
        if add_task_id:df["TaskID"]=task_id
        dfs += df

    return dfs

# Cell
# this is merely a class to differentiate between fastai processing and renewbale pre-processing functionality
class RenewablesTabularProc(TabularProc):
    pass

# Cell
class CreateTimeStampIndex(RenewablesTabularProc):
    order=0
    def __init__(self, col_name, offset_correction=None):
        self.col_name = col_name
        self.offset_correction = offset_correction

    def encodes(self, to):
        df = to.items


        if self.col_name in df.columns:
            df.reset_index(drop=True, inplace=True)
            df.rename({self.col_name: "TimeUTC"}, axis=1, inplace=True)
            #  in case the timestamp is index give it a proper timestamp,e.g., in GermanSolarFarm dataset
            if "0000-" in df.TimeUTC[0]:
                df.TimeUTC = df.TimeUTC.apply(
                    lambda x: x.replace("0000-", "2015-").replace("0001-", "2016-")
                )
            df.TimeUTC = pd.to_datetime(df.TimeUTC, infer_datetime_format=True, utc=True)
            df.set_index("TimeUTC", inplace=True)
            df.index = df.index.rename("TimeUTC")

            #  for GermanSolarFarm, the index is not corret. Should have a three hour resolution but is one...
            if self.offset_correction is not None:
                i, new_index = 0, []
                for cur_index in df.index:
                    new_index.append(cur_index + pd.DateOffset(hours=i))
                    i += self.offset_correction
                df.index = new_index

        else:  warnings.warn(f"Timetamps column {self.col_name} not in columns {df.columns}")

# Cell
class AddSeasonalFeatures(RenewablesTabularProc):
    order=0
    def encodes(self, to):
        to.items["Month"] = to.items.index.month
        to.items["Day"] = to.items.index.day
        to.items["Hour"] = to.items.index.hour

# Cell
class FilterYear(RenewablesTabularProc):
    "Filter a list of years. By default the years are dropped."
    order = 10
    def __init__(self, year, drop=True):
        "year(s) to filter, whether to drop or keep the years."
        year = listify(year)
        self.year = L(int(y) for y in year)
        self.drop = drop

    def encodes(self, to):
        mask = None
        for y in self.year:
            cur_mask = to.items.index.year == y
            if mask is None: mask = cur_mask
            else: mask = mask | cur_mask

        if not self.drop: mask = ~mask
        to.items.drop(to.items[mask].index, inplace=True)

# Cell
class DropCols(RenewablesTabularProc):
    "Drops rows by column name."
    order = 10
    def __init__(self, cols):
        self.cols = listify(cols)

    def encodes(self, to):
        to.items.drop(self.cols, axis=1, inplace=True, errors="ignore")

# Cell
class FilterByCol(RenewablesTabularProc):
    "Drops rows by column."
    order = 0
    def __init__(self, col_name, drop=True, drop_col_after_filter=True):
        self.col_name = col_name
        self.drop = drop
        self.drop_col_after_filter=drop_col_after_filter

    def encodes(self, to):
        mask = to.items[self.col_name].astype(bool).values
        if not self.drop: mask = ~mask
        to.items.drop(to.items[mask].index, inplace=True)
        if self.drop_col_after_filter: to.items.drop(self.col_name, axis=1, inplace=True, errors="ignore")


# Cell
class FilterMonths(RenewablesTabularProc):
    "Filter dataframe for specific months."
    order = 10
    def __init__(self, months=range(1,13), drop=False):
        self.months = listify(months)
        self.drop = drop

    def encodes(self, to):
        mask = to.items.index.month.isin(self.months)
        if not self.drop: mask = ~mask
        to.items.drop(to.items[mask].index, inplace=True)

# Cell
class BinFeatures(TabularProc):
    "Creates bin from categorical features."
    order = 1
    def __init__(self, column_names, bin_sizes=5):
        # TODO: Add possiblitiy to add custom bins
        self.column_names = listify(column_names)
        self.bin_sizes = listify(bin_sizes)
        if len(self.bin_sizes) == 1: self.bin_sizes = L(self.bin_sizes[0] for _ in self.column_names)

    def setups(self, to:Tabular):
        train_to = getattr(to, 'train', to)
        self.bin_edges = {c:pd.qcut(train_to.items[c], q=bs, retbins=True)[1] for c,bs in zip(self.column_names,self.bin_sizes)}


    def encodes(self, to):
        for c in self.bin_edges.keys():
            to.items.loc[:,c] = pd.cut(to.items[c], bins=self.bin_edges[c],
                                       labels=range(1, len(self.bin_edges[c])),
                                       include_lowest=True)


# Cell
class TabularRenewables(TabularPandas):
    def __init__(self, dfs, procs=None, cat_names=None, cont_names=None, do_setup=True, reduce_memory=False,
                 y_names=None, add_y_to_x=False, add_x_to_y=False, pre_process=None, device=None, splits=None, y_block=RegressionBlock()):

        self.pre_process = pre_process

        # TODO: check whether elements in procs are of type RenewablesTabularProc and create warning

        if pre_process is not None:
            self.prepared_to = TabularPandas(dfs, y_names=y_names, procs=pre_process, cont_names=cont_names,
                                          do_setup=True, reduce_memory=False)
            prepared_df = self.prepared_to.items
        else:
            prepared_df = dfs
        if splits is not None: splits = splits(range_of(prepared_df))
        super().__init__(prepared_df,
            procs=procs,
            cat_names=cat_names,
            cont_names=cont_names,
            y_names=y_names,
            splits=splits,
            do_setup=do_setup,
            inplace=True,
            y_block=y_block,
            reduce_memory=reduce_memory)

    def new(self, df, pre_process=None, splits=None):
        return type(self)(df, do_setup=False, reduce_memory=False, y_block=TransformBlock(),
                          pre_process=pre_process, splits=splits,
                          **attrdict(self, 'procs','cat_names','cont_names','y_names', 'device'))

    def show(self, max_n=10, **kwargs):
        to_tmp = self.new(self.all_cols[:max_n])
        to_tmp.items["TaskID"] = self.items.TaskID[:max_n]
        display_df(to_tmp.decode().items)


# Cell

class ReadTabBatchRenewables(ItemTransform):
    "Transform `TabularPandas` values into a `Tensor` with the ability to decode"
    def __init__(self, to): self.to = to.new_empty()

    def encodes(self, to):
        self.task_ids = to.items[["TaskID"]]
        if not to.with_cont: res = (tensor(to.cats).long(),)
        else: res = (tensor(to.cats).long(),tensor(to.conts).float())
        ys = [n for n in to.y_names if n in to.items.columns]
        if len(ys) == len(to.y_names): res = res + (tensor(to.targ),)
        if to.device is not None: res = to_device(res, to.device)
        return res

    def decodes(self, o):

        o = [_maybe_expand(o_) for o_ in to_np(o) if o_.size != 0]
        vals = np.concatenate(o, axis=1)
        try: df = pd.DataFrame(vals, columns=self.to.all_col_names)
        except: df = pd.DataFrame(vals, columns=self.to.x_names)

        to = self.to.new(df)
        to.items["TaskID"]=self.task_ids.values

        return to


# Cell
@delegates()
class TabDataLoaderRenewables(TfmdDL):
    "A transformed `DataLoader` for Tabular data"
    def __init__(self, dataset, bs=16, shuffle=False, after_batch=None, num_workers=0, **kwargs):
        if after_batch is None: after_batch = L(TransformBlock().batch_tfms)+ReadTabBatchRenewables(dataset)
        super().__init__(dataset, bs=bs, shuffle=shuffle, after_batch=after_batch, num_workers=num_workers, **kwargs)

    def create_batch(self, b): return self.dataset.iloc[b]
    def do_item(self, s):      return 0 if s is None else s

TabularRenewables._dl_type = TabDataLoaderRenewables

# Cell
class NormalizePerTask(TabularProc):
    "Normalize per TaskId"
    order = 1
    def __init__(self, task_id_col="TaskID"):
        self.task_id_col = task_id_col
    def setups(self, to:Tabular):
        self.means = getattr(to, 'train', to)[to.cont_names + "TaskID"].groupby("TaskID").mean()
        self.stds = getattr(to, 'train', to)[to.cont_names + "TaskID"].groupby("TaskID").std(ddof=0)+1e-7


    def encodes(self, to):
        for task_id in to.items[self.task_id_col].unique():
            # in case this is a new task, we update the means and stds
            if task_id not in self.means.index:
                mu = getattr(to, 'train', to)[to.cont_names + "TaskID"].groupby("TaskID").mean()

                self.means= self.means.append(mu)
                self.stds = self.stds.append(getattr(to, 'train', to)[to.cont_names + "TaskID"].groupby("TaskID").std(ddof=0)+1e-7)


            mask = to.loc[:,self.task_id_col] == task_id

            to.loc[mask, to.cont_names] = ((to.conts[mask] - self.means.loc[task_id]) / self.stds.loc[task_id])

    def decodes(self, to):
        for task_id in to.items[self.task_id_col].unique():
            # in case this is a new task, we update the means and stds
            if task_id not in self.means.index:
                warnings.warn("Missing task id, could not decode.")

            mask = to.loc[:,self.task_id_col] == task_id

            to.loc[mask, to.cont_names] = to.conts[mask] * self.stds.loc[task_id] + self.means.loc[task_id]
        return to