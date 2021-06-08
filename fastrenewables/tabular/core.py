# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_tabular.core.ipynb (unless otherwise specified).

__all__ = ['str_to_path', 'read_hdf', 'read_csv', 'read_files', 'RenewablesTabularProc', 'CreateTimeStampIndex',
           'get_samples_per_day', 'Interpolate', 'FilterInconsistentSamplesPerDay', 'AddSeasonalFeatures',
           'FilterByCol', 'FilterYear', 'FilterHalf', 'FilterMonths', 'FilterDays', 'DropCols', 'Normalize',
           'BinFeatures', 'RenewableSplits', 'ByWeeksSplitter', 'TrainTestSplitByDays', 'TabularRenewables',
           'ReadTabBatchRenewables', 'TabDataLoaderRenewables', 'NormalizePerTask', 'VerifyAndNormalizeTarget',
           'TabDataset', 'TabDataLoader', 'TabDataLoaders']

# Cell
#export
import pandas as pd
from fastai.data.external import *
from fastcore.all import *
from pathlib import PosixPath
from fastcore.test import *
from fastai.tabular.all import *
import fastai
from fastai.tabular.core import _maybe_expand
from itertools import chain
from sklearn.model_selection import train_test_split

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
    include_in_new=False
    pass

# Cell
class CreateTimeStampIndex(RenewablesTabularProc):
    order=0
    include_in_new=True
    def __init__(self, col_name, offset_correction=None):
        self.col_name = col_name
        self.offset_correction = offset_correction

    def encodes(self, to):
        df = to.items

        def create_timestamp_index(df, drop_index=True):
            df.reset_index(drop=drop_index, inplace=True)
            df.rename({self.col_name: "TimeUTC"}, axis=1, inplace=True)
            #  in case the timestamp is index give it a proper timestamp,e.g., in GermanSolarFarm dataset
            if "0000-" in str(df.TimeUTC[0]):
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

        if self.col_name in df.columns:
            create_timestamp_index(df, drop_index=True)
        # properly already processed
        elif self.col_name == to.items.index.name:
            create_timestamp_index(df, drop_index=False)
        else:
            warnings.warn(f"Timetamps column {self.col_name} not in columns {df.columns} or df.index.name")

# Cell
def get_samples_per_day(df, n_samples_to_check=100, expected_samples=[8,24,96]):
    """
    Extract the amount of entries per day from the DataFrame in the first n_samples_to_check.
    Aborts, ones the first meaningful *number of samples per day* is found.
    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.

    Returns
    -------
    integer
        amount of entries per day.
    """
    samples_per_day = -1

    if len(df) == 0: return samples_per_day

    df_sorted = df.sort_index()
    indexes = df_sorted.index.unique()
    mins = 0
    for i in range(1, min(n_samples_to_check+1,len(indexes))):
        mins = (indexes[i] - indexes[i -1]).seconds // 60
        if mins == 0: continue
        if (24*60)%mins==0:
            samples_per_day = (24*60)/mins
            if samples_per_day in expected_samples: break

    if samples_per_day == -1:
        raise ValueError(f"{mins} is an unknown sampling time.")

    return int(samples_per_day)



# Cell
def _interpolate_df(df, sample_time="15Min", limit=5, drop_na=False):
        df = df[~df.index.duplicated()]
        upsampled = df.resample(sample_time)
        df  = upsampled.interpolate(method="linear", limit=limit)

        if drop_na: df = df.dropna(axis=0)

        if "Hour" in df.columns:
            df["Hour"] = df.index.hour
        if "Month" in df.columns:
            df["Month"] = df.index.month
        if "Day" in df.columns:
            df["Day"] = df.index.day
        if "Week" in df.columns:
            df["Week"] = df.index.week

        return df

# Cell
def _apply_group_by(to:pd.DataFrame, group_by_col, func, **kwargs):
    if group_by_col in to.columns:
        dfs = L()
        for k,df_g in to.groupby(group_by_col):
            dfs += func(df_g, **kwargs)
        df = pd.concat(dfs, axis=0)
    else:
        df = func(to, **kwargs)
    return df

# Cell
class Interpolate(RenewablesTabularProc):
    order=50
    include_in_new=True
    def __init__(self, sample_time = "15Min", limit=5, drop_na=True, group_by_col="TaskID"):
        self.sample_time = sample_time
        self.limit = limit
        self.drop_na = drop_na
        self.group_by_col = group_by_col

    def setups(self, to: Tabular):
        self.n_samples_per_day = get_samples_per_day(to.items)
        if self.n_samples_per_day == -1:
            warnings.warn("Could not determine samples per day. Skip processing.")

    def encodes(self, to):

        if self.n_samples_per_day == -1: return

        # if values of a columns are the same in each row (categorical features)
        # we make that those stay the same during interpolation
        if self.group_by_col in to.items.columns:
            d = defaultdict(object)
            non_unique_columns = L()
            for group_id, df in to.items.groupby(self.group_by_col):
                for c in df.columns:
                    if len(df[c].unique())==1 and c!=self.group_by_col:
                        d[(group_id,c)] = df[c][0]
                    else:
                        non_unique_columns += c

            non_unique_columns = np.unique(non_unique_columns)
        else:
            non_unique_columns = to.items.columns
        # interpolate non unique columns
        df = _apply_group_by(to.items.loc[:,np.unique(non_unique_columns)], self.group_by_col, _interpolate_df)
        to.items = df
        if self.group_by_col in to.items.columns:
            for group_id,col_name in d.keys():
                mask = to[self.group_by_col]==group_id
                to.items.loc[mask,col_name]=d[(group_id, col_name)]

        # TODO: to infer dtype through pd.inferdtype
        # use this for conversion
        # in case there is an object inside, throw an error
        if len(to.cont_names)>0:
            mask = to[to.cont_names].isna().values[:,0]
            to.items = to.items[~mask]
        # pandas converts the datatype to float if np.NaN is present, lets revert that
        to.items = to.items.convert_dtypes()

# Cell
def _create_consistent_number_of_sampler_per_day(
    df: pd.DataFrame, n_samples_per_day: int = 24
) -> pd.DataFrame:
    """
    Remove days with less than the specified amount of samples from the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        the DataFrame used for the conversion.
    n_samples_per_day : integer
        the amount of samples each day in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        the given DataFrame, now with a consistent amount of samples each day.
    """
    # Create a list of booleans, where each day with 'less than n_samples_per_day' samples is denoted with 'True'
    mask = df.groupby(pd.Grouper(freq="D")).apply(len)
    mask = (mask < n_samples_per_day)

    bad_days = list(chain.from_iterable([list(pd.date_range(start=b, periods=n_samples_per_day, freq=f'{(24 * 60) // n_samples_per_day}Min'))
                for b in mask[mask].index]))

    return df[~df.index.isin(bad_days)]



# Cell
class FilterInconsistentSamplesPerDay(RenewablesTabularProc):
    order=100
    include_in_new=True
    def __init__(self, group_by_col="TaskID"):
        self.group_by_col = group_by_col

    def setups(self, to: Tabular):
        self.n_samples_per_day = get_samples_per_day(to.items)

    def encodes(self, to):
        to.items = _apply_group_by(to.items, self.group_by_col, _create_consistent_number_of_sampler_per_day,
                        n_samples_per_day=self.n_samples_per_day)
#         to.items = _create_consistent_number_of_sampler_per_day(to.items, n_samples_per_day=self.n_samples_per_day)

        assert (to.items.shape[0]%self.n_samples_per_day) == 0, "Incorrect number of samples after filter"

# Cell
class AddSeasonalFeatures(RenewablesTabularProc):
    order=0
    include_in_new=True
    def __init__(self, as_cont=True):
        self.as_cont = as_cont

    def encodes(self, to):
        as_sin = lambda value, max_value: np.sin(2*np.pi*value/max_value)
        as_cos = lambda value, max_value: np.cos(2*np.pi*value/max_value)

        if self.as_cont:
            to.items["MonthSin"] = as_sin(to.items.index.month, 12)
            to.items["MonthCos"] = as_cos(to.items.index.month, 12)
            to.items["DaySin"] = as_sin(to.items.index.day, 31)
            to.items["DayCos"] = as_cos(to.items.index.day, 31)
            to.items["HourSin"] = as_sin(to.items.index.hour, 24)
            to.items["HourCos"] = as_cos(to.items.index.hour, 24)

        else:
            to.items["Month"] = to.items.index.month
            to.items["Day"] = to.items.index.day
            to.items["Hour"] = to.items.index.hour

# Cell
class FilterByCol(RenewablesTabularProc):
    "Drops rows by column."
    order = 9
    def __init__(self, col_name, drop=True, drop_col_after_filter=True):
        self.col_name = col_name
        self.drop = drop
        self.drop_col_after_filter=drop_col_after_filter

    def encodes(self, to):
        mask = to.items[self.col_name].astype(bool).values
        if self.drop: mask = ~mask
        to.items = to.items[mask]
        if self.drop_col_after_filter: to.items.drop(self.col_name, axis=1, inplace=True, errors="ignore")

# Cell
class FilterYear(RenewablesTabularProc):
    "Filter a list of years. By default the years are dropped."
    order = 9
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
class FilterHalf(RenewablesTabularProc):
    "First half of the data is used for training and the other half of validation/testing."
    order = 9
    def __init__(self, drop=False, bydate=True):
        """
        Whether to drop or keep the first half.
        When bydate is true the average date, between the first and last date is used to filter the data.
        If bydate is false the amount of data is splitted by half, so that train and validation/testing have an equal amount of available data.
        """
        self.drop = drop
        self.bydate = bydate

    def setups(self, to: Tabular):
        df = to.items.sort_index()
        if self.bydate:

            self.first_date = df.index[0]
            self.last_date = df.index[-1]
            self.split_date = self.first_date + (self.last_date-self.first_date)/2
        else:
            idx = len(df)//2
            self.split_date = df.index[idx]

    def encodes(self, to):
        mask = to.items.index < self.split_date

        if not self.drop: mask = ~mask
        to.items.drop(to.items[mask].index, inplace=True)

# Cell
class FilterMonths(RenewablesTabularProc):
    "Filter dataframe for specific months."
    order = 9
    def __init__(self, months=range(1,13), drop=False):
        self.months = listify(months)
        self.drop = drop

    def encodes(self, to):
        mask = to.items.index.month.isin(self.months)
        if not self.drop: mask = ~mask
        to.items.drop(to.items[mask].index, inplace=True)

# Cell
class FilterDays(RenewablesTabularProc):
    "Filter dataframe for specific months."
    order = 10
    def __init__(self, num_days):
        self.num_days = num_days

    def setups(self, to: Tabular):
        self.n_samples_per_day = get_samples_per_day(to.items)

    def encodes(self, to):
        to.items = to.items[-(self.n_samples_per_day * self.num_days):]


# Cell
class DropCols(RenewablesTabularProc):
    "Drops rows by column name."
    include_in_new=True
    order = 10
    def __init__(self, cols):
        self.cols = listify(cols)

    def encodes(self, to):
        to.items.drop(self.cols, axis=1, inplace=True, errors="ignore")

# Cell
class Normalize(RenewablesTabularProc):
    "Normalize per TaskId"
    order = 1
    include_in_new=True
    def __init__(self, cols_to_ignore=[]):
        self.cols_to_ignore = cols_to_ignore

    def setups(self, to: Tabular):
        self.rel_cols = [c for c in to.cont_names if c not in self.cols_to_ignore]
        self.means = getattr(to, "train", to)[self.rel_cols].mean()
        self.stds = getattr(to, "train", to)[self.rel_cols].std(ddof=0) + 1e-7

    def encodes(self, to):
        to.loc[:, self.rel_cols] = (to.loc[:, self.rel_cols] - self.means) / self.stds

    def decodes(self, to):
        to.loc[:, self.rel_cols] = to.loc[:, self.rel_cols] * self.stds + self.means

# Cell
class BinFeatures(TabularProc):
    "Creates bin from categorical features."
    order = 1
    include_in_new=True
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
#export
def _add_prop(cls, o):
    setattr(cls, camel2snake(o.__class__.__name__), o)

# Cell
class RenewableSplits:
    pass


class ByWeeksSplitter(RenewableSplits):
    def __init__(self, every_n_weeks: int = 4):
        self.every_n_weeks = every_n_weeks

    @staticmethod
    def _inner(cur_dataset, every_n_weeks):
        # plus one for one week of validation data
        mask = ((cur_dataset.index.isocalendar().week % (every_n_weeks+1)) == 0).values
        indices = np.arange(len(cur_dataset))
        return list(indices[list(~mask)]), list(indices[list(mask)])


    def __call__(self, o):
        return self._inner(o, self.every_n_weeks)


class TrainTestSplitByDays(RenewableSplits):
    def __init__(self, test_size: float = 0.25):
        self.test_size = test_size

    @staticmethod
    def _inner(cur_dataset, test_size):
        unique_days = np.unique(cur_dataset.index.date)
        train_days, test_days = train_test_split(
            unique_days, random_state=42, test_size=0.25
        )
        train_mask = np.array([True if d in train_days else False for d in cur_dataset.index.date])

        indices = np.arange(len(cur_dataset))
        return list(indices[list(train_mask)]), list(indices[list(~train_mask)])


    def __call__(self, o):
        return self._inner(o, self.test_size)

# Cell
class TabularRenewables(TabularPandas):
    def __init__(self, dfs, procs=None, cat_names=None, cont_names=None, do_setup=True, reduce_memory=False,
                 y_names=None, add_y_to_x=False, add_x_to_y=False,
                 pre_process=None, device=None, splits=None, y_block=RegressionBlock(),
                 group_id="TaskID",
                inplace=False):

        self.pre_process = pre_process
        self._original_pre_process = self.pre_process
        cont_names = listify(cont_names)
        cat_names = listify(cat_names)
        y_names = listify(y_names)
        self.pre_process = listify(pre_process)
        self.group_id = group_id

        for pp in listify(procs):
            if isinstance(pp, RenewablesTabularProc):
                warnings.warn(f"Element {pp} of procs is RenewablesTabularProc, might not work with TabularPandas.")


        if len(self.pre_process) > 0:
            self.prepared_to = TabularPandas(dfs, y_names=y_names,
                                             procs=self.pre_process, cont_names=cont_names,
                                          do_setup=True, reduce_memory=False, inplace=inplace, y_block=y_block)
            self.pre_process = self.prepared_to.procs
            prepared_df = self.prepared_to.items
            for pp in self.pre_process:
                if getattr(pp, "include_in_new", False): _add_prop(self, pp)
        else:
            prepared_df = dfs

        if splits is not None:
            if isinstance(splits, RenewableSplits): self.splits = splits(prepared_df)
            else: self.splits = splits(range_of(prepared_df))
        else:
            self.splits = None

        super().__init__(prepared_df,
            procs=procs,
            cat_names=cat_names,
            cont_names=cont_names,
            y_names=y_names,
            splits=self.splits,
            do_setup=do_setup,
            inplace=inplace,
            y_block=y_block,
            reduce_memory=reduce_memory)

    def new(self, df, pre_process=None, splits=None, include_preprocess=False):
        pre_process = listify(pre_process)
        if include_preprocess:
            for pp in self._original_pre_process:
                if getattr(pp, "include_in_new", False):
                    pre_process += [pp]

        return type(self)(df, do_setup=False, reduce_memory=False, y_block=TransformBlock(),
                          pre_process=pre_process, splits=splits,
                          **attrdict(self, 'procs','cat_names','cont_names','y_names', 'device', 'group_id'))

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
        # TODO: some pre-processing causes to.conts.values of type object, while types
        # of the dataframe are float, therefore assure conversion through astype
        # --> this is caused by Interpolate
        else: res = (tensor(to.cats).long(),tensor(to.conts.astype(float)).float())
        ys = [n for n in to.y_names if n in to.items.columns]
        # same problem as above with type of to.targ
        # preliminary bug fix
        # is continous target?
        if getattr(to, 'regression_setup', False):
            ys_type = float
        else:
            ys_type = int
        if len(ys) == len(to.y_names): res = res + (tensor(to.targ.astype(ys_type)),)
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
    "Normalize per TaskId. Either via z-normalization (znorm) or min-max normalization (minmaxnorm)"
    order = 1
    include_in_new=True
    def __init__(self, task_id_col="TaskID", ignore_cont_cols=[], norm_type="znorm"):
        self.task_id_col = task_id_col
        self.ignore_cont_cols = ignore_cont_cols
        self.norm_type = norm_type
        if self.norm_type not in ["znorm", "minmaxnorm"]:
            raise ValueError("normtype needs to one of znorm or minmaxnorm")
    def setups(self, to:Tabular):
        self.relevant_cols = L(c for c in to.cont_names if c not in self.ignore_cont_cols)

        self.means = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").mean()
        self.stds = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").std(ddof=0)+1e-7
        self.mins = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").min()+1e-12
        self.maxs = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").max()+1e-12


    def encodes(self, to):
        to.loc[:, self.relevant_cols] = to.loc[:, self.relevant_cols].astype(np.float64)
        for task_id in to.items[self.task_id_col].unique():
            # in case this is a new task, we update the means and stds
            if task_id not in self.means.index:
                mu = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").mean()
                std = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").std(ddof=0)+1e-7
                min_v = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").min()+1e-12
                max_v = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").max()+1e-12

                self.means= self.means.append(mu)
                self.stds = self.stds.append(std)
                self.mins= self.means.append(min_v)
                self.maxs = self.stds.append(max_v)


            mask = to.loc[:,self.task_id_col] == task_id
            if self.norm_type == "znorm":
                to.loc[mask, self.relevant_cols] = ((to.loc[mask, self.relevant_cols] - self.means.loc[task_id]) / self.stds.loc[task_id])
            elif self.norm_type == "minmaxnorm":
                to.loc[mask, self.relevant_cols] = ((to.loc[mask, self.relevant_cols] - self.mins.loc[task_id]) / (self.maxs.loc[task_id]-self.mins.loc[task_id]))



    def decodes(self, to, split_idx=None):

        for task_id in to.items[self.task_id_col].unique():
            # in case this is a new task, we update the means and stds
            if task_id not in self.means.index:
                warnings.warn("Missing task id, could not decode.")

            mask = to.loc[:,self.task_id_col] == task_id
            if self.norm_type == "znorm":
                to.loc[mask, self.relevant_cols] = to.conts[mask] * self.stds.loc[task_id] + self.means.loc[task_id]
            elif self.norm_type == "minmaxnorm":
                to.loc[mask, self.relevant_cols] = to.conts[mask] * (self.maxs.loc[task_id]-self.mins.loc[task_id]) + self.mins.loc[task_id]
        return to

# Cell
class VerifyAndNormalizeTarget(TabularProc):
    "Normalize per TaskId"
    order = 1
    include_in_new=True
    def __init__(self, reset_min_value=0.0, reset_max_value=1.05, \
                 max_value_for_normalization=1.5, task_id_col="TaskID",):
        self.task_id_col = task_id_col
        self.reset_min_value, self.reset_max_value = reset_min_value, reset_max_value
        self.max_value_for_normalization = max_value_for_normalization
    def setups(self, to:Tabular):
        self.relevant_cols = to.y_names

        self.maxs = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").max()
        self.mins = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").min()+1e-9


    def encodes(self, to):
#         return
        to.loc[:, self.relevant_cols] = to.loc[:, self.relevant_cols].astype(np.float64)
        for task_id in to.items[self.task_id_col].unique():
            # in case this is a new task, we update the maxs and mins
            if task_id not in self.maxs.index:
                task_max = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").max()
                task_min = getattr(to, 'train', to)[self.relevant_cols + "TaskID"].groupby("TaskID").min()+1e-9

                self.maxs.append(task_max)
                self.mins.append(task_min)


            mask = to.loc[:,self.task_id_col] == task_id


            if (to.loc[mask, self.relevant_cols].max() > self.max_value_for_normalization).any():
                to.loc[mask, self.relevant_cols] = (to.loc[mask, self.relevant_cols] - self.mins.loc[task_id]) \
                                                            / (self.maxs.loc[task_id] - self.mins.loc[task_id])

            for cur_relevant_column in self.relevant_cols:
                to.loc[mask, cur_relevant_column] = to.loc[mask, cur_relevant_column].where(~(to.loc[mask, cur_relevant_column] > self.reset_max_value),
                                                       self.reset_max_value)
                to.loc[mask, cur_relevant_column] = to.loc[mask, cur_relevant_column].where(~(to.loc[mask, cur_relevant_column] < self.reset_min_value),
                                                       self.reset_min_value)
        return to


# Cell
class TabDataset(fastuple):
    "A dataset from a `TabularRenewable` object"
    # Stolen from https://muellerzr.github.io/fastblog/2020/04/22/TabularNumpy.html
    def __init__(self, to):
        self.cats = tensor(to_np(to.cats).astype(np.long))
        self.conts = tensor(to_np(to.conts).astype(np.float32))

        if getattr(to, 'regression_setup', False):
            ys_type = np.float32
        else:
            ys_type = np.long
        self.ys = tensor(to_np(to.ys).astype(ys_type))

        self.cont_names = to.cont_names
        self.cat_names = to.cat_names
        self.y_names = to.y_names

    def __getitem__(self, idx):
        idx = idx[0]
        return self.cats[idx:idx+self.bs], self.conts[idx:idx+self.bs], self.ys[idx:idx+self.bs]

    def __len__(self): return len(self.cats)

    def show(self, max_n=10, **kwargs):
        df_cont = pd.DataFrame(data=self.conts[:max_n], columns=self.cont_names)
        df_cat = pd.DataFrame(data=self.cats[:max_n], columns=self.cat_names)
        df_y = pd.DataFrame(data=self.ys[:max_n], columns=self.y_names)
        display_df(pd.concat([df_cont, df_cat, df_y], axis=1))

    def show_batch(self, max_n=10, **kwargs):
        self.show()


# Cell
class TabDataLoader(DataLoader):
    def __init__(self, dataset, bs=32, num_workers=0, device='cuda',
                 to_device=True, shuffle=False, drop_last=True,**kwargs):
        "A `DataLoader` based on a `TabDataset"
        device = device if torch.cuda.is_available() else "cpu"

        super().__init__(dataset, bs=bs, num_workers=num_workers, shuffle=shuffle,
                         device=device, drop_last=drop_last, **kwargs)

        self.dataset.bs=bs
        if to_device:self.to_device()

    def create_item(self, s): return s

    def to_device(self, device=None):
        if device is None: device = self.device
        self.dataset.cats.to(device)
        self.dataset.conts.to(device)
        self.dataset.ys.to(device)

    def create_batch(self, b):
        "Create a batch of data"
        cat, cont, y = self.dataset[b]
        return cat.to(self.device), cont.to(self.device), y.to(self.device)

    def get_idxs(self):
        "Get index's to select"
        idxs = Inf.count if self.indexed else Inf.nones
        if self.n is not None: idxs = list(range(len(self.dataset)))
        return idxs

    def shuffle_fn(self):
        "Shuffle the interior dataset"
        rng = np.random.permutation(len(self.dataset))
        self.dataset.cats = self.dataset.cats[rng]
        self.dataset.conts = self.dataset.conts[rng]
        self.dataset.ys = self.dataset.ys[rng]

# Cell
class TabDataLoaders(DataLoaders):
    def __init__(self, to, bs=64, val_bs=None, shuffle_train=True, device='cpu', **kwargs):
        train_ds = TabDataset(to.train)
        valid_ds = TabDataset(to.valid)
        val_bs = bs if val_bs is None else val_bs
        train = TabDataLoader(train_ds, bs=bs, shuffle=shuffle_train, device=device, **kwargs)
        valid = TabDataLoader(valid_ds, bs=val_bs, shuffle=False, device=device, **kwargs)
        super().__init__(train, valid, device=device, **kwargs)
