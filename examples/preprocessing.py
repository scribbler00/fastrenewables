import sys, glob

# sys.path.append("../")
from fastai.tabular.all import *
from fastrenewables.core import *


# files = glob.glob("./data/*.h5")
# len(files), files[0:2]
# n_files = 2
# dfs = read_files(files[0:n_files], key_metadata="metadata")
# cols_to_drop = L(
#     "long",
#     "lat",
#     "loc_id",
#     "target_file_name",
#     "input_file_name",
#     "num_train_samples",
#     "num_test_samples",
# )
# to = TabularRenewables(
#     dfs,
#     y_names="PowerGeneration",
#     pre_process=[DropCols(cols_to_drop), FilterByCol("TestFlag"), AddSeasonalFeatures],
#     #                     TODO: Normalize per task, add task embedding and implement normalization trough task id
#     procs=[NormalizePerTask, Categorify],
#     add_x_to_y=False,
#     include_task_id=False
#     #                     splits=RandomSplitter(valid_pct=0.2)
# )
# to.setup()
# to.process()
# print(to)
# # for task_id in normalize_per_task.means.index:
# #     print(task_id)
# # #     t = to.items[to.cont_names + ["TaskID"]]
# # #     t[t["TaskID"] == task_id] = t[t["TaskID"] == task_id] - a.means[task_id]
# # (
# #     (to.conts[to.loc[:, "TaskID"] == task_id] - a.means.loc[task_id])
# #     / a.stds.loc[task_id]
# # ).describe()
# # print(to.items)
files = glob.glob("./data/*.h5")
n_files = 2


cont_names = [
    "T_HAG_2_M",
    "RELHUM_HAG_2_M",
    "PS_SFC_0_M",
    "ASWDIFDS_SFC_0_M",
    "ASWDIRS_SFC_0_M",
    "WindSpeed58m",
    "SinWindDirection58m",
    "CosWindDirection58m",
    "WindSpeed60m",
    "SinWindDirection60m",
    "CosWindDirection60m",
    "WindSpeed58mMinus_t_1",
    "SinWindDirection58mMinus_t_1",
    "CosWindDirection58mMinus_t_1",
    "WindSpeed60mMinus_t_1",
    "SinWindDirection60mMinus_t_1",
    "CosWindDirection60mMinus_t_1",
    "WindSpeed58mPlus_t_1",
    "SinWindDirection58mPlus_t_1",
    "CosWindDirection58mPlus_t_1",
    "WindSpeed60mPlus_t_1",
    "SinWindDirection60mPlus_t_1",
    "CosWindDirection60mPlus_t_1",
]

cat_names = ["Month", "Day", "Hour", "rotor_diameter_m", "hub_height_m"]
y_names = "PowerGeneration"
pd.options.mode.chained_assignment = None
pre_process = [
    AddSeasonalFeatures,
    FilterByCol("TestFlag", drop_col_after_filter=False),
    #              DropYear(year=2020),
    #              FilterMonths([1,2,3,4]),
    NormalizePerTask,
]

procs = [NormalizePerTask, Categorify]
procs = [Categorify]

dfs = read_files(files[0:n_files], key_metadata="metadata")

to = TabularRenewables(
    pd.concat(dfs, axis=0),
    cont_names=cont_names,
    cat_names=cat_names,
    y_names=y_names,
    pre_process=pre_process,
    procs=procs,
    splits=RandomSplitter(valid_pct=0.2),
)
print(to.items.index)
dls = to.dataloaders(bs=256)
dls.show_batch()
learner = tabular_learner(dls, metrics=rmse)
print(learner.dls.train_ds.cat_names)
print(learner.model)
learner.fit_one_cycle(10)