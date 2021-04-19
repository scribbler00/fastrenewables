import sys, glob

sys.path.append("../")
from fastai.tabular.all import *
from fastrenewables.core import *


files = glob.glob("./data/*.h5")
len(files), files[0:2]
n_files = 2
dfs = read_files(files[0:n_files], key_metadata="metadata")
cols_to_drop = L(
    "long",
    "lat",
    "loc_id",
    "target_file_name",
    "input_file_name",
    "num_train_samples",
    "num_test_samples",
)
to = TabularRenewables(
    dfs,
    y_names="PowerGeneration",
    pre_process=[DropCols(cols_to_drop), FilterByCol("TestFlag"), AddSeasonalFeatures],
    #                     TODO: Normalize per task, add task embedding and implement normalization trough task id
    procs=[NormalizePerTask, Categorify],
    add_x_to_y=False,
    include_task_id=False
    #                     splits=RandomSplitter(valid_pct=0.2)
)
to.setup()
to.process()
print(to)
# for task_id in normalize_per_task.means.index:
#     print(task_id)
# #     t = to.items[to.cont_names + ["TaskID"]]
# #     t[t["TaskID"] == task_id] = t[t["TaskID"] == task_id] - a.means[task_id]
# (
#     (to.conts[to.loc[:, "TaskID"] == task_id] - a.means.loc[task_id])
#     / a.stds.loc[task_id]
# ).describe()
# print(to.items)