# AUTOGENERATED BY NBDEV! DO NOT EDIT!

__all__ = ["index", "modules", "custom_doc_links", "git_url"]

index = {"contains_instance": "00a_utils.ipynb",
         "filter_preds": "00a_utils.ipynb",
         "VILoss": "00b_losses.ipynb",
         "Quantile_Score": "00b_losses.ipynb",
         "CnnMSELoss": "00b_losses.ipynb",
         "VAEReconstructionLoss": "00b_losses.ipynb",
         "GaussianNegativeLogLikelihoodLoss": "00b_losses.ipynb",
         "RSSLoss": "00b_losses.ipynb",
         "convert_layer_to_bayesian": "00c_utils_blitz.ipynb",
         "convert_to_bayesian_model": "00c_utils_blitz.ipynb",
         "set_train_mode": "00c_utils_blitz.ipynb",
         "BayesLinReg": "00d_baselines.ipynb",
         "RidgeRegression": "00d_baselines.ipynb",
         "relu": "00d_baselines.ipynb",
         "identity": "00d_baselines.ipynb",
         "sigmoid": "00d_baselines.ipynb",
         "tmp": "00d_baselines.ipynb",
         "test_acts": "00d_baselines.ipynb",
         "ELM": "00d_baselines.ipynb",
         "sample_bayes_linear_model": "00d_baselines.ipynb",
         "MCLeanPowerCurve": "00d_baselines.ipynb",
         "crps_for_quantiles": "00e_metrics.ipynb",
         "rmse_nll": "00e_metrics.ipynb",
         "normalized_sum_of_squared_residuals_np": "00e_metrics.ipynb",
         "distance_ideal_curve": "00e_metrics.ipynb",
         "normalized_sum_of_squared_residuals_torch": "00e_metrics.ipynb",
         "energy_distance": "00e_metrics.ipynb",
         "unfreeze_n_final_layer": "00f_utils_pytorch.ipynb",
         "freeze": "00f_utils_pytorch.ipynb",
         "unfreeze": "00f_utils_pytorch.ipynb",
         "print_requires_grad": "00f_utils_pytorch.ipynb",
         "monte_carlo_dropout": "00f_utils_pytorch.ipynb",
         "DummyDataset": "00g_synthetic_data.ipynb",
         "SineDataset": "00g_synthetic_data.ipynb",
         "str_to_path": "01_tabular.core.ipynb",
         "read_hdf": "01_tabular.core.ipynb",
         "read_csv": "01_tabular.core.ipynb",
         "read_files": "01_tabular.core.ipynb",
         "RenewablesTabularProc": "01_tabular.core.ipynb",
         "CreateTimeStampIndex": "01_tabular.core.ipynb",
         "get_samples_per_day": "01_tabular.core.ipynb",
         "Interpolate": "01_tabular.core.ipynb",
         "FilterInconsistentSamplesPerDay": "01_tabular.core.ipynb",
         "FilterOutliers": "01_tabular.core.ipynb",
         "FilterEinsmanWind": "01_tabular.core.ipynb",
         "AddSeasonalFeatures": "01_tabular.core.ipynb",
         "FilterInfinity": "01_tabular.core.ipynb",
         "FilterByCol": "01_tabular.core.ipynb",
         "FilterYear": "01_tabular.core.ipynb",
         "FilterHalf": "01_tabular.core.ipynb",
         "FilterMonths": "01_tabular.core.ipynb",
         "FilterDays": "01_tabular.core.ipynb",
         "DropCols": "01_tabular.core.ipynb",
         "Normalize": "01_tabular.core.ipynb",
         "BinFeatures": "01_tabular.core.ipynb",
         "RenewableSplits": "01_tabular.core.ipynb",
         "ByWeeksSplitter": "01_tabular.core.ipynb",
         "TrainTestSplitByDays": "01_tabular.core.ipynb",
         "TabularRenewables": "01_tabular.core.ipynb",
         "ReadTabBatchRenewables": "01_tabular.core.ipynb",
         "TabDataLoaderRenewables": "01_tabular.core.ipynb",
         "unique_cols": "01_tabular.core.ipynb",
         "NormalizePerTask": "01_tabular.core.ipynb",
         "VerifyAndNormalizeTarget": "01_tabular.core.ipynb",
         "TabDataset": "01_tabular.core.ipynb",
         "TabDataLoader": "01_tabular.core.ipynb",
         "TabDataLoaders": "01_tabular.core.ipynb",
         "RenewableDataLoaders": "02_tabular.data.ipynb",
         "EmbeddingType": "03_tabular.model.ipynb",
         "get_emb_sz_list": "03_tabular.model.ipynb",
         "EmbeddingModule": "03_tabular.model.ipynb",
         "MultiLayerPerceptron": "03_tabular.model.ipynb",
         "get_structure": "03_tabular.model.ipynb",
         "convert_to_tensor": "04_tabular.learner.ipynb",
         "fast_prediction": "04_tabular.learner.ipynb",
         "RenewableLearner": "04_tabular.learner.ipynb",
         "renewable_learner": "04_tabular.learner.ipynb",
         "convert_to_timeseries_representation": "05_timeseries.core.ipynb",
         "convert_to_timeseries_representation_different_lengths": "05_timeseries.core.ipynb",
         "Timeseries": "05_timeseries.core.ipynb",
         "TimeseriesDataset": "05_timeseries.core.ipynb",
         "TimeSeriesDataLoader": "05_timeseries.core.ipynb",
         "TimeSeriesDataLoaders": "05_timeseries.core.ipynb",
         "reduce_target_timeseries_to_element": "05_timeseries.core.ipynb",
         "RenewableTimeSeriesDataLoaders": "06_timeseries.data.ipynb",
         "Chomp1d": "07_timeseries.model.ipynb",
         "BasicTemporalBlock": "07_timeseries.model.ipynb",
         "ResidualBlock": "07_timeseries.model.ipynb",
         "TemporalConvNet": "07_timeseries.model.ipynb",
         "TemporalCNN": "07_timeseries.model.ipynb",
         "convert_to_tensor_ts": "08_timeseries.learner.ipynb",
         "fast_prediction_ts": "08_timeseries.learner.ipynb",
         "RenewableTimeseriesLearner": "08_timeseries.learner.ipynb",
         "renewable_timeseries_learner": "08_timeseries.learner.ipynb",
         "flatten_ts": "10_gan.model.ipynb",
         "LinBnAct": "10_gan.model.ipynb",
         "GANMLP": "10_gan.model.ipynb",
         "GANCNN": "10_gan.model.ipynb",
         "GAN": "10_gan.model.ipynb",
         "WGAN": "10_gan.model.ipynb",
         "GANLearner": "11_gan.learner.ipynb",
         "Autoencoder": "12_autoencoder_models.ipynb",
         "AutoencoderForecast": "12_autoencoder_models.ipynb",
         "UnFlatten": "12_autoencoder_models.ipynb",
         "VariationalAutoencoder": "12_autoencoder_models.ipynb",
         "MeanStdWrapper": "13_probabilistic_models.ipynb"}

modules = ["utils.py",
           "losses.py",
           "utils_blitz.py",
           "baselines.py",
           "metrics.py",
           "utils_pytorch.py",
           "synthetic_data.py",
           "tabular/core.py",
           "tabular/data.py",
           "tabular/model.py",
           "tabular/learner.py",
           "timeseries/core.py",
           "timeseries/data.py",
           "timeseries/model.py",
           "timeseries/learner.py",
           "gan/model.py",
           "gan/learner.py",
           "models/autoencoders.py",
           "models/prob.py"]

doc_url = "https://scribbler00.github.io/fastrenewables/"

git_url = "https://github.com/scribbler00/fastrenewables/tree/master/"

def custom_doc_links(name): return None
