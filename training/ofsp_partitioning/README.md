# OFSP Partitioning
Partition of the features into views with the Optimal Feature Set Partitioning (OFSP) method. Each view gets a
disjoint subset of the features and its sub-model classifies all the classes; in the data plane, the views decide by
majority vote. This is the alternative to the class partitioning of `model_partitioning`.

## Getting Started

Install the dependencies of the whole pipeline from the repository root (see the README there):
```bash
pip install -r requirements.txt
```

All configurable parameters are provided via the `src/params.ini` file.

### Configurable Parameters
Default section:
- `log_level`: the logging level.
- `use_case`: the parameters section to use (`TON-IOT`).
- `max_usable_cores`: cores available to the process.
- `n_views`: the number of views, i.e. of switches (5).
- `random_state`, `n_estimators`, `max_depth`: the Random Forest that scores each view during the search.
- `objective`: what the search maximizes on the out-of-bag predictions: `accuracy`, `flow_acc` (accuracy weighted
  by flow) or `macro_f1` (macro F1 weighted by flow).
- `scope`: how a feature is placed: `view` puts it in the view whose own score it increases the most; `ensemble`
  puts it in the view that gives the best score of all the views voting together.

Use case section (`[TON-IOT]`):
- `classes_filter`: the classes used.
- `train_data_dir_path`: the training data of `data_generation` (`train_<n>_pkts.parquet`).
- `flow_counts_train_file_path`: the packet count of each training flow, used as sample weight.
- `results_dir_path`: where the solution is written.
- `inference_point_list`: the inference points; one partition is built for each.
- `features_set`: the feature list of the `[FEATURES]` section to partition.

### Running the analysis
After configuring the `src/params.ini` file you can trigger the analysis by:
```bash
cd src
python3 run_ofsp_partitioning.py
```

### Pipeline Outputs
For each inference point n, `results/TON-IOT_OFSP_solution_<n>pkts.csv`, with the features of each view. It is the
`cluster_data_file_path` of `cluster_analysis` when `partitioning_type = ofsp`.
