import ast
import configparser
import os
import sys
from os import path
import pandas as pd
import multiprocessing as mp
from ast import literal_eval
import logging
import pickle

# --- Fix for cross-module imports ---
# Dynamically get the path to the DUNE root folder (2 levels up from this script)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Add the project root to sys.path so Python can find 'cluster_analysis'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ------------------------------------

from cluster_analysis.src.model_analysis.modelAnalyzer import UNSWModelAnalyzer, TONModelAnalyzer
from cluster_analysis.src.model_performance.performanceAnalyzer import select_best_unconstained_model

skip_pcfi = False

try:
    width = os.get_terminal_size().columns
except OSError:
    width = 120

cluster_info = None

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


def run_analysis(n_point):
    logger.info(f"Starting analysis of: npoint {n_point}")
    f_name = f"{results_dir_path}/{use_case}_models_{n_point}pkts.csv"

    max_depth_list = [None]
    n_trees_list = [1, 5, 10, 20, 30, 40]

    model_analyzer = None
    if use_case == 'UNSW':
        model_analyzer = UNSWModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                           flow_counts_test_file_path, classes_filter, features_filter, logger=logger,
                                           n_trees_list=n_trees_list, max_depth_list=max_depth_list)
    elif use_case == 'TON-IOT':
        model_analyzer = TONModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                          flow_counts_test_file_path, classes_filter, features_filter, logger=logger,
                                          n_trees_list=n_trees_list, max_depth_list=max_depth_list)

    model_analyzer.classes = classes_filter
    model_analyzer.classes_df = pd.DataFrame(classes_filter, columns=['class'])
    # model_analyzer.load_cluster_data(classes_filter)
    model_analyzer.analyze_model_n_packets(n_point, f_name, force_rewrite, grid_search=grid_search)
    logger.info(f"Finished analyzing n={n_point}, Results at: {results_dir_path}")



def run_model_generation(model_info_dict):
    logger.info(f"Starting the best model generation and getting the importance weights by PCFI")

    best_model_analyzer = None
    if use_case == 'UNSW':
        best_model_analyzer = UNSWModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                           flow_counts_test_file_path, classes_filter, features_filter, logger=logger)
    elif use_case == 'TON-IOT':
        best_model_analyzer = TONModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                          flow_counts_test_file_path, classes_filter, features_filter, logger=logger)

    best_model_analyzer.classes = classes_filter
    best_model_analyzer.classes_df = pd.DataFrame(classes_filter, columns=['class'])
    # Train the final selected model on the FULL train set (no grid-search
    # sampling), using the memory-safe read in _prepare_data.
    best_model_analyzer.load_full_data = True

    rf_opt, FL_class_report, test_label_names = best_model_analyzer.generate_model(model_info_dict)
    with open(f"{results_dir_path}/selected_model.pkl", 'wb') as model_file:
        pickle.dump(rf_opt, model_file)
    logger.info(f"Saved trained model to: {results_dir_path}/selected_model.pkl")
    score_per_class_df = best_model_analyzer.get_score_per_class(FL_class_report)

    # Create folder for saving results
    if not os.path.exists(f'{results_dir_path}/perf_results'):
        os.makedirs(f'{results_dir_path}/perf_results')

    if not skip_pcfi:
        from pcfi.pcfi import get_feats_importance
        feat_importance_df = get_feats_importance(rf_opt, classes_filter, model_info_dict['feats'], test_label_names)
        feat_importance_df.to_csv(f'{results_dir_path}/perf_results/importance_weights.csv', index=False)
        logger.info(f"Finished running PCFI, Results at: {results_dir_path}")
    else:
        logger.info(f"Skipped PCFI (--skip-pcfi). Results at: {results_dir_path}")

    score_per_class_df.to_csv(f'{results_dir_path}/perf_results/score_per_cluster_per_class_df.csv', index=False)


def main():
    consumed_cores = min([max_usable_cores, len(inference_points_list)*2])
    logger.info(f'Will use {consumed_cores} cores. Starting pool...')

    with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
        input_data = inference_points_list
        try:
            # issue tasks to the process pool
            pool.imap_unordered(run_analysis, input_data, chunksize=chunksize)
            # shutdown the process pool
            pool.close()
        except KeyboardInterrupt:
            logger.error("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        # wait for all issued task to complete
        pool.join()

    del pool

    logger.info("Selecting the best unconstrained model")
    best_model_info = select_best_unconstained_model(results_dir_path, inference_points_list)
    logger.info(f"The best unconstrained model: {best_model_info}")
    run_model_generation(best_model_info)


if __name__ == '__main__':
    basepath = path.dirname(__file__)
    logging.basicConfig()
    config = configparser.ConfigParser()
    config.read(path.abspath(path.join(basepath,'params.ini')))
    config['DEFAULT']['training_root'] = path.abspath(path.join(basepath, '..', '..'))
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)

    # These are parameters for the analysis
    force_rewrite = config['DEFAULT']['force_rewrite'] == 'True'
    grid_search = config['DEFAULT']['grid_search'] == 'True'
    max_usable_cores = int(config['DEFAULT']['max_usable_cores'])
    chunksize = int(config['DEFAULT']['chunksize'])

    train_data_dir_path = config[use_case]['train_data_dir_path']
    test_data_dir_path = config[use_case]['test_data_dir_path']
    inference_points_list = ast.literal_eval(config[use_case]['inference_point_list'])
    flow_counts_test_file_path = config[use_case]['flow_counts_test_file_path']
    flow_counts_train_file_path = config[use_case]['flow_counts_train_file_path']

    results_dir_path = config[use_case]['results_dir_path']
    # The analysis workers write into this folder, so it must exist before they start.
    os.makedirs(results_dir_path, exist_ok=True)
    classes_filter = ast.literal_eval(config[use_case]['classes_filter'])
    print(classes_filter)
    features_set = config[use_case]['features_set']
    if features_set not in config['FEATURES']:
        raise ValueError('Features set must be one of: ' + str(list(config._sections['FEATURES'])))
    features_filter = ast.literal_eval(config['FEATURES'][features_set])

    if '--skip-pcfi' in sys.argv:
        skip_pcfi = True

    raise SystemExit(main())
