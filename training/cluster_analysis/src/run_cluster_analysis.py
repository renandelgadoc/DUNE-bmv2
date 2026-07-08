import ast
import configparser
import os
import pickle
from os import path
from sklearn.tree import export_text

from model_performance.performanceAnalyzer import calculate_TOTAL_TCAM_usage
from model_analysis.modelAnalyzer import UNSWModelAnalyzer, TONModelAnalyzer
from model_performance.performanceAnalyzer import calculate_f1_score, select_best_models_per_cluster, \
    append_best_models_info_to_cluster_info, generate_score_per_class_report_for_best_models, \
    generate_ensemble_score_report_ofsp
from tabulate import tabulate
import multiprocessing as mp
from ast import literal_eval
from itertools import product
import pandas as pd
import logging

try:
    width = os.get_terminal_size().columns
except OSError:
    width = 120

cluster_info = None

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


def run_analysis(input_data):
    n_point = input_data[0]
    cluster_id = input_data[1]
    try:
        return __run_analysis(n_point, cluster_id)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []

def __run_analysis(n_point, cluster_id):
    logger.info(f"Starting analysis of: Cluster id: {cluster_id}, npoint {n_point}")
    f_name = f"{results_dir_path}/{use_case}_models_{n_point}pkts_Cluster{cluster_id}.csv"
    model_analyzer = None
    # an even number of trees is problematic for implementing the voting table. Try to avoid them.
    n_trees_list = [1, 3, 5]
    # Max values per TCAM table, i.e., 85 would require 2 TCAM tables
    max_leaves_list = [41, 85, 129, 173, 217, 261, 305, 349, 393, 437, 481, 500]
    if use_case == 'UNSW':
        model_analyzer = UNSWModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                           flow_counts_test_file_path, classes_filter, features_filter,
                                           cluster_data_file_path, logger,
                                           n_trees_list=n_trees_list, max_leaves_list=max_leaves_list)
    elif use_case == 'TON-IOT':
        model_analyzer = TONModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                          flow_counts_test_file_path, classes_filter, features_filter,
                                          cluster_data_file_path, logger,
                                          n_trees_list=n_trees_list, max_leaves_list=max_leaves_list)
    model_analyzer.partitioning_type = partitioning_type
    model_analyzer.load_cluster_data(cluster_info.loc[cluster_id])
    model_analyzer.analyze_model_n_packets(n_point, f_name, force_rewrite, grid_search)
    logger.info(f"Finished analyzing n={n_point}, Cluster={cluster_id}. Results at: {results_dir_path}")


def generate_and_save_best_models(best_models_df, cluster_info_local):
    """Generate and save the best RF model for each cluster using pickle."""
    models_subdir = f"{results_dir_path}/models"
    os.makedirs(models_subdir, exist_ok=True)
    for cluster_id, row in best_models_df.iterrows():
        model_info = {}
        # Extract depth: stored as a string repr of a list (one per estimator), take the max
        depth = row['depth']
        if isinstance(depth, str):
            depth = literal_eval(depth)
        if isinstance(depth, list):
            model_info['depth'] = max(depth)
        else:
            model_info['depth'] = int(depth) if depth is not None else None

        model_info['tree'] = int(row['tree'])
        model_info['feats'] = row['feats']
        model_info['npkts'] = int(row['N'])

        # N_Leaves: scalar (simple analysis) or list (grid search)
        n_leaves = row['N_Leaves']
        if isinstance(n_leaves, list):
            model_info['max_leaf_nodes'] = max(n_leaves)
        else:
            model_info['max_leaf_nodes'] = int(n_leaves)

        # Create the appropriate model analyzer for this cluster
        model_analyzer = None
        if use_case == 'UNSW':
            model_analyzer = UNSWModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                               flow_counts_test_file_path, classes_filter, features_filter,
                                               cluster_data_file_path, logger)
        elif use_case == 'TON-IOT':
            model_analyzer = TONModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                              flow_counts_test_file_path, classes_filter, features_filter,
                                              cluster_data_file_path, logger)

        # Set cluster data directly (cluster_info is already indexed by Cluster, so 'Cluster' key is gone)
        cluster_row = cluster_info_local.loc[cluster_id]
        model_analyzer.partitioning_type = partitioning_type
        if partitioning_type == 'ofsp':
            # OFSP: every view covers all classes, no 'Other'. Deterministic,
            # shared order so all views agree on the integer<->class mapping.
            # 'normal' forced to index 0 so it wins the ensemble majority-vote
            # tie-break (ambiguous packets default to benign); rest sorted. This
            # is the encoding baked into the SAVED models, so it must match
            # load_cluster_data and calculate_score.build_class_order_from_cluster_info.
            _s = set(cluster_row['Class List'])
            classes = (['normal'] if 'normal' in _s else []) + sorted(c for c in _s if c != 'normal')
        else:
            # sorted(set(...)) is deterministic: list(set(...)) has hash-randomised
            # order, so the saved model's integer<->class mapping could differ from
            # run to run and no longer match the P4 deployment / scoring order.
            classes = sorted(set(cluster_row['Class List']))
            classes.append('Other')
        model_analyzer.classes = classes
        model_analyzer.classes_df = pd.DataFrame(classes, columns=['class']).reset_index()
        model_analyzer.feature_list = cluster_row['Feature List']
        # The final saved model is trained on the FULL train set (no grid-search
        # sampling), using the memory-safe streaming read in _prepare_data.
        model_analyzer.load_full_data = True

        logger.info(f"Generating best model for Cluster {cluster_id} (n={model_info['npkts']}, "
                    f"trees={model_info['tree']}, leaves={model_info['max_leaf_nodes']})")
        rf_model, FL_class_report, test_label_names = model_analyzer.generate_model(model_info)

        model_path = f"{models_subdir}/selected_model_cluster{cluster_id}.pkl"
        with open(model_path, 'wb') as model_file:
            pickle.dump(rf_model, model_file)
        logger.info(f"Saved trained model for Cluster {cluster_id} to: {model_path}")

        # Tree text export disabled: these .txt dumps were inspection-only and not
        # consumed by any downstream step.
        # for tree_idx, estimator in enumerate(rf_model.estimators_):
        #     tree_txt_path = f"{models_subdir}/cluster{cluster_id}_tree{tree_idx}.txt"
        #     tree_text = export_text(estimator, feature_names=model_info['feats'])
        #     with open(tree_txt_path, 'w') as tree_file:
        #         tree_file.write(tree_text)
        #     logger.info(f"Saved tree {tree_idx} text for Cluster {cluster_id} to: {tree_txt_path}")


def main():
    global cluster_info
    cluster_id_list = cluster_info['Cluster'].to_list()
    consumed_cores = min([max_usable_cores, len(inference_points_list) * len(cluster_id_list)])
    logger.info(f'Will use {consumed_cores} cores. Starting pool...')

    input_data = list(product(inference_points_list, cluster_id_list))
    with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
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
    # __run_analysis(input_data[0][0], input_data[0][1])

    #ToDo: only high-level functions should be called here
    cluster_info = pd.read_csv(cluster_data_file_path, converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
    # cluster_info = cluster_info.drop(['Unnamed: 0'], axis=1)
    cluster_info = cluster_info.set_index('Cluster', drop=True)
    if partitioning_type == 'ofsp':
        # OFSP: every view shares the same full class list -> de-duplicate
        # (cluster_info['Class List'].sum() would repeat each class once per view).
        # 'normal' forced to index 0 (ensemble tie-break -> benign); rest sorted.
        _s = set(cluster_info['Class List'].iloc[0])
        classes = (['normal'] if 'normal' in _s else []) + sorted(c for c in _s if c != 'normal')
    else:
        classes = cluster_info['Class List'].sum()
        classes.sort()

    flow_pkt_counts = pd.read_csv(flow_counts_test_file_path)
    if use_case == 'TON-IOT':
        support = flow_pkt_counts['type'].value_counts().loc[classes].sort_index()
    else:
        support = flow_pkt_counts['label'].value_counts().loc[classes].sort_index()

    logger.info("Selecting the best models for each cluster...")
    best_models_df = select_best_models_per_cluster(cluster_info, results_dir_path)
    cluster_info = append_best_models_info_to_cluster_info(cluster_info, best_models_df)

    logger.info("Generating and saving best models for each cluster...")
    generate_and_save_best_models(best_models_df, cluster_info)

    # Create folder for saving results
    if not os.path.exists(f'{results_dir_path}/perf_results'):
        os.makedirs(f'{results_dir_path}/perf_results')

    cluster_info.to_csv(f'{results_dir_path}/perf_results/cluster_info_df.csv')

    print('=' * width)
    if partitioning_type == 'ofsp':
        # OFSP/MEL: the meaningful metric is the performance-weighted ensemble
        # vote over all views (not a per-cluster stitch-up), and it needs the
        # saved per-view models, so it runs after generate_and_save_best_models.
        if use_case == 'UNSW':
            ensemble_analyzer = UNSWModelAnalyzer(train_data_dir_path, test_data_dir_path,
                                                  flow_counts_train_file_path, flow_counts_test_file_path,
                                                  classes_filter, features_filter, cluster_data_file_path, logger)
        else:
            ensemble_analyzer = TONModelAnalyzer(train_data_dir_path, test_data_dir_path,
                                                 flow_counts_train_file_path, flow_counts_test_file_path,
                                                 classes_filter, features_filter, cluster_data_file_path, logger)
        ensemble_analyzer.partitioning_type = 'ofsp'
        ensemble_analyzer.classes = classes
        ensemble_analyzer.classes_df = pd.DataFrame(classes, columns=['class']).reset_index()
        ensemble_report, score_per_class_df = generate_ensemble_score_report_ofsp(
            ensemble_analyzer, best_models_df, results_dir_path, use_case, inference_points_list[0])
        score_per_class_df.to_csv(f'{results_dir_path}/perf_results//score_per_cluster_per_class_df.csv')
        macro = ensemble_report['macro avg']['f1-score']
        weighted = ensemble_report['weighted avg']['f1-score']
        print(f"Ensemble (OFSP/MEL weighted vote) F1 score:\n \tMacro: {macro}\n  \tWeighted: {weighted}")
    else:
        score_per_class_df = generate_score_per_class_report_for_best_models(classes, best_models_df, support)
        score_per_class_df.to_csv(f'{results_dir_path}/perf_results//score_per_cluster_per_class_df.csv')
        score = calculate_f1_score(score_per_class_df)
        print(f"Average F1 score:\n \tMacro: {score[0]}\n  \tWeighted: {score[1]}")
    tcam = calculate_TOTAL_TCAM_usage(cluster_info, partitioning_type)
    print(f"TOTAL TCAM usage: {tcam}")
    # print the cluster info with the best model
    print(f"Final Models information:")
    print(tabulate(cluster_info, headers='keys', tablefmt='psql'))
    print("END")
    # except ValueError as e:
    #     logger.error(f"F1 score could not be calculated. The following error was raised: {e}")


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
    partitioning_type = config['DEFAULT'].get('partitioning_type', 'spp')

    train_data_dir_path = config[use_case]['train_data_dir_path']
    test_data_dir_path = config[use_case]['test_data_dir_path']
    inference_points_list = ast.literal_eval(config[use_case]['inference_point_list'])
    flow_counts_test_file_path = config[use_case]['flow_counts_test_file_path']
    flow_counts_train_file_path = config[use_case]['flow_counts_train_file_path']

    results_dir_path = config[use_case]['results_dir_path']
    classes_filter = ast.literal_eval(config[use_case]['classes_filter'])
    features_set = config[use_case]['features_set']
    if features_set not in config['FEATURES']:
        raise ValueError('Features set must be one of: ' + str(list(config._sections['FEATURES'])))
    features_filter = ast.literal_eval(config['FEATURES'][features_set])

    # This is the main input to the program. Should be loaded into an object....
    cluster_data_file_path = config[use_case]['cluster_data_file_path']
    cluster_info = pd.read_csv(cluster_data_file_path,
                               converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
    raise SystemExit(main())
