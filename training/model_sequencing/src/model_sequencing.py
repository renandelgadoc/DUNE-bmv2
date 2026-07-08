import ast
import collections
import logging
import configparser

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tsp import order_blocks_with_tsp

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else ast.literal_eval(val)

def assign_sample_nature(row):
    '''
    Function to assign related values to each packet to specify if it will 
    run packet-level or flow-level classification
    '''
    if (row["Min Packet Length"] == -1 and
        row["Max Packet Length"] == -1 and
        row["Flow IAT Min"] == -1 and
        row["Flow IAT Max"] == -1):
        return "pkt"
    else:
        return "flw"
    

def get_class_to_cluster_map(cluster_info_df):
    '''
    Function to get the list of clusters, classes and 
    list of clusters corresponding to the given classes.
    '''
    class_to_cluster_map = {}

    for cluster_id, class_list in cluster_info_df['Class List'].items():
        for _class in class_list:
            class_to_cluster_map[_class] = cluster_id

    return class_to_cluster_map

# ToDo: The name of the function does not map its description.
#  Also, there is a lot of code which is not used in this function (lines 142-159).
#  Please, consider reusing cluster_analysis.extend_test_data_with_flow_level_results.
#  If not possible, try to generalize the function to be used in both cases.
#  In any case, unique_labels, and array_of_indices are not needed.
# def expand_rows_and_get_scores_others(y_true, y_pred, y_test_ALL, sample_nature, multiply, test_flow_pkt_cnt,
#                                       test_flow_IDs):
#     """
#     Function to calculate the score of the model in terms of Flow-Level metric and 
#     return the classification results of all packets
#     """
#     expanded_y_true = []
#     expanded_y_pred = []
#     expanded_y_true_all = []
#     #
#     expanded_weights = []
#     expanded_flow_IDs = []
    
#     for true_label, pred_label, y_test_ALL_label, nature, mult, pkt_cnt, f_id in zip(y_true, y_pred, y_test_ALL, sample_nature, multiply, test_flow_pkt_cnt, test_flow_IDs):
#         if nature == 'flw':
#             expanded_y_true.extend([true_label] * (mult+1))
#             expanded_y_pred.extend([pred_label] * (mult+1))
#             expanded_y_true_all.extend([y_test_ALL_label] * (mult+1))
#             #
#             expanded_weights.extend([1/pkt_cnt] * (mult+1))
#             expanded_flow_IDs.extend([f_id]* (mult+1))
#         else:
#             expanded_y_true.append(true_label)
#             expanded_y_pred.append(pred_label)
#             expanded_y_true_all.append(y_test_ALL_label)
#             #
#             expanded_weights.append(1/pkt_cnt)
#             expanded_flow_IDs.append(f_id)
    
#     expanded_y_true = [int(label) for label in expanded_y_true]
#     expanded_y_pred = [int(label) for label in expanded_y_pred]
    
#     return expanded_y_true, expanded_y_pred, expanded_weights, expanded_y_true_all


def analyze_model(use_case, classes_filter, npkts, n_tree, max_leaf, feats, classes, all_classes_in_clusters, flow_counts_test_file_path, flow_counts_train_file_path, train_data_dir_path, test_data_dir_path, cluster_id=None, models_dir=None):
    """
    The function that trains a RandomForestClassifier with the provided data and parameters.
    """
    import os as _os_seq, pickle as _pickle_seq

    def _memlog(tag):
        # Lightweight RSS + system-available memory print (no psutil dependency).
        try:
            with open('/proc/self/statm') as _f:
                _rss = int(_f.read().split()[1]) * 4096 / 1e9  # resident pages -> GB
            _av = float('nan')
            with open('/proc/meminfo') as _f:
                for _line in _f:
                    if _line.startswith('MemAvailable:'):
                        _av = int(_line.split()[1]) / 1e6  # kB -> GB
                        break
            logging.getLogger(f'{use_case}').info(
                f'[MEM] cluster {cluster_id} {tag}: RSS={_rss:.2f}GB MemAvailable={_av:.2f}GB')
        except Exception as _e:
            logging.getLogger(f'{use_case}').info(f'[MEM] {tag}: (read failed: {_e})')

    # Decide up front whether we can reuse cluster_analysis's saved model. If so we
    # SKIP loading/processing the (large) train parquet entirely -- it is only ever
    # needed to refit. This is what keeps memory low when reusing models.
    _model_path = None
    if models_dir is not None and cluster_id is not None:
        _model_path = _os_seq.path.join(models_dir, f"selected_model_cluster{cluster_id}.pkl")
    _have_saved_model = _model_path is not None and _os_seq.path.exists(_model_path)
    _memlog(f'enter (have_saved_model={_have_saved_model})')

    if(use_case == 'UNSW'):
        # Load Train and Test data
        train_data = pd.read_csv(train_data_dir_path+"/train_data_"+str(npkts)+".csv")
        test_data = pd.read_csv(test_data_dir_path+"/16-10-05.pcap.txt_"+str(npkts)+"_pkts.csv")

        flow_pkt_counts = pd.read_csv(flow_counts_test_file_path)

        flow_count_dict = flow_pkt_counts.set_index("flow.id")["count"].to_dict()
        # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)

        # To get packet count of each flow in train data
        packet_data = pd.read_csv("/home/beyzabutun/shared/UNSW_PCAPS/train/train_data_hybrid/UNSW_train_ALL_PKT_DATE.csv")
        packet_data = packet_data[['Flow ID', 'packet_count', 'File']]
        packet_data['File_ID'] = packet_data['Flow ID'] + ' ' + packet_data['File']
        packet_data = packet_data.drop_duplicates(subset='File_ID', keep='first')
        train_data['File_ID'] = train_data['Flow ID'] + ' ' + train_data['File']
        
        flow_count_dict_train = packet_data.set_index("File_ID")["packet_count"].to_dict()
        # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
        train_data["pkt_count"] = train_data["File_ID"].map(flow_count_dict_train)
        ###########
    if(use_case=='TON-IOT'):
        # Load Train and Test data (parquet if available, else CSV)
        import os as _os
        _train_pq = train_data_dir_path+"/train_"+str(npkts)+"_pkts.parquet"
        _test_pq = test_data_dir_path+"/test_"+str(npkts)+"_pkts.parquet"
        if not _have_saved_model:
            train_data = pd.read_parquet(_train_pq) if _os.path.exists(_train_pq) else pd.read_csv(train_data_dir_path+"/train_"+str(npkts)+"_pkts.csv")
            _memlog('after train load')
        test_data = pd.read_parquet(_test_pq) if _os.path.exists(_test_pq) else pd.read_csv(test_data_dir_path+"/test_"+str(npkts)+"_pkts.csv")
        _memlog('after test load')
        #
        flow_pkt_counts = pd.read_csv(flow_counts_test_file_path)

        if not _have_saved_model:
            flow_pkt_counts_train = pd.read_csv(flow_counts_train_file_path)
            flow_count_dict_train = flow_pkt_counts_train.set_index("Flow ID")["count"].to_dict()
            train_data["pkt_count"] = train_data["Flow ID"].map(flow_count_dict_train)

        flow_count_dict = flow_pkt_counts.set_index("Flow ID")["count"].to_dict()
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)
    

    all_minus_one = (test_data['Min Packet Length'] == -1) & (test_data['Max Packet Length'] == -1) & (test_data['Packet Length Mean'] == -1)
    # Assign values to the multiply column based on the conditions
    test_data['multiply'] = np.where(all_minus_one, 1, test_data['pkt_count'] - npkts)

    if not _have_saved_model: train_data = train_data.sample(frac=1, random_state=42)
    test_data  = test_data.sample(frac=1, random_state=42)

    if not _have_saved_model: train_data = train_data.dropna(subset=['srcport', 'dstport'])
    test_data  = test_data.dropna(subset=['srcport', 'dstport'])

    if not _have_saved_model: train_data = train_data[train_data['Label'].isin(classes_filter)]
    test_data = test_data[test_data['Label'].isin(classes_filter)]

    if not _have_saved_model: train_data['Label_NEW'] = np.where((train_data['Label'].isin(classes)), train_data['Label'], 'Other')
    test_data['Label_NEW'] = np.where((test_data['Label'].isin(classes)), test_data['Label'], 'Other')

    if not _have_saved_model: train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
    test_data['sample_nature']  = test_data.apply(assign_sample_nature, axis=1)

    if not _have_saved_model: train_data['weight'] = np.where(train_data['sample_nature'] == 'flw', (train_data['pkt_count'] - npkts + 1)/train_data['pkt_count'], 1/train_data['pkt_count'])
    if not _have_saved_model: weight_of_samples = list(train_data['weight'])

    if not _have_saved_model: train_samples = train_data[feats]
    if not _have_saved_model: train_labels = train_data['Label_NEW'].replace(classes, range(len(classes)))

    test_samples = test_data[feats]
    test_labels = test_data['Label_NEW'].replace(classes, range(len(classes)))
    test_labels_all_clusters = test_data['Label'].replace(all_classes_in_clusters, range(len(all_classes_in_clusters)))

    # Reuse the model already trained + saved by cluster_analysis instead of
    # retraining an identical forest. Same features, same sorted(set)+'Other'
    # label encoding, same random_state/bootstrap -> predictions are identical.
    # Falls back to retraining only if the .pkl is missing (see _have_saved_model above).
    if _have_saved_model:
        with open(_model_path, "rb") as _mf:
            model = _pickle_seq.load(_mf)
        logging.getLogger(f'{use_case}').info(f'Loaded saved model for cluster {cluster_id}: {_model_path} (no retrain)')
    else:
        logging.getLogger(f'{use_case}').info(f'No saved model for cluster {cluster_id} -> RETRAINING')
        model = RandomForestClassifier(n_estimators = n_tree, max_leaf_nodes=max_leaf, n_jobs=4,
                                            random_state=42, bootstrap=False)
        model.fit(train_samples[feats], train_labels, sample_weight=weight_of_samples)
    _memlog('before predict')
    y_pred = model.predict(test_samples[feats])
    _memlog('after predict')

    # --- ZERO MEMORY OPTIMIZATION ---
    # Calculate how many times each row WOULD have repeated
    repeats = np.where(test_data['sample_nature'] == 'flw', test_data['multiply'] + 1.0, 1.0)
    
    # Calculate the combined final weight mathematically
    final_weights = repeats / test_data['pkt_count']

    # Convert everything cleanly to numpy arrays
    y_true = np.asarray(test_labels, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_true_all = np.asarray(test_labels_all_clusters, dtype=np.int32)
    final_weights = np.asarray(final_weights, dtype=np.float32)
                           
    return y_true, y_pred, final_weights, y_true_all
    
def get_confusion_matrix(use_case, classes_filter, clusters_best_model_info, flow_counts_test_file_path,
                         flow_counts_train_file_path, train_data_dir_path, test_data_dir_path, models_dir=None):
    """
    Function to get the confusion matrix of the models of clusters
    """
    cluster_list = clusters_best_model_info['Class List'].tolist()
    all_classes_in_clusters = clusters_best_model_info['Class List'].sum()
    class_to_cluster_map = get_class_to_cluster_map(clusters_best_model_info)


    cm_matrix = pd.DataFrame()
    cm_matrix['Classes'] = all_classes_in_clusters
    cm_matrix_cluster = pd.DataFrame()

    for cluster_id in range(0, len(cluster_list)):
        npkts = int(clusters_best_model_info.loc[cluster_id]['N'])
        n_tree = int(clusters_best_model_info.loc[cluster_id]['Tree'])
        max_n_leaves = int(clusters_best_model_info.loc[cluster_id]['N_Leaves'])
        feat_names = clusters_best_model_info.loc[cluster_id]['Feature List']

        # Match cluster_analysis's encoding EXACTLY: sorted(set(Class List)) + 'Other'.
        # The saved models were trained with this integer<->class order, so we must
        # build the test labels AND decode the models' predictions with the same
        # order. (Also avoids mutating cluster_list[cluster_id] in place.)
        classes = sorted(set(cluster_list[cluster_id]))
        classes.append('Other')

        cluster_perf_dict = collections.defaultdict(int)
        #
        logging.getLogger(f'{use_case}').info(f'The model info: \n {npkts, n_tree, max_n_leaves, feat_names, classes}')
        #
        
        # Unpack the Zero-Memory optimized arrays
        y_true, y_pred, final_weights, y_true_all = analyze_model(use_case, classes_filter, npkts, n_tree, max_n_leaves, feat_names, classes, all_classes_in_clusters, flow_counts_test_file_path, flow_counts_train_file_path, train_data_dir_path, test_data_dir_path, cluster_id=cluster_id, models_dir=models_dir)
        
        # Generate DataFrame instantly with original size
        pred_df = pd.DataFrame({'True_Label_Cluster': y_true, 'Pred_Label_Cluster': y_pred,
                                'True_Label_All': y_true_all, 'Weight_per_packet': final_weights})

        # Precompute indices for classes to avoid repeated .index() calls
        class_indices = {cls: idx for idx, cls in enumerate(classes)}
        all_classes_in_clusters_indices = {cla: idx for idx, cla in enumerate(all_classes_in_clusters)}

        for _class in classes[:-1]:
            performance_vals = []
            # filter out already classes not in cluster
            classes_in_cluster_df = pred_df.loc[pred_df['Pred_Label_Cluster'] == classes.index(_class)]
            for cla in all_classes_in_clusters:
                if cla in classes:
                    cla_ind = class_indices[cla]
                    label_column = 'True_Label_Cluster'
                    cluster_of_cls = cluster_id
                else:
                    cla_ind = all_classes_in_clusters_indices[cla]
                    label_column = 'True_Label_All'
                    cluster_of_cls = class_to_cluster_map[cla]

                metric_val = classes_in_cluster_df.loc[
                    (pred_df[label_column] == cla_ind),
                    'Weight_per_packet'
                ].sum()
                performance_vals.append(metric_val)
                cluster_perf_dict[cluster_of_cls] +=  metric_val

            cm_matrix[_class] = performance_vals
        cm_matrix_cluster[str(cluster_id)] = cluster_perf_dict.values()

    return cm_matrix, cm_matrix_cluster


def normalize_confusion_matrix(cm_matrix_cluster):
    """
    Function to normalize the confusion matrix
    """
    return cm_matrix_cluster.div(cm_matrix_cluster.sum(axis=0), axis=1).mul(100)


if __name__ == '__main__':
    logging.basicConfig()
    # Read the data
    import os
    basepath = os.path.dirname(os.path.abspath(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(basepath, 'params.ini'))
    config['DEFAULT']['training_root'] = os.path.abspath(os.path.join(basepath, '..', '..'))
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    force_rewrite = config['DEFAULT'].get('force_rewrite', 'False') == 'True'
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)
    
    # Parameters necessary for the model sequencing
    flow_counts_train_file_path = config[use_case]['flow_counts_train_file_path'] 
    classes_filter = config[use_case]['classes_filter'] 
    classes_filter = ast.literal_eval(classes_filter)
    flow_counts_test_file_path = config[use_case]['flow_counts_test_file_path'] 
    best_models_path = config[use_case]['best_models_per_cluster_path']
    # Directory holding the trained cluster models (selected_model_cluster{id}.pkl)
    # saved by cluster_analysis. best_models_per_cluster_path points at
    # <results>/perf_results/cluster_info_df.csv, so <results> is two levels up
    # and the models live in the <results>/models subdirectory.
    models_dir_path = config[use_case].get(
        'models_dir_path', os.path.join(os.path.dirname(os.path.dirname(best_models_path)), 'models'))
    train_data_dir_path = config[use_case]['train_data_dir_path']
    test_data_dir_path = config[use_case]['test_data_dir_path']
    results_dir_path = config[use_case]['results_dir_path']
    
    # Get the cluster information which is obtained in the previous steps
    clusters_best_model_info = pd.read_csv(best_models_path,
                                           converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
    clusters_best_model_info = clusters_best_model_info.set_index('Cluster', drop=True)
    
    # Get the confusion matrix between the models of the corresponding clusters
    logger.info(f'The analysis starts...')
    try:
        if force_rewrite:
            # Ignore any cached confusion matrices and recompute from scratch.
            raise FileNotFoundError("force_rewrite=True: recomputing confusion matrices")
        cm_matrix = pd.read_csv(results_dir_path + '/' + use_case+'_confusion_matrix.csv')
        cm_matrix_cluster = pd.read_csv(results_dir_path + '/' + use_case + '_cm_matrix_cluster.csv')
    except FileNotFoundError:
        cm_matrix, cm_matrix_cluster = get_confusion_matrix(use_case, classes_filter, clusters_best_model_info,
                                                            flow_counts_test_file_path, flow_counts_train_file_path,
                                                            train_data_dir_path, test_data_dir_path,
                                                            models_dir=models_dir_path)
        cm_matrix.to_csv(results_dir_path + '/' + use_case+'_confusion_matrix.csv', index=False)
        cm_matrix_cluster.to_csv(results_dir_path + '/' + use_case + '_cm_matrix_cluster.csv', index=False)

    cm_matrix_cluster_normalized_df = normalize_confusion_matrix(cm_matrix_cluster)
    logger.info(f'Normalized confusion matrix is: ')
    logger.info(f'\n{cm_matrix_cluster_normalized_df}')
    
    cost_FP = cm_matrix_cluster_normalized_df.values
    cost_F1 = clusters_best_model_info['Macro_f1_FL'].values

    # Order the clusters based on the FP, and F1 scores
    blocks_seq = order_blocks_with_tsp(cost_FP, cost_F1)

    logger.info(f'The sequence of the blocks is: {blocks_seq}')

    import json
    sequence_file = results_dir_path + '/' + use_case + '_model_sequence.json'
    with open(sequence_file, 'w') as f:
        json.dump(blocks_seq, f)
    logger.info(f'Saved model sequence to: {sequence_file}')