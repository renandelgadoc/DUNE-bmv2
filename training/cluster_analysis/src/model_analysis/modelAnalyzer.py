import gc
import pickle
import os
import signal
import shutil

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from os import path
from abc import ABC, abstractmethod
from itertools import compress

import warnings

def assign_sample_nature(row):
    """Aux function to check the conditions and assign values"""
    if (row["Min Packet Length"] == -1 and
            row["Max Packet Length"] == -1 and
            row["Flow IAT Min"] == -1 and
            row["Flow IAT Max"] == -1):
        return "pkt"
    else:
        return "flw"


def extend_test_data_with_flow_level_results(y_test, y_pred, samples_nature_test, y_multiply_test, flow_pkt_cnt_test, flow_ids_test):
    expanded_y_test = []
    expanded_y_pred = []
    expanded_weights = []
    expanded_flow_IDs = []
    # If we have 10 packets in the flow and do inference at packet 3, we get multiply = 10 - 3 = 7.
    # We add 1 to include the n-th packet which is where we make inference.
    # This will mean that we want to attribute the score of the n-th packet to the remaining 7th packet but
    # since we want the n-th packet itself to be included in the list of classified packets
    # (this list was empty from the start), we add 1 to include it in the number of packets that take the
    # classification result.

    for true_label, pred_label, nature, multiplier, pkt_cnt, f_id in zip(y_test, y_pred, samples_nature_test,
                                                                         y_multiply_test,
                                                                         flow_pkt_cnt_test, flow_ids_test):
        if nature == 'flw':
            expanded_y_test.extend([true_label] * (multiplier + 1)) # +1 to account for the nth pkt (see above)
            expanded_y_pred.extend([pred_label] * (multiplier + 1))
            #
            expanded_weights.extend([1 / pkt_cnt] * (multiplier + 1))
            expanded_flow_IDs.extend([f_id] * (multiplier + 1))
        else:
            expanded_y_test.append(true_label)
            expanded_y_pred.append(pred_label)
            #
            expanded_weights.append(1 / pkt_cnt)
            expanded_flow_IDs.append(f_id)

    return expanded_y_test, expanded_y_pred, expanded_weights, expanded_flow_IDs



class ModelAnalyzer(ABC):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter, features_filter,
                 max_leaves_list=None, max_depth_list=None, n_trees_list=None, cluster_data_file_path=None,
                 logger=None):
        self.max_leaves_list = [None] if max_leaves_list is None else max_leaves_list
        self.max_depth_list = [None] if max_depth_list is None else max_depth_list
        self.n_trees_list = [None] if n_trees_list is None else n_trees_list

        if logger is None:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        if cluster_data_file_path is not None:
            self.cluster_data_file_path = cluster_data_file_path
            self.cluster_flag = True
        else:
            self.cluster_flag = False

        self.train_data_folder_path = train_data_folder_path
        self.test_data_folder_path = test_data_folder_path
        self.flow_counts_train_file_path = flow_counts_train_file_path
        self.flow_counts_test_file_path = flow_counts_test_file_path
        self.classes_filter = classes_filter
        self.features_filter = features_filter
        self.feature_list = None
        self.classes = None
        self.classes_df = None
        # When True, the train set is loaded in FULL (no sampling) using a
        # memory-safe streaming read (see _prepare_data). Used only for training
        # the final saved models, not for the grid search.
        self.load_full_data = False
        # 'spp'  -> disjoint classes per cluster, with an 'Other' catch-all
        # 'ofsp' -> every view covers all classes, no 'Other'. Set externally.
        self.partitioning_type = 'spp'
        warnings.filterwarnings("ignore")
        pd.options.mode.chained_assignment = None

    @abstractmethod
    def prepare_data(self, npkts, classes_filter=None):
        pass

    def prepare_cluster_data(self, npkts, classes_filter=None):
        train_data, test_data = self.prepare_data(npkts, self.classes_filter)

        # Update labels for 'Label_NEW' column
        train_data['Label_NEW'] = np.where((train_data['Label'].isin(self.classes)), train_data['Label'], 'Other')
        test_data['Label_NEW'] = np.where((test_data['Label'].isin(self.classes)), test_data['Label'], 'Other')

        # Debug info
        self.logger.debug(f"Train data count: {train_data['Label_NEW'].value_counts()}")
        self.logger.debug(f"Test data count: {test_data['Label_NEW'].value_counts()}")

        return train_data, test_data

    def _prepare_data(self, npkts, classes_filter, train_file, test_file, flow_counts_train, flow_counts_test):
        """
        Generalized method to handle repeated logic in prepare_data for UNSW and TON analyzers.
        """
        # Load Train and Test data (Parquet if available, else CSV with tight dtypes)
        _read_dtypes = {
            'tcp.flags.syn': np.int8, 'tcp.flags.ack': np.int8, 'tcp.flags.push': np.int8,
            'tcp.flags.fin': np.int8, 'tcp.flags.reset': np.int8, 'tcp.flags.ece': np.int8,
            'ip.proto': np.int8, 'ip.hdr_len': np.int8, 'ip.tos': np.int8, 'ip.ttl': np.uint8,
            'tcp.hdr_len': np.int8,
            'srcport': np.uint16, 'dstport': np.uint16,
            'ip.len': np.uint16, 'tcp.window_size_value': np.uint16, 'udp.length': np.uint16,
            'Min Packet Length': np.float32, 'Max Packet Length': np.float32,
            'Packet Length Mean': np.float32, 'Packet Length Total': np.float32,
            'UDP Len Min': np.float32, 'UDP Len Max': np.float32,
            'Flow IAT Min': np.float32, 'Flow IAT Max': np.float32,
            'Flow IAT Mean': np.float32, 'Time to Inference': np.float32,
            'SYN Flag Count': np.int8, 'ACK Flag Count': np.int8,
            'PSH Flag Count': np.int8, 'FIN Flag Count': np.int8,
            'RST Flag Count': np.int8, 'ECE Flag Count': np.int8,
        }
        train_pq = os.path.splitext(train_file)[0] + '.parquet'
        test_pq = os.path.splitext(test_file)[0] + '.parquet'
        self.logger.info(f'Loading train data: {train_pq if os.path.exists(train_pq) else train_file}')
        if os.path.exists(train_pq):
            import pyarrow.parquet as _pq
            _pf = _pq.ParquetFile(train_pq)
            _total = _pf.metadata.num_rows
            _target = self.GRID_SEARCH_SAMPLE_SIZE if self.GRID_SEARCH_SAMPLE_SIZE else _total
            _frac = min(1.0, (_target * 1.5) / _total)
            if self.load_full_data:
                # Final (selected) model: load the ENTIRE train set into memory in
                # one shot (no row-group loop, no sampling), with tight dtypes.
                # The parquet stores wide types (int64/double/bool), so the cast is
                # applied at the Arrow level BEFORE converting to pandas -- this
                # avoids ever materialising the full float64/int64 columns in
                # pandas, which is what blew up memory. 'Label' is left untouched.
                import pyarrow as _pa
                self.logger.info(f'Loading FULL train data ({_total:,} rows) into memory with tight dtypes (no sampling)...')
                _need = list({*self.features_filter, 'Label', 'Flow ID',
                              'Min Packet Length', 'Max Packet Length',
                              'Flow IAT Min', 'Flow IAT Max'})
                _arrow_t = {np.int8: _pa.int8(), np.uint8: _pa.uint8(),
                            np.uint16: _pa.uint16(), np.float32: _pa.float32()}
                _tbl = _pf.read(columns=_need)
                for _i, _name in enumerate(_tbl.schema.names):
                    if _name in _read_dtypes:
                        _tbl = _tbl.set_column(_i, _name, _tbl.column(_i).cast(_arrow_t[_read_dtypes[_name]]))
                train_data = _tbl.to_pandas(split_blocks=True, self_destruct=True)
                del _tbl
                gc.collect()
            elif _frac < 1.0:
                self.logger.info(f'Sampling {_frac:.1%} from each row group (target ~{int(_total * _frac):,} rows)')
                _parts = []
                for _i in range(_pf.metadata.num_row_groups):
                    _rg = _pf.read_row_group(_i).to_pandas()
                    _parts.append(_rg.sample(frac=_frac, random_state=42))
                    del _rg
                train_data = pd.concat(_parts, ignore_index=True)
                del _parts
                gc.collect()
            else:
                train_data = pd.read_parquet(train_pq)
        else:
            train_data = pd.read_csv(train_file, dtype=_read_dtypes)
        # train_data = pd.read_parquet(train_pq) if os.path.exists(train_pq) else pd.read_csv(train_file, dtype=_read_dtypes)
        self.logger.info(f'Train data loaded: {len(train_data)} rows')
        self.logger.info(f'Loading test data: {test_pq if os.path.exists(test_pq) else test_file}')
        test_data = pd.read_parquet(test_pq) if os.path.exists(test_pq) else pd.read_csv(test_file, dtype=_read_dtypes)
        self.logger.info(f'Test data loaded: {len(test_data)} rows')

        # Apply class filters if provided
        if classes_filter is not None:
            train_data = train_data.loc[train_data['Label'].isin(classes_filter)]
            test_data = test_data.loc[test_data['Label'].isin(classes_filter)]
            gc.collect()

        # Map flow packet counts for train and test sets.
        train_data["pkt_count"] = train_data["Flow ID"].map(flow_counts_train)
        # For UNSW this is fine because no multiple files exist, i.e., the test data comes from a single file.
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_counts_test)

        # Clean data (shuffle skipped — RF with random_state is order-invariant)
        train_data = train_data.dropna(subset=['srcport', 'dstport'])
        gc.collect()
        test_data = test_data.dropna(subset=['srcport', 'dstport'])
        gc.collect()

        # Assign 'sample_nature' vectorized (avoids slow row-by-row apply)
        _pkt_mask_train = ((train_data["Min Packet Length"] == -1) &
                           (train_data["Max Packet Length"] == -1) &
                           (train_data["Flow IAT Min"] == -1) &
                           (train_data["Flow IAT Max"] == -1))
        train_data['sample_nature'] = np.where(_pkt_mask_train, 'pkt', 'flw')

        _pkt_mask_test = ((test_data["Min Packet Length"] == -1) &
                          (test_data["Max Packet Length"] == -1) &
                          (test_data["Flow IAT Min"] == -1) &
                          (test_data["Flow IAT Max"] == -1))
        test_data['sample_nature'] = np.where(_pkt_mask_test, 'pkt', 'flw')

        # Assign 'multiply' column based on the conditions.
        test_data['multiply'] = np.where(test_data['sample_nature']=='pkt', 1, test_data['pkt_count'] - npkts)

        train_data['weight'] = np.where(train_data['sample_nature'] == 'flw',
                                        (train_data['pkt_count'] - npkts + 1) / train_data['pkt_count'],
                                        1 / train_data['pkt_count'])
        return train_data, test_data


    def analyze_models(self, train_data, test_data, filename, grid_search=False):

        # open file to save output of analysis
        root = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        tmp_filename = f'{root}_tmp{extension}'
        if os.path.isfile(tmp_filename):
            self.logger.info(f"Deleting existing temp file: {tmp_filename}")
            os.remove(tmp_filename)

        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        def signal_handler(sig, frame):
            self.logger.info('You pressed Ctrl+C! Deleting temporary files...')
            os.remove(tmp_filename)
            raise KeyboardInterrupt()

        if grid_search:
            self.write_grid_search(signal_handler, tmp_filename, train_data, test_data)
        else:
            self.write_simple_analysis(signal_handler, tmp_filename, train_data, test_data)

        shutil.move(tmp_filename, filename)
        self.logger.info(f'Finished model analysis. Saved results to: {filename}')
        signal.signal(signal.SIGINT, original_sigint_handler)
        return []

    def write_simple_analysis(self, signal_handler, tmp_filename, train_data, test_data):

        # Get Variables and Labels
        if self.cluster_flag:
            label_column_name = 'Label_NEW'
        else:
            label_column_name = 'Label'
        y_multiply_test = test_data['multiply']
        flow_pkt_cnt_test = test_data['pkt_count'].to_list()
        flow_ids_test = test_data['Flow ID'].to_list()
        x_train, y_train, samples_nature_train = self.get_x_y_flow(train_data, label_column_name)
        x_test, y_test, samples_nature_test = self.get_x_y_flow(test_data, label_column_name)
        test_label_names, test_indices = self.get_test_labels(test_data, label_column_name)
        weight_of_samples = list(train_data['weight'])
        self.logger.debug(f'Num Labels: {len(test_label_names)}')

        with open(tmp_filename, "w") as res_file:
            self.logger.info(f'Writing grid search results to: {tmp_filename}')
            print('depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;num_samples;'
                  'Macro_F1_PL;Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
                  file=res_file)
            # register signal handler to delete file if code is not completed
            signal.signal(signal.SIGINT, signal_handler)
            n_tree =1
            if self.feature_list is not None:
                feats = self.feature_list
            else:
                feats = x_train.columns.values.tolist()
            for leaf in self.max_leaves_list:
                # Prepare a model for the given (depth, n_tree, feat)
                model = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=leaf, n_jobs=8, random_state=42,
                                               bootstrap=False)
                # Train (fit) the model with the data
                model.fit(x_train[feats], y_train, sample_weight=weight_of_samples)
                # Infer (predict) the labels
                y_pred = model.predict(x_test[feats]).tolist()

                # (expanded_y_test,
                #  expanded_y_pred,
                #  expanded_weights,
                #  expanded_flow_IDs) = extend_test_data_with_flow_level_results(y_test,
                #                                                                y_pred,
                #                                                                samples_nature_test,
                #                                                                y_multiply_test,
                #                                                                flow_pkt_cnt_test,
                #                                                                flow_ids_test)
                
                # num_samples = len(expanded_y_test)

                # FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_label_names,
                #                                         target_names=test_label_names, output_dict=True,
                #                                         sample_weight=expanded_weights)

                # macro_f1_FL = FL_class_report['macro avg']['f1-score']
                # weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                # micro_f1_FL = FL_class_report['accuracy']

                # PL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_label_names,
                #                                         target_names=test_label_names, output_dict=True)

                # macro_f1_PL = PL_class_report['macro avg']['f1-score']
                # weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                # micro_f1_PL = PL_class_report['accuracy']

                # depth = [estimator.tree_.max_depth for estimator in model.estimators_]
                # print(str(depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(leaf) + ";" +
                #       str(macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" + str(feats) + ';' +
                #       str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(weighted_f1_PL) + ';' +
                #       str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                #       file=res_file)

                # 1. Calculate the weights inline using your already existing variables
                # (This completely replaces the need for the extension function)
                pl_weights = np.where(np.asarray(samples_nature_test) == 'flw', 
                                      np.asarray(y_multiply_test) + 1.0, 1.0)
                fl_weights = pl_weights / np.asarray(flow_pkt_cnt_test, dtype=np.float32)
                
                # 2. The mathematical equivalent of len(expanded_y_test)
                num_samples = int(np.sum(pl_weights))

                # 3. Flow Level (FL) Report 
                # CHANGE: labels=test_label_names is now labels=test_indices
                FL_class_report = classification_report(y_test, y_pred, 
                                                        labels=test_indices,          # <--- Fixed line
                                                        target_names=test_label_names, 
                                                        output_dict=True,
                                                        sample_weight=fl_weights)

                macro_f1_FL = FL_class_report['macro avg']['f1-score']
                weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                # If the accuracy and micro avg results are equal, then micro is omitted
                if 'micro avg' in FL_class_report:
                    micro_f1_FL = FL_class_report['micro avg']['f1-score']
                else:
                    micro_f1_FL = FL_class_report['accuracy']

                # 4. Packet Level (PL) Report 
                # CHANGE: labels=test_label_names is now labels=test_indices
                PL_class_report = classification_report(y_test, y_pred, 
                                                        labels=test_indices,          # <--- Fixed line
                                                        target_names=test_label_names, 
                                                        output_dict=True,
                                                        sample_weight=pl_weights)

                macro_f1_PL = PL_class_report['macro avg']['f1-score']
                weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                # If the accuracy and micro avg results are equal, then micro is omitted
                if 'micro avg' in PL_class_report:
                    micro_f1_PL = PL_class_report['micro avg']['f1-score']
                else:
                    micro_f1_PL = PL_class_report['accuracy']

                # 5. Print results to file
                depth = [int(estimator.tree_.max_depth) for estimator in model.estimators_]
                print(str(depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(leaf) + ";" +
                      str(macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" + str(feats) + ';' +
                      str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(weighted_f1_PL) + ';' +
                      str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                      file=res_file)


    def write_grid_search(self, signal_handler, tmp_filename, train_data, test_data):

        # Get Variables and Labels
        y_multiply_test = test_data['multiply']
        flow_pkt_cnt_test = test_data['pkt_count'].to_list()
        flow_ids_test = test_data['Flow ID'].to_list()
        if self.cluster_flag:
            label_column_name = 'Label_NEW'
        else:
            label_column_name = 'Label'
        x_train, y_train, samples_nature_train = self.get_x_y_flow(train_data, label_column_name)
        x_test, y_test, samples_nature_test = self.get_x_y_flow(test_data, label_column_name)
        test_label_names, test_indices = self.get_test_labels(test_data, label_column_name)
        weight_of_samples = list(train_data['weight'])
        self.logger.debug(f'Num Labels: {len(test_label_names)}')

        # ToDo: filter out results with 2 or 4 trees
        with open(tmp_filename, "w") as res_file:
            self.logger.info(f'Writing grid search results to: {tmp_filename}')
            print('depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;num_samples;'
                  'Macro_F1_PL;Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
                  file=res_file)
            # register signal handler to delete file if code is not completed
            signal.signal(signal.SIGINT, signal_handler)
            # FOR EACH (n_tree, depth, leaf, feat)
            _total_combos = len(self.n_trees_list) * len(self.max_depth_list) * len(self.max_leaves_list)
            _combo_idx = 0
            for n_tree in self.n_trees_list:
                for depth in self.max_depth_list:
                    for leaf in self.max_leaves_list:
                        _combo_idx += 1
                        self.logger.info(f'[Grid {_combo_idx}/{_total_combos}] n_tree={n_tree}, depth={depth}, leaf={leaf} — fitting...')
                        # get feature orders to use
                        if self.feature_list is not None:
                            m_feats = [self.feature_list]
                        else:
                            m_feats = [x_train.columns.values.tolist()]
                        for feats in m_feats:
                            # ToDo: extract method analyse_grid_point to share code with write_simple_analysis
                            # Prepare a model for the given (depth, n_tree, feat, leaves)
                            model = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=leaf, max_depth=depth,
                                                           n_jobs=4, random_state=42, bootstrap=False)
                            # Train (fit) the model with the data
                            model.fit(x_train[feats], y_train, sample_weight=weight_of_samples)
                            self.logger.info(f'[Grid {_combo_idx}/{_total_combos}] n_tree={n_tree} — predict + scoring...')
                            # Infer (predict) the labels
                            y_pred = model.predict(x_test[feats]).tolist()

                            # # The result of the flow level inference must be applied to the consecutive packets.
                            # # ToDo: use true or test consistently to avoid confusions.
                            # #  Be consistent with the naming in the paper.
                            # (expanded_y_test,
                            #  expanded_y_pred,
                            #  expanded_weights,
                            #  expanded_flow_IDs) = extend_test_data_with_flow_level_results(y_test,
                            #                                                                y_pred,
                            #                                                                samples_nature_test,
                            #                                                                y_multiply_test,
                            #                                                                flow_pkt_cnt_test,
                            #                                                                flow_ids_test)
                            # num_samples = len(expanded_y_test)

                            # FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_indices,
                            #                                         target_names=test_label_names, output_dict=True,
                            #                                         sample_weight=expanded_weights)

                            # macro_f1_FL = FL_class_report['macro avg']['f1-score']
                            # weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                            # # If the accuracy and micro avg results are equal, then micro is omitted
                            # if 'micro avg' in FL_class_report:
                            #     micro_f1_FL = FL_class_report['micro avg']['f1-score']
                            # else:
                            #     micro_f1_FL = FL_class_report['accuracy']

                            # PL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_indices,
                            #                                         target_names=test_label_names, output_dict=True)

                            # macro_f1_PL = PL_class_report['macro avg']['f1-score']
                            # weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                            # # If the accuracy and micro avg results are equal, then micro is omitted
                            # if 'micro avg' in PL_class_report:
                            #     micro_f1_PL = PL_class_report['micro avg']['f1-score']
                            # else:
                            #     micro_f1_PL = PL_class_report['accuracy']

                            # selected_depth = [estimator.tree_.max_depth for estimator in model.estimators_]

                            # # ToDo: check if the leaves given by the below method is the intended one.
                            # selected_leafs = [estimator.tree_.n_leaves for estimator in model.estimators_]
                            # # The metric on which we select the best model is then the macro_f1_FL
                            # # ToDo: evaluate model while searching, and keep in memory the best found so far.
                            # #  Save the others as CSV files just in case.
                            # print(str(selected_depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(selected_leafs) + ";" +
                            #       str(macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" +
                            #       str(feats) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(weighted_f1_PL) +
                            #       ';' + str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                            #       file=res_file)

                            # 1. Calculate the weights inline using your already existing variables
                            # (This completely replaces the need for extend_test_data_with_flow_level_results)
                            pl_weights = np.where(np.asarray(samples_nature_test) == 'flw', 
                                                  np.asarray(y_multiply_test) + 1.0, 1.0)
                            fl_weights = pl_weights / np.asarray(flow_pkt_cnt_test, dtype=np.float32)
                            
                            # 2. Mathematical equivalent of len(expanded_y_test)
                            num_samples = int(np.sum(pl_weights))

                            # 3. Flow Level (FL) Report -> Uses y_test, y_pred, and fl_weights
                            FL_class_report = classification_report(y_test, y_pred, labels=test_indices,
                                                                    target_names=test_label_names, output_dict=True,
                                                                    sample_weight=fl_weights)

                            macro_f1_FL = FL_class_report['macro avg']['f1-score']
                            weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                            # If the accuracy and micro avg results are equal, then micro is omitted
                            if 'micro avg' in FL_class_report:
                                micro_f1_FL = FL_class_report['micro avg']['f1-score']
                            else:
                                micro_f1_FL = FL_class_report['accuracy']

                            # 4. Packet Level (PL) Report -> Uses y_test, y_pred, and pl_weights
                            PL_class_report = classification_report(y_test, y_pred, labels=test_indices,
                                                                    target_names=test_label_names, output_dict=True,
                                                                    sample_weight=pl_weights)

                            macro_f1_PL = PL_class_report['macro avg']['f1-score']
                            weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                            # If the accuracy and micro avg results are equal, then micro is omitted
                            if 'micro avg' in PL_class_report:
                                micro_f1_PL = PL_class_report['micro avg']['f1-score']
                            else:
                                micro_f1_PL = PL_class_report['accuracy']

                            selected_depth = [int(estimator.tree_.max_depth) for estimator in model.estimators_]

                            # ToDo: check if the leaves given by the below method is the intended one.
                            selected_leafs = [int(estimator.tree_.n_leaves) for estimator in model.estimators_]
                            
                            # The metric on which we select the best model is then the macro_f1_FL
                            # ToDo: evaluate model while searching, and keep in memory the best found so far.
                            #  Save the others as CSV files just in case.
                            print(str(selected_depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(selected_leafs) + ";" +
                                  str(macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" +
                                  str(feats) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(weighted_f1_PL) +
                                  ';' + str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                                  file=res_file)
                            self.logger.info(f'[Grid {_combo_idx}/{_total_combos}] n_tree={n_tree} — macro_f1_FL={macro_f1_FL:.4f}, weighted_f1_FL={weighted_f1_FL:.4f} — done')

    GRID_SEARCH_SAMPLE_SIZE = 5_000_000

    def analyze_model_n_packets(self, npkts, outfile, force=False, grid_search=False):
        """Function for grid search on hyperparameters, features and models"""
        if not force:
            if path.isfile(outfile):
                self.logger.info(
                    f"File {outfile} is present. To overwrite existing files pass force=True when running the"
                    f" analysis")
                return
        if self.cluster_flag:
            train_data, test_data = self.prepare_cluster_data(npkts, self.classes_filter)
        else:
            train_data, test_data = self.prepare_data(npkts, self.classes_filter)

        if self.GRID_SEARCH_SAMPLE_SIZE and len(train_data) > self.GRID_SEARCH_SAMPLE_SIZE:
            self.logger.info(f'Subsampling train data for grid search: {len(train_data)} -> {self.GRID_SEARCH_SAMPLE_SIZE}')
            train_data = train_data.sample(n=self.GRID_SEARCH_SAMPLE_SIZE, random_state=42)
            gc.collect()

        self.analyze_models(train_data, test_data, outfile, grid_search)
        self.logger.debug(f'Analysis completed. Check output file: {outfile}')

    def load_cluster_data(self, cluster_data_series):
        if self.partitioning_type == 'ofsp':
            # OFSP: each view already covers ALL classes (only the feature
            # subset differs), so there is no 'Other' catch-all. Use a
            # deterministic, reproducible order shared across every view so the
            # integer<->class mapping is identical for all of them. 'normal' is
            # forced to index 0 so it wins the ensemble majority-vote tie-break
            # (ComputeMajority favours the lowest index), i.e. ambiguous packets
            # default to benign. The rest stay sorted for determinism. Scoring
            # mirrors this in calculate_score.build_class_order_from_cluster_info.
            _s = set(cluster_data_series['Class List'])
            classes = (['normal'] if 'normal' in _s else []) + sorted(c for c in _s if c != 'normal')
            print(f"classes in cluter: {classes}")
        else:
            # Preserve Class List order (deterministic) so the integer<->class
            # encoding is reproducible and matches model_sequencing and the
            # simulation's calculate_score (build_class_order_from_cluster_info),
            # which both use Class List order. Previously list(set(...)) was
            # hash-randomized (PYTHONHASHSEED) -> non-deterministic, unknowable
            # mapping and silent intra-cluster label swaps at scoring time.
            classes = sorted(set(cluster_data_series['Class List']))
            classes.append('Other')
            print(f"classes in cluter: {classes}")
        self.logger.debug(f'Cluster ID: {cluster_data_series["Cluster"]}', classes)

        classes_df = pd.DataFrame(classes, columns=['class'])
        classes_df = classes_df.reset_index()

        feature_list = cluster_data_series['Feature List']
        self.logger.debug(f'Cluster ID: {cluster_data_series["Cluster"]}', feature_list)
        self.feature_list = feature_list
        self.classes = classes
        self.classes_df = classes_df

    def get_score_per_class(self, cl_report_FL):
        '''
        Get f1 score per class in the model
        Arguments:
        cl_report_FL - Classification report of the selected model (the score in Flow-level)
        Return:
        score_per_class_df - Dataframe including the score and number of samples per class
        '''
        valid_classes = [c_name for c_name in cl_report_FL.keys() if c_name in self.classes]
        score_per_class_df = pd.DataFrame.from_records(
            [(c_name, cl_report_FL[c_name]['f1-score'], cl_report_FL[c_name]['support']) for c_name in valid_classes],
            columns=['class', 'f1_score', 'support']
        )
        score_per_class_df['f1_score'] *= 100
        return score_per_class_df.sort_values(by='f1_score', ascending=False)

    def get_test_labels(self, test_data, label_column_name='Label'):
        array_of_indices = []
        unique_labels = test_data[label_column_name].unique()
        for lab in unique_labels:
            index = self.classes_df[self.classes_df['class'] == lab].index.values[0]
            array_of_indices.append(index)
        return unique_labels, array_of_indices

    def get_x_y_flow(self, dataset, label_column_name='Label'):
        # ToDo: selecting the subset of wanted features has nothing to do with the below code.
        x = dataset[self.features_filter]
        # gets the corresponding index for each label. Maybe a scikit requirement?
        y = dataset[label_column_name].replace(self.classes, range(len(self.classes))).values.tolist()
        sample_nature = dataset['sample_nature']
        return x, y, sample_nature

    def generate_model(self, model_info):
        '''
        Generate a Random Forest model with the given hyper-parameters
        Arguments:
        model_info - A dict including the hyperparameters of the model to generate
        Return:
        model - Random Forest model generated
        FL_class_report - Classification report of the model
        '''
        if self.cluster_flag:
            train_data, test_data = self.prepare_cluster_data(model_info['npkts'], self.classes_filter)
            label_column_name = 'Label_NEW'
        else:
            train_data, test_data = self.prepare_data(model_info['npkts'], self.classes_filter)
            label_column_name = 'Label'
        # Get Variables and Labels
        y_multiply_test = test_data['multiply']
        flow_pkt_cnt_test = test_data['pkt_count'].to_list()
        flow_ids_test = test_data['Flow ID'].to_list()
        x_train, y_train, samples_nature_train = self.get_x_y_flow(train_data, label_column_name)
        x_test, y_test, samples_nature_test = self.get_x_y_flow(test_data, label_column_name)
        test_label_names, test_indices = self.get_test_labels(test_data, label_column_name)
        weight_of_samples = list(train_data['weight'])
        max_depth = model_info.get('depth', None)
        max_leaf_nodes = model_info.get('max_leaf_nodes', None)
        model = RandomForestClassifier(n_estimators=model_info['tree'], max_depth=max_depth,
                                       max_leaf_nodes=max_leaf_nodes, n_jobs=4,
                                       random_state=42, bootstrap=False)

        # Train (fit) the model with the data
        model.fit(x_train[model_info['feats']], y_train, sample_weight=weight_of_samples)
        # Infer (predict) the labels
        y_pred = model.predict(x_test[model_info['feats']]).tolist()
        # The result of the flow level inference must be applied to the consecutive packets.
        # (expanded_y_test,
        #  expanded_y_pred,
        #  expanded_weights,
        #  expanded_flow_IDs) = extend_test_data_with_flow_level_results(y_test,
        #                                                                y_pred,
        #                                                                samples_nature_test,
        #                                                                y_multiply_test,
        #                                                                flow_pkt_cnt_test,
        #                                                                flow_ids_test)

        # FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_indices,
        #                                         target_names=test_label_names, output_dict=True,
        #                                         sample_weight=expanded_weights)

        pl_weights = np.where(np.asarray(samples_nature_test) == 'flw', 
                                np.asarray(y_multiply_test) + 1.0, 1.0)
        fl_weights = pl_weights / np.asarray(flow_pkt_cnt_test, dtype=np.float32)
        
        # 2. The mathematical equivalent of len(expanded_y_test)
        num_samples = int(np.sum(pl_weights))

        # 3. Flow Level (FL) Report 
        # CHANGE: labels=test_label_names is now labels=test_indices
        FL_class_report = classification_report(y_test, y_pred, 
                                                labels=test_indices,          # <--- Fixed line
                                                target_names=test_label_names, 
                                                output_dict=True,
                                                sample_weight=fl_weights)

        return model, FL_class_report, test_label_names



class UNSWModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter, features_filter, cluster_data_file_path=None, logger=None,
                 max_leaves_list=None, max_depth_list=None, n_trees_list=None
                 ):
        super().__init__(train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                         flow_counts_test_file_path, classes_filter, features_filter,
                         cluster_data_file_path=cluster_data_file_path, logger=logger,
                         max_leaves_list=max_leaves_list, max_depth_list=max_depth_list, n_trees_list=n_trees_list)

    def prepare_data(self, npkts, classes_filter=None):
        # Load train and test flow count files
        usecols = ['Flow ID', 'count', 'File']
        flow_counts_train = pd.read_csv(self.flow_counts_train_file_path, usecols=usecols)
        flow_counts_test = pd.read_csv(self.flow_counts_test_file_path)

        # Prepare flow packet counts mapping dictionaries
        flow_count_dict_test = flow_counts_test.set_index("Flow ID")["count"].to_dict()
        flow_counts_train['Flow ID'] = flow_counts_train['Flow ID'] + ' ' + flow_counts_train['File']
        flow_counts_train = flow_counts_train.drop_duplicates(subset=['Flow ID'], keep='first')
        flow_count_dict_train = flow_counts_train.set_index("Flow ID")["count"].to_dict()

        # Prepare train and test file paths
        train_file = f"{self.train_data_folder_path}/train_data_{npkts}.csv"
        test_file = f"{self.test_data_folder_path}/16-10-05.pcap.txt_{npkts}_pkts.csv"

        # Call base method to perform common steps
        train_data, test_data = self._prepare_data(npkts, classes_filter, train_file, test_file, flow_count_dict_train,
                                                   flow_count_dict_test)
        return train_data, test_data


class TONModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter, features_filter, cluster_data_file_path=None, logger=None,
                 max_leaves_list=None, max_depth_list=None, n_trees_list=None
                 ):
        super().__init__(train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                         flow_counts_test_file_path, classes_filter, features_filter,
                         cluster_data_file_path=cluster_data_file_path, logger=logger,
                         max_leaves_list=max_leaves_list, max_depth_list=max_depth_list, n_trees_list=n_trees_list)

    def prepare_data(self, npkts, classes_filter=None):
        # Load train and test flow count files
        flow_counts_train = pd.read_csv(self.flow_counts_train_file_path)
        flow_counts_test = pd.read_csv(self.flow_counts_test_file_path)

        # Prepare flow packet counts mapping dictionaries
        flow_count_dict_train = flow_counts_train.set_index("Flow ID")["count"].to_dict()
        flow_count_dict_test = flow_counts_test.set_index("Flow ID")["count"].to_dict()

        # Prepare train and test file paths
        train_file = f"{self.train_data_folder_path}/train_{npkts}_pkts.csv"
        test_file = f"{self.test_data_folder_path}/test_{npkts}_pkts.csv"

        # Call base method to perform common steps
        train_data, test_data = self._prepare_data(npkts, classes_filter, train_file, test_file,
                                                   flow_count_dict_train, flow_count_dict_test)
        return train_data, test_data


def get_feature_importance_sets(n_tree, x_train, y_train, weight_of_samples, max_leaf=None, max_depth=None):
    """
        Generate feature importance sets in descending order using a trained Random Forest model.

        This function trains a RandomForestClassifier with the provided data and parameters,
        extracts the feature importances, and returns the feature names sorted by their
        importance in descending order. It produces cumulative subsets of feature names
        such that each set includes the most important features up to that rank.

        Arguments:
        n_tree: int
            Number of trees in the RandomForestClassifier.
        x_train: np.ndarray
            Training features, where each row represents a sample and each column a feature.
        y_train: np.ndarray
            Training labels corresponding to each sample in the training data.
        weight_of_samples: np.ndarray
            Sample weights to account for unequal importance or representativity of samples.
        max_leaf: int
            Maximum number of leaf nodes for each tree in the RandomForestClassifier.
        max_depth: int
            Maximum depth for each tree in the RandomForestClassifier.

        Returns:
        List[List[str]]
            A list of lists of strings, where each nested list contains a cumulative subset of feature names sorted by importance.
    """
    rf_opt = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_leaf, random_state=42, bootstrap=False,
                                    n_jobs=4, max_depth=max_depth)
    rf_opt.fit(x_train, y_train, sample_weight=weight_of_samples)

    # Get the indices that would sort the feature_importances_ array in descending order
    sorted_indices = np.argsort(rf_opt.feature_importances_)[::-1]

    # Sort both arrays using the sorted indices
    sorted_feature_importances = rf_opt.feature_importances_[sorted_indices]
    sorted_feature_names = rf_opt.feature_names_in_[sorted_indices].tolist()

    return [sorted_feature_names[:i] for i in range(1, len(sorted_feature_names) + 1)]


def get_pkt_class_report(y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    return _compute_scores_by_nature("pkt", y_pred, y_test, sample_nature, unique_labels, array_of_indices)


def get_flow_class_report(y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    return _compute_scores_by_nature("pkt", y_pred, y_test, sample_nature, unique_labels, array_of_indices)


def _compute_scores_by_nature(target_nature, y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    is_target = [x==target_nature for x in sample_nature]
    y_test = list(compress(y_test, is_target))
    y_pred = list(compress(y_pred, is_target))
    return classification_report(y_test, y_pred, labels=unique_labels,
                                                  target_names=array_of_indices, output_dict=True)

