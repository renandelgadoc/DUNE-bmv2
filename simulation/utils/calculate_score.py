import os
import pandas as pd
import numpy as np
import argparse
import sys
import pprint
from sklearn.metrics import classification_report, confusion_matrix

def normalize_ground_truth(ground_truth_df):
    ground_truth_df = ground_truth_df.drop(columns=['Unnamed: 0'], errors='ignore').copy()
    if 'packet_counts' not in ground_truth_df.columns and 'count' in ground_truth_df.columns:
        ground_truth_df['packet_counts'] = ground_truth_df['count']
    if 'type' in ground_truth_df.columns:
        ground_truth_df['type'] = ground_truth_df['type'].astype(str).str.lower()

    required_columns = {'Flow ID', 'type', 'packet_counts'}
    missing_columns = required_columns - set(ground_truth_df.columns)
    if missing_columns:
        raise ValueError(
            f"Ground truth file is missing required columns after normalization: {sorted(missing_columns)}"
        )

    # Collapse duplicate Flow IDs. The same 5-tuple (e.g. NetBIOS/mDNS
    # broadcasts) appears across many scenario capture files, so a raw Flow ID
    # is not unique in the ground truth. Left as-is, the results<->GT merge
    # below is many-to-many and every classified packet whose flow is duplicated
    # gets counted once per GT row -> inflated support and wrong metrics.
    # Duplicates are label-consistent (verified), so we keep one row per Flow ID
    # with the summed packet count.
    if ground_truth_df['Flow ID'].duplicated().any():
        ground_truth_df = (
            ground_truth_df
            .groupby('Flow ID', as_index=False)
            .agg(type=('type', 'first'), packet_counts=('packet_counts', 'sum'))
        )
    return ground_truth_df

def compute_classification_report(results_df, ground_truth_df, classes, is_default=True):
# 1. Force the ports and protocols to numeric values. Broken/missing entries become NaN.
    results_df['src_port'] = pd.to_numeric(results_df['src_port'], errors='coerce')
    results_df['dst_port'] = pd.to_numeric(results_df['dst_port'], errors='coerce')    
    results_df = results_df.dropna(subset=['src_port', 'dst_port']).copy()

    results_df['src_port'] = results_df['src_port'].astype(int)
    results_df['dst_port'] = results_df['dst_port'].astype(int)
    results_df['transport_proto'] = results_df['transport_proto'].astype(int)
    results_df['class'] = results_df['class'].astype(int)

    results_df['Flow ID'] = (
        results_df['src_ip'].astype(str) + ' ' +
        results_df['dst_ip'].astype(str) + ' ' +
        results_df['src_port'].astype(str) + ' ' +
        results_df['dst_port'].astype(str) + ' ' +
        results_df['transport_proto'].astype(str)
    )
    
    _results_dir = os.environ.get('DUNE_RESULTS_DIR', 'results')
    os.makedirs(_results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(_results_dir, "results_df.csv"), index=False)

    # Merge dataframes
    labeled_results_csv = pd.merge(results_df, ground_truth_df, on=['Flow ID'])

    if labeled_results_csv.empty:
        print("\n[ERROR] The merged dataset is empty!")
        print("No matching 'Flow ID' combinations were found between your results and the ground truth.")
        sys.exit(1)

    # Map class names to integer ids
    class_mapping = {cls: i + 1 for i, cls in enumerate(classes)}
    # Persist the class-name <-> integer-id mapping so the integer labels in the
    # report/confusion matrices are always interpretable after the fact. The ids
    # follow the same Class List order used at training (cluster_analysis) and in
    # the P4 deployment, so this file is the ground truth for "which id is which".
    import json as _json
    with open(os.path.join(_results_dir, "class_id_mapping.json"), "w") as _mf:
        _json.dump({str(cls): int(idx) for cls, idx in class_mapping.items()}, _mf, indent=2)
    labeled_results_csv = labeled_results_csv[labeled_results_csv['type'].isin(class_mapping)].copy()
    labeled_results_csv['ground_truth'] = labeled_results_csv['type'].map(class_mapping)
    
    # Adjust binary class mapping to multiclass mapping if necessary
    if labeled_results_csv['class'].max() <= 1 and len(classes) > 2:
        print("[INFO] Model outputs binary classes. Adjusting 'class' 0 to map to 'normal' (1)...")
        labeled_results_csv.loc[labeled_results_csv['class'] == 0, 'class'] = 1

    labeled_results_csv['packet_counts'] = pd.to_numeric(
        labeled_results_csv['packet_counts'], errors='coerce'
    )
    labeled_results_csv = labeled_results_csv[labeled_results_csv['packet_counts'] > 0].copy()

    if labeled_results_csv.empty:
        print("\n[ERROR] Dataset became empty after filtering out invalid packet counts.")
        sys.exit(1)

    # --- WEIGHTS DISABLED START ---
    # Your framework originally calculated packet balances here:
    # labeled_results_csv['weight'] = 1 / labeled_results_csv['packet_counts']
    # --- WEIGHTS DISABLED END ---
    
    labeled_results_csv['weight'] = 1 / labeled_results_csv['packet_counts']

    labeled_results_csv.to_csv(os.path.join(_results_dir, "labeled_results.csv"), index=False)

    # Keep unique pairs dynamically present for JSON dictionary output configuration
    unique_df = labeled_results_csv.drop_duplicates(subset=["type", "ground_truth"], keep="first")
    unique_df.to_csv(os.path.join(_results_dir, "unique_labels.csv"), index=False)

    # Mappings for the Multiclass Text Layout
    all_numeric_labels = list(class_mapping.values())
    all_class_names = list(class_mapping.keys())
    # In the default pipeline the confusion matrix includes an explicit
    # "unclassified" column for packets the pipeline never classified
    # (P4 UNKNOWN_CLASS = 0; real classes are 1..N), so they are counted/visible
    # instead of being silently dropped from the matrix. In OFSP/ensemble the
    # majority vote always emits a class, so that column would be empty and is
    # left out entirely, per the user's request for nothing unclassified in OFSP.
    # Note: the per-class F1/recall already treats an unclassified prediction as a
    # false negative for its true class, so this only affects the matrix display,
    # not the reported precision/recall/F1.
    if is_default:
        cm_labels = [0] + all_numeric_labels
        cm_names = ['unclassified'] + all_class_names
    else:
        cm_labels = all_numeric_labels
        cm_names = all_class_names

    # === UNWEIGHTED (packet-level) ===

    text_report = classification_report(
        labeled_results_csv['ground_truth'],
        labeled_results_csv['class'],
        labels=all_numeric_labels,
        target_names=all_class_names,
        zero_division=0,
    )

    dict_report = classification_report(
        labeled_results_csv['ground_truth'],
        labeled_results_csv['class'],
        labels=unique_df['ground_truth'],
        target_names=unique_df['type'],
        output_dict=True,
        zero_division=0,
    )

    if 'accuracy' in dict_report and isinstance(dict_report['accuracy'], (int, float)):
        acc_value = dict_report['accuracy']
        dict_report['accuracy'] = {
            'f1-score': acc_value,
            'precision': acc_value,
            'recall': acc_value,
            'support': dict_report.get('macro avg', {}).get('support', 0)
        }

    cm = confusion_matrix(
        labeled_results_csv['ground_truth'],
        labeled_results_csv['class'],
        labels=cm_labels,
    )

    cm_df = pd.DataFrame(cm, index=cm_names, columns=cm_names)
    truth_table_text = "\n" + "="*20 + " TRUTH TABLE (CONFUSION MATRIX) " + "="*20 + "\n"
    truth_table_text += "Rows = Real Class (Ground Truth)  |  Columns = Guessed Class (Predicted)\n\n"
    truth_table_text += cm_df.to_string() + "\n"

    # === WEIGHTED (flow-level, weight = 1/packet_counts) ===

    weighted_text_report = classification_report(
        labeled_results_csv['ground_truth'],
        labeled_results_csv['class'],
        labels=all_numeric_labels,
        target_names=all_class_names,
        sample_weight=labeled_results_csv['weight'],
        zero_division=0,
    )

    weighted_dict_report = classification_report(
        labeled_results_csv['ground_truth'],
        labeled_results_csv['class'],
        labels=unique_df['ground_truth'],
        target_names=unique_df['type'],
        sample_weight=labeled_results_csv['weight'],
        output_dict=True,
        zero_division=0,
    )

    if 'accuracy' in weighted_dict_report and isinstance(weighted_dict_report['accuracy'], (int, float)):
        acc_value = weighted_dict_report['accuracy']
        weighted_dict_report['accuracy'] = {
            'f1-score': acc_value,
            'precision': acc_value,
            'recall': acc_value,
            'support': weighted_dict_report.get('macro avg', {}).get('support', 0)
        }

    weighted_cm = confusion_matrix(
        labeled_results_csv['ground_truth'],
        labeled_results_csv['class'],
        labels=cm_labels,
        sample_weight=labeled_results_csv['weight'],
    )

    weighted_cm_df = pd.DataFrame(weighted_cm, index=cm_names, columns=cm_names)
    weighted_truth_table_text = "\n" + "="*20 + " WEIGHTED TRUTH TABLE (FLOW-LEVEL) " + "="*20 + "\n"
    weighted_truth_table_text += "Rows = Real Class (Ground Truth)  |  Columns = Guessed Class (Predicted)\n"
    weighted_truth_table_text += "Weight = 1/packet_counts (flow-level scoring)\n\n"
    weighted_truth_table_text += weighted_cm_df.to_string() + "\n"

    # === DEFAULT ONLY: multiclass reports that ALSO list 'unclassified' ===
    # No ground-truth sample is truly unclassified, so this class has support 0.
    # Listing it turns unclassified predictions into false positives, which pulls
    # the macro and micro averages down (micro collapses to the plain accuracy),
    # while the weighted average and the per-real-class metrics stay unchanged.
    # Left empty in OFSP/ensemble, where nothing unclassified is reported.
    report_with_unclassified = ""
    weighted_report_with_unclassified = ""
    if is_default:
        _uncls_labels = [0] + all_numeric_labels
        _uncls_names = ['unclassified'] + all_class_names
        report_with_unclassified = classification_report(
            labeled_results_csv['ground_truth'],
            labeled_results_csv['class'],
            labels=_uncls_labels,
            target_names=_uncls_names,
            zero_division=0,
        )
        weighted_report_with_unclassified = classification_report(
            labeled_results_csv['ground_truth'],
            labeled_results_csv['class'],
            labels=_uncls_labels,
            target_names=_uncls_names,
            sample_weight=labeled_results_csv['weight'],
            zero_division=0,
        )

    # === BINARY: normal (0) vs attack (1) ===
    # A packet the default pipeline left unclassified (P4 UNKNOWN_CLASS = 0) is NOT taken as an
    # attack: it is a third outcome, "unclassified" (2), that counts as an error for its true
    # class. So an unclassified attack is a false negative (it lowers the recall of attack) and an
    # unclassified normal is neither a true negative nor a false positive (it lowers the recall of
    # normal, i.e. the specificity). The report then has an "unclassified" row with support 0,
    # as in the multiclass report with unclassified: its F1 is 0 and it enters the macro average,
    # not the weighted one. In OFSP/ensemble the vote always gives a class, so there is no
    # unclassified outcome and the report is the plain normal x attack one.
    normal_id = class_mapping.get('normal', 1)
    binary_gt = (labeled_results_csv['ground_truth'] != normal_id).astype(int)
    binary_pred = np.where(labeled_results_csv['class'] == normal_id, 0, 1)
    if is_default:
        binary_pred = np.where(labeled_results_csv['class'] == 0, 2, binary_pred)
        bin_labels, bin_names = [0, 1, 2], ['normal', 'attack', 'unclassified']
    else:
        bin_labels, bin_names = [0, 1], ['normal', 'attack']

    binary_text_report = classification_report(
        binary_gt, binary_pred, labels=bin_labels,
        target_names=bin_names, zero_division=0,
    )

    binary_dict_report = classification_report(
        binary_gt, binary_pred, labels=bin_labels,
        target_names=bin_names, output_dict=True, zero_division=0,
    )

    # Rows are the real classes (normal, attack); the unclassified column only exists in default.
    binary_cm = confusion_matrix(binary_gt, binary_pred, labels=bin_labels)[:2]
    binary_cm_df = pd.DataFrame(binary_cm, index=['normal', 'attack'], columns=bin_names)
    binary_truth_table_text = "\n" + "="*20 + " BINARY TRUTH TABLE (CONFUSION MATRIX) " + "="*20 + "\n"
    binary_truth_table_text += "Rows = Real Class (Ground Truth)  |  Columns = Guessed Class (Predicted)\n\n"
    binary_truth_table_text += binary_cm_df.to_string() + "\n"

    # === BINARY WEIGHTED (flow-level) ===

    binary_weighted_text_report = classification_report(
        binary_gt, binary_pred, labels=bin_labels,
        target_names=bin_names,
        sample_weight=labeled_results_csv['weight'], zero_division=0,
    )

    binary_weighted_dict_report = classification_report(
        binary_gt, binary_pred, labels=bin_labels,
        target_names=bin_names,
        sample_weight=labeled_results_csv['weight'],
        output_dict=True, zero_division=0,
    )

    binary_weighted_cm = confusion_matrix(binary_gt, binary_pred, labels=bin_labels,
                                          sample_weight=labeled_results_csv['weight'])[:2]
    binary_weighted_cm_df = pd.DataFrame(binary_weighted_cm, index=['normal', 'attack'], columns=bin_names)
    binary_weighted_truth_table_text = "\n" + "="*20 + " BINARY WEIGHTED TRUTH TABLE (FLOW-LEVEL) " + "="*20 + "\n"
    binary_weighted_truth_table_text += "Rows = Real Class (Ground Truth)  |  Columns = Guessed Class (Predicted)\n\n"
    binary_weighted_truth_table_text += binary_weighted_cm_df.to_string() + "\n"

    return (text_report, dict_report, truth_table_text,
            binary_text_report, binary_dict_report, binary_truth_table_text,
            weighted_text_report, weighted_dict_report, weighted_truth_table_text,
            binary_weighted_text_report, binary_weighted_dict_report, binary_weighted_truth_table_text,
            report_with_unclassified, weighted_report_with_unclassified)


def calculate_score(c_report_dict):
    macro_f1 = c_report_dict['macro avg']['f1-score']
    weighted_f1 = c_report_dict['weighted avg']['f1-score']
    if 'micro avg' in c_report_dict:
        micro_f1 = c_report_dict['micro avg']['f1-score']
    elif isinstance(c_report_dict.get('accuracy'), dict):
        micro_f1 = c_report_dict['accuracy']['f1-score']
    else:
        micro_f1 = c_report_dict['accuracy']
    return macro_f1, weighted_f1, micro_f1

def build_class_order_from_cluster_info(cluster_info_path, sequence_file=None):
    """Derive the class-to-ID mapping from cluster_info_df.csv to match the P4 pipeline.
    If sequence_file is provided, use the model sequence order instead of CSV row order."""
    import json
    from ast import literal_eval
    cluster_info = pd.read_csv(cluster_info_path)

    if sequence_file:
        with open(sequence_file, 'r') as f:
            sequence = json.load(f)
    else:
        sequence = cluster_info["Cluster"].tolist() if "Cluster" in cluster_info.columns else list(range(len(cluster_info)))

    ordered_classes = []
    for cluster_id in sequence:
        row = cluster_info[cluster_info["Cluster"] == cluster_id].iloc[0] if "Cluster" in cluster_info.columns else cluster_info.iloc[cluster_id]
        class_list = row['Class List']
        if isinstance(class_list, str):
            class_list = literal_eval(class_list)
        # Use sorted(set(...)) per cluster to match the deterministic class order
        # the models are trained/saved with (cluster_analysis) and that the P4
        # pipeline's positional class ids therefore follow. Raw Class List order
        # would only accidentally agree when it is already sorted.
        for cls in sorted(set(class_list)):
            ordered_classes.append(cls)
    # Ensemble (OFSP): every cluster carries the FULL class list, so the
    # concatenation above repeats classes. When that happens, fall back to the
    # deterministic sorted(set(...)) order used at training/inference time
    # (clf.classes_), which is the order the pipeline's 1-based class ids follow.
    # In the default (disjoint) case there are no repeats and order is preserved.
    if len(ordered_classes) != len(set(ordered_classes)):
        # 'normal' forced to index 0 to mirror the OFSP model encoding
        # (cluster_analysis load_cluster_data), where it wins the majority-vote
        # tie-break. The rest stay sorted for determinism.
        _s = set(ordered_classes)
        ordered_classes = (['normal'] if 'normal' in _s else []) + sorted(c for c in _s if c != 'normal')
    return ordered_classes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--cluster-info", default=None,
                        help="Path to cluster_info_df.csv to derive class order from P4 pipeline")
    parser.add_argument("--sequence-file", default=None,
                        help="Path to model_sequence.json from model_sequencing")
    parser.add_argument("--partitioning-type", default="default",
                        help="Pipeline mode: 'default'/'spp' adds an extra multiclass report that "
                             "includes the 'unclassified' class; 'ofsp'/'ensemble' omits anything "
                             "unclassified (report and confusion-matrix column)")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.cluster_info:
        classes = build_class_order_from_cluster_info(args.cluster_info, args.sequence_file)
    else:
        classes = ['normal', 'ddos', 'ransomware', 'xss', 'scanning', 'injection', 'password']
    # src_ip/dst_ip read as 'category' so the 37M+ packet rows store each IP
    # string once (+ int codes) instead of one Python str per row. This only
    # changes storage, not values -> identical metrics, but cuts the peak RAM
    # of the load/merge enough to score the full pcap (OFSP) without OOM.
    results_df = pd.read_csv(args.results, index_col=None,
                             dtype={'src_ip': 'category', 'dst_ip': 'category'})
    ground_truth_df = normalize_ground_truth(pd.read_csv(args.ground_truth))

    is_default = args.partitioning_type not in ('ofsp', 'ensemble')
    (c_report_text, c_report_dict, truth_table_text,
     binary_text, binary_dict, binary_truth_table,
     w_report_text, w_report_dict, w_truth_table_text,
     bw_report_text, bw_report_dict, bw_truth_table_text,
     c_report_unclassified, w_report_unclassified) = compute_classification_report(
        results_df, ground_truth_df, classes, is_default)

    # Save the pretty multi-class text report table AND the Truth Table together
    report_dir = os.path.dirname(args.outfile) or "results"
    with open(os.path.join(report_dir, "classification-report.txt"), "w") as f:
        f.write("="*20 + " MULTICLASS CLASSIFICATION (PACKET-LEVEL) " + "="*20 + "\n\n")
        f.write(c_report_text)
        f.write("\n")
        f.write(truth_table_text)
        f.write("\n\n")
        f.write("="*20 + " MULTICLASS CLASSIFICATION (FLOW-LEVEL WEIGHTED) " + "="*20 + "\n\n")
        f.write(w_report_text)
        f.write("\n")
        f.write(w_truth_table_text)
        f.write("\n\n")
        # Default-only: the same multiclass reports, but now listing the
        # 'unclassified' class alongside the real ones for comparison.
        if c_report_unclassified:
            f.write("="*20 + " MULTICLASS WITH UNCLASSIFIED (PACKET-LEVEL) " + "="*20 + "\n\n")
            f.write(c_report_unclassified)
            f.write("\n\n")
            f.write("="*20 + " MULTICLASS WITH UNCLASSIFIED (FLOW-LEVEL WEIGHTED) " + "="*20 + "\n\n")
            f.write(w_report_unclassified)
            f.write("\n\n")
        f.write("="*20 + " BINARY CLASSIFICATION (PACKET-LEVEL) " + "="*20 + "\n\n")
        f.write(binary_text)
        f.write("\n")
        f.write(binary_truth_table)
        f.write("\n\n")
        f.write("="*20 + " BINARY CLASSIFICATION (FLOW-LEVEL WEIGHTED) " + "="*20 + "\n\n")
        f.write(bw_report_text)
        f.write("\n")
        f.write(bw_truth_table_text)

    # Print the truth tables to terminal console
    print(truth_table_text)
    print(binary_truth_table)

    # Format dictionary structure using pretty-print's precise single-quote formatting style
    pprint_compatible_str = pprint.pformat(c_report_dict, width=80, compact=False)
    macro_f1, weighted_f1, micro_f1 = calculate_score(c_report_dict)
    binary_macro_f1, binary_weighted_f1, binary_micro_f1 = calculate_score(binary_dict)
    w_macro_f1, w_weighted_f1, w_micro_f1 = calculate_score(w_report_dict)
    bw_macro_f1, bw_weighted_f1, bw_micro_f1 = calculate_score(bw_report_dict)

    # Write directly into downstream target output file for database processing
    with open(args.outfile, "w") as f:
        f.write(f"Collision count: {results_df['collision'].sum()}\n")
        f.write(pprint_compatible_str + "\n")
        f.write(f"The score information:   Macro={macro_f1} Weighted={weighted_f1} Micro={micro_f1}\n")
        f.write(f"\n--- Binary (normal vs attack) ---\n")
        f.write(pprint.pformat(binary_dict, width=80, compact=False) + "\n")
        f.write(f"The score information:   Macro={binary_macro_f1} Weighted={binary_weighted_f1} Micro={binary_micro_f1}\n")
        f.write(f"\n--- Flow-level weighted (multiclass) ---\n")
        f.write(pprint.pformat(w_report_dict, width=80, compact=False) + "\n")
        f.write(f"The score information:   Macro={w_macro_f1} Weighted={w_weighted_f1} Micro={w_micro_f1}\n")
        f.write(f"\n--- Flow-level weighted (binary) ---\n")
        f.write(pprint.pformat(bw_report_dict, width=80, compact=False) + "\n")
        f.write(f"The score information:   Macro={bw_macro_f1} Weighted={bw_weighted_f1} Micro={bw_micro_f1}")