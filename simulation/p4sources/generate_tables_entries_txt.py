import numpy as np
import pandas as pd
import warnings
from statistics import mode
import argparse
import itertools

pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")

import logging
# Formatter
FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

## definition of useful functions
## gets all splits and conditions
def get_splits(forest, feature_names):
    data = []
    # generate dataframe with all thresholds and features
    for t in range(len(forest.estimators_)):
        clf = forest[t]
        n_nodes = clf.tree_.node_count
        features = [feature_names[i] if i >= 0 else None for i in clf.tree_.feature]
        for i in range(0, n_nodes):
            node_id = i
            left_child_id = clf.tree_.children_left[i]
            right_child_id = clf.tree_.children_right[i]
            threshold = clf.tree_.threshold[i]
            feature = features[i]
            if threshold != -2.0:
                data.append(
                    [t, node_id, left_child_id, right_child_id, threshold, feature]
                )
    data = pd.DataFrame(data)
    data.columns = ["Tree", "NodeID", "LeftID", "RightID", "Threshold", "Feature"]
    return data


## gets the feature table of each feature from the splits
def get_feature_table(splits_data, feature_name):
    feature_data = splits_data[splits_data["Feature"] == feature_name]
    feature_data = feature_data.sort_values(by="Threshold")
    feature_data = feature_data.reset_index(drop=True)
    ##
    # feature_data["Threshold"] = (feature_data["Threshold"]).astype(int)
    # feature_data["Threshold"] = feature_data["Threshold"].astype(int)
    # The tree goes left when value <= threshold. The values in the switch are
    # integers, and for an integer value "value <= t" is the same as
    # "value <= floor(t)". astype(int) cuts toward zero instead, so t = -0.5 became 0:
    # the split that separates -1 (flow features before the inference point) from the
    # real counts (0, 1, ...) moved, and a real count of 0 went to the -1 side.
    # np.floor keeps it at -1. Positive thresholds do not change (floor(0.5) = 0).
    feature_data["Threshold"] = np.floor(feature_data["Threshold"]).astype(int)
    ##
    code_table = pd.DataFrame()
    code_table["Threshold"] = feature_data["Threshold"]
    # print(feature_data)
    if len(code_table) == 0:
        # Feature has no splits in any tree — return a single wildcard entry
        code_table = pd.DataFrame({"Threshold": [0]})
        return code_table
    # create a column for each split in each tree
    for tree_id, node in zip(list(feature_data["Tree"]), list(feature_data["NodeID"])):
        colname = "s" + str(tree_id) + "_" + str(node)
        code_table[colname] = np.where(
            (
                code_table["Threshold"]
                <= feature_data[
                    (feature_data["NodeID"] == node) & (feature_data["Tree"] == tree_id)
                ]["Threshold"].values[0]
            ),
            0,
            1,
        )
    # add a row to represent the values above the largest threshold
    temp = [max(code_table["Threshold"]) + 1]
    temp.extend(list([1] * (len(code_table.columns) - 1)))
    code_table.loc[len(code_table)] = temp
    code_table = code_table.drop_duplicates(subset=["Threshold"])
    # A row with a negative threshold is the range of values below 0, which a switch
    # field (unsigned) never holds; its range would be invalid ("0,-1"). The row is
    # dropped; its split columns stay, so every value >= 0 is coded as "right of the
    # negative split", as in the trained tree.
    code_table = code_table[code_table["Threshold"] >= 0]
    code_table = code_table.reset_index(drop=True)
    # When every split of the feature is negative, only one row is left, whose range
    # would be every value (0..max). P4Runtime refuses a range that covers all values
    # ("range don't care match"), so the row is split in two (0..0 and 1..max) with the
    # same code, as the table had before the negative row was dropped.
    if len(code_table) == 1:
        second_row = code_table.iloc[[0]].copy()
        second_row["Threshold"] = code_table["Threshold"].iloc[0] + 1
        code_table = pd.concat([code_table, second_row]).reset_index(drop=True)
    return code_table


## get feature tables with ranges and codes only
def get_feature_codes_with_ranges(feature_table, num_of_trees):
    Codes = pd.DataFrame()
    for tree_id in range(num_of_trees):
        colname = "code" + str(tree_id)
        tree_cols = [col for col in feature_table.columns if ("s" + str(tree_id) + "_") in col]
        if not tree_cols:
            Codes[colname] = ["0b"] * len(feature_table)
        else:
            Codes[colname] = feature_table[tree_cols].apply(
                lambda x: "".join(x.dropna().astype(str)), axis=1)
            Codes[colname] = ["0b" + x for x in Codes[colname]]

    # Create a list to hold the range strings
    range_list = []
    thresholds = feature_table["Threshold"].values
    num_rows = len(thresholds)

    for i in range(num_rows):
        if i == 0:
            # First row: 0 to first threshold
            range_str = f"0,{thresholds[0]}"
        elif i == num_rows - 1:
            # Last row: threshold to threshold (as per your logic)
            range_str = f"{thresholds[i]},{thresholds[i]}"
        else:
            # Middle rows: prev+1 to current
            range_str = f"{thresholds[i-1] + 1},{thresholds[i]}"

        range_list.append(range_str)

    # Assign the list to the column all at once
    # This automatically sets the dtype to 'object' and avoids .loc issues
    feature_table["Range"] = range_list

    Ranges = feature_table["Range"]
    return Ranges, Codes

## get list of splits crossed to get to leaves
def retrieve_branches(estimator):
    number_nodes = estimator.tree_.node_count
    children_left_list = estimator.tree_.children_left
    children_right_list = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    # Calculate if a node is a leaf
    is_leaves_list = [
        (False if cl != cr else True)
        for cl, cr in zip(children_left_list, children_right_list)
    ]
    # Store the branches paths
    paths = []
    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]
            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output
        else:
            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]
            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])
            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left_list[i]])
                paths.append([i, children_right_list[i]])


## get classes and certainties
def get_classes(clf):
    leaves = []
    classes = []
    certainties = []
    for branch in list(retrieve_branches(clf)):
        leaves.append(branch[-1])
    for leaf in leaves:
        if clf.tree_.n_outputs == 1:
            value = clf.tree_.value[leaf][0]
        else:
            value = clf.tree_.value[leaf].T[0]
        class_name = np.argmax(value)
        certainty = int(round(max(value) / sum(value), 2) * 100)
        classes.append(class_name)
        certainties.append(certainty)
    return classes, certainties


## get the codes corresponging to the branches followed
def get_leaf_paths(clf):
    depth = clf.max_depth
    branch_codes = []
    for branch in list(retrieve_branches(clf)):
        code = [0] * len(branch)
        for i in range(1, len(branch)):
            if branch[i] == clf.tree_.children_left[branch[i - 1]]:
                code[i] = 0
            elif branch[i] == clf.tree_.children_right[branch[i - 1]]:
                code[i] = 1
        branch_codes.append(list(code[1:]))
    return branch_codes


## get the order of the splits to enable code generation
def get_order_of_splits(data, feature_names):
    splits_order = []
    for feature_name in feature_names:
        feature_data = data[data.iloc[:, 4] == feature_name]
        feature_data = feature_data.sort_values(by="Threshold")
        for node in list(feature_data.iloc[:, 0]):
            splits_order.append(node)
    return splits_order


def get_splits_per_tree(clf, feature_names):
    data = []
    n_nodes = clf.tree_.node_count
    # set feature names
    features = [feature_names[i] if i >= 0 else None for i in clf.tree_.feature]
    # generate dataframe with all thresholds and features
    for i in range(0, n_nodes):
        node_id = i
        left_child_id = clf.tree_.children_left[i]
        right_child_id = clf.tree_.children_right[i]
        threshold = clf.tree_.threshold[i]
        feature = features[i]
        if threshold != -2.0:
            data.append([node_id, left_child_id, right_child_id, threshold, feature])
    data = pd.DataFrame(data)
    data.columns = ["NodeID", "LeftID", "RightID", "Threshold", "Feature"]
    return data


## Get codes and masks
def get_codes_and_masks(clf, feature_names):
    splits = get_order_of_splits(get_splits_per_tree(clf, feature_names), feature_names)
    depth = clf.max_depth
    codes = []
    masks = []
    for branch, coded in zip(list(retrieve_branches(clf)), get_leaf_paths(clf)):
        code = [0] * len(splits)
        mask = [0] * len(splits)
        for index, split in enumerate(splits):
            if split in branch:
                mask[index] = 1
        masks.append(mask)
        codes.append(code)
    masks = pd.DataFrame(masks)
    masks["Mask"] = masks[masks.columns[0:]].apply(
        lambda x: "".join(x.dropna().astype(str)), axis=1
    )
    masks = ["0b" + x for x in masks["Mask"]]
    indices = range(0, len(splits))
    temp = pd.DataFrame(columns=["split", "index"], dtype=object)
    temp["split"] = splits
    temp["index"] = indices
    final_codes = []
    for branch, code, coded in zip(
        list(retrieve_branches(clf)), codes, get_leaf_paths(clf)
    ):
        indices_to_use = temp[temp["split"].isin(branch)].sort_values(by="split")[
            "index"
        ]
        for i, j in zip(range(0, len(coded)), list(indices_to_use)):
            code[j] = coded[i]
        final_codes.append(code)
    final_codes = pd.DataFrame(final_codes)
    final_codes["Code"] = final_codes[final_codes.columns[0:]].apply(
        lambda x: "".join(x.dropna().astype(str)), axis=1
    )
    final_codes = ["0b" + x for x in final_codes["Code"]]
    return final_codes, masks


## End of model manipulation ##


def extractKBits(num):
    # convert number into binary first
    binary = bin(int(num))
    # remove first two characters
    binary = binary[2:].zfill(30)
    # extract required bits
    num_bin = str("0b") + binary[0:20]
    num_dec = int(num_bin, 2)
    return num_dec

def convert_model(model, feature_offset, code_table_offset, commands):
    logging.info('Converting model : %s', model)
    # import and get entries from trained models ##
    clf = pd.read_pickle(model)

    # list the feature names
    feature_names = clf.feature_names_in_
    logging.info('Model features : %s', feature_names)

    tree_code_sizes = [[] for _ in range(len(clf.estimators_))]

    logging.info('Generating feature table commands')
    for fea in range(0, len(feature_names)):
        table_name = f"InferenceModel.TableFeature{fea + feature_offset}"
        action_name = f"InferenceModel.SetCode{fea + feature_offset}"
        logging.info('Table : %s', table_name)
        Ranges, Codes = get_feature_codes_with_ranges(
            get_feature_table(get_splits(clf, feature_names), feature_names[fea]),
            len(clf.estimators_),
        )
        for i, ran in enumerate(Ranges):
            start = ran.split(",")[0]
            end = ran.split(",")[1] if i < len(Ranges) - 1 else "65535"
            
            # ALTERAÇÃO 1: Usa bin(int(X, 2)) para remover o padding de zeros.
            # Skip trees that never split on this feature (0-bit codeword, str "0b"):
            # int("0b", 2) raises, and the generated SetCode action omits those
            # params (see model.p4.j2 / used_codes), so the command must too.
            codes_str = " ".join(
                bin(int(Codes.iloc[i, j], 2))
                for j in range(len(clf.estimators_))
                if len(str(Codes.iloc[i, j])) - 2 > 0
            )
            commands.append(f"table_add {table_name} {action_name} {start}->{end} => {codes_str} 1")
            
        for j in range(0, len(clf.estimators_)):
            tree_code_sizes[j].append(len(Codes.iloc[0, j]) - 2)

    logging.info('Generating code table commands')
    for tree_id in range(0, len(clf.estimators_)):
        table_name = f"InferenceModel.CodeTable{tree_id + code_table_offset}"
        action_name = f"InferenceModel.SetClass{tree_id + code_table_offset}"
        logging.info('Table : %s', table_name)
        Final_Codes, Final_Masks = get_codes_and_masks(
            clf.estimators_[tree_id], feature_names
        )
        Classe, Certain = get_classes(clf.estimators_[tree_id])
        for cod, mas, cla, cer in zip(Final_Codes, Final_Masks, Classe, Certain):
            start = 0
            cod = str(cod).removeprefix("0b")
            mas = str(mas).removeprefix("0b")
            match_fields = []
            for i, size in enumerate(tree_code_sizes[tree_id]):
                if size == 0:  # feature unused in this tree -> no codeword bits; CodeTable key skips it too
                    continue
                code_bits = cod[start : start + size]
                mask_bits = mas[start : start + size]
                start += size

                # ALTERAÇÃO 2: Usa bin() no valor convertido para decimal para limpar os zeros das máscaras e códigos
                match_fields.append(f"{bin(int(code_bits, 2))}&&&{bin(int(mask_bits, 2))}")
                
            match_str = " ".join(match_fields)
            commands.append(f"table_add {table_name} {action_name} {match_str} => {cla + 1} 1")

    # if len(clf.estimators_) > 1:
    #     logging.info('Generating voting table commands')
    #     for i in range(1, 4):
    #         for j in range(1, 4):
    #             for k in range(1, 4):
    #                 if (i != j) & (j != k) & (i != k):
    #                     pass
    #                 else:
    #                     commands.append(f"table_add InferenceModel.VotingTable InferenceModel.SetVotingResult {i} {j} {k} => {mode([i, j, k])}")
    # The former loop above only covered 3 trees and the classes 1 to 3. Same entries as
    # the controller (convert_RF_and_populate_tables.py): one per combination of the
    # classes of the trees (1-based ids), with the most common class as the result.
    if len(clf.estimators_) > 1:
        logging.info('Generating voting table commands')
        p4_classes = [c + 1 for c in range(len(clf.classes_))]
        for combo in itertools.product(p4_classes, repeat=len(clf.estimators_)):
            keys = " ".join(str(cls) for cls in combo)
            commands.append(f"table_add InferenceModel.VotingTable InferenceModel.SetVotingResult {keys} => {mode(list(combo))}")

    return len(feature_names), len(clf.estimators_)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output-file", default="commands.txt")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format=FORMAT)

    np.random.seed(42)

    commands = []
    feature_offset = 0
    code_table_offset = 0
    logging.info('Converting models : %s', args.models)
    for model in args.models:
        len_features, len_code_tables = convert_model(
            model, feature_offset, code_table_offset, commands
        )
        feature_offset += len_features
        code_table_offset += len_code_tables

    with open(args.output_file, 'w') as f:
        for cmd in commands:
            f.write(cmd + '\n')

    print(f"\nCommands written to: {args.output_file}")
    print(f"Total table entries: {len(commands)}")


if __name__ == "__main__":
    main()