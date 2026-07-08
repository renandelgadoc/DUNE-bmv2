import numpy as np
import pandas as pd
import scipy.stats as sps

#### PCFI - Per Class Feature Importance ####
def f_gini(v):
    # Add a small epsilon to prevent 0/0 division if v.sum() is 0
    total = v.sum()
    if total == 0:
        return 0.0
    p = v / total
    return np.multiply(p, 1-p).sum()

def f_entropy(v):
    total = v.sum()
    if total == 0:
        return 0.0
    return sps.entropy(pk=v)

def f_misclassification(v):
    total = v.sum()
    if total == 0:
        return 0.0
    p = v / total
    # Fixed: added the missing return statement for misclassification impurity!
    return 1.0 - np.max(p)
    
def calcImportanceMatrix(rf_clf):
    """
    Calculates the importance matrix of predictors for each class.

    Arguments:
        rf_clf - The random forest classifier to calculate the importance matrix for.

    Returns:
        importance_matrix - The importance matrix with the importance of each predictor in predicting a class.
        A n by m numpy array.

    """

    # get the number of classes being predicted by the random forest
    classes = rf_clf.classes_
    n_classes = len(classes)

    # init storage for the predictor importances by classes by trees
    importance_matrix = []

    import time as _time
    _n_trees = len(rf_clf.estimators_)
    _pcfi_start = _time.time()
    print(f"[PCFI] calcImportanceMatrix: processing {_n_trees} trees...", flush=True)

    for _t_idx, dec_tree in enumerate(rf_clf.estimators_):

        # get the criterion used to measure impurity
        criterion = dec_tree.get_params()['criterion']
        if criterion == 'gini':
            f_impurity = f_gini
        elif criterion == 'entropy':
            f_impurity = f_entropy
        elif criterion == 'misclassification':
            f_impurity = f_misclassification
        else:
            f_impurity = 0

        # get the number of features and nodes in the tree
        feature = dec_tree.tree_.feature
        n_features = dec_tree.tree_.n_features
        n_nodes = dec_tree.tree_.__getstate__()['node_count']
        nodes = dec_tree.tree_.__getstate__()['nodes']
        parent_node_ind = -np.ones(shape=n_nodes, dtype='<i8')
        
        for par_ind,node in enumerate(nodes):
            if node[0] != -1:
                parent_node_ind[node[0]] = par_ind
            if node[1] != -1:
                parent_node_ind[node[1]] = par_ind

        # identify the leaves of the tree
        is_leaves = np.array([node[0]==-1 and node[1]==-1 for node in nodes])
        leaves_index = np.nonzero(is_leaves)[0]
        values_sorted = dec_tree.tree_.__getstate__()['values']
        
        node_pred = np.argmax(values_sorted[:,0,:], axis=1)
        leaves_class_index = node_pred[is_leaves]
        
        node_unvisited = np.ones((n_classes, n_nodes), dtype=bool)
        tree_importances = np.zeros((n_classes, n_features))
        for leaf_i,leaf_c_i in zip(leaves_index,leaves_class_index):
            current_i = parent_node_ind[leaf_i]
            
            # walk the tree and calculate the importance of the predictor
            while current_i != -1 and node_unvisited[leaf_c_i,current_i]:
                current_node = nodes[current_i]
                left_node = nodes[current_node['left_child']]
                right_node = nodes[current_node['right_child']]
                current_feature = current_node['feature']
                
                current_values = values_sorted[current_i,0,:]
                left_values = values_sorted[current_node['left_child'],0,:]
                right_values = values_sorted[current_node['right_child'],0,:]
                
                current_values_class = np.array([
                    current_values[leaf_c_i],
                    current_values[np.arange(len(current_values)) != leaf_c_i].sum()
                ])
                left_values_class = np.array([
                    left_values[leaf_c_i],
                    left_values[np.arange(len(left_values)) != leaf_c_i].sum()
                ])
                right_values_class = np.array([
                    right_values[leaf_c_i],
                    right_values[np.arange(len(right_values)) != leaf_c_i].sum()
                ])
                tree_importances[leaf_c_i,current_feature] += (
                        current_node['weighted_n_node_samples'] * f_impurity(current_values_class) -
                        left_node['weighted_n_node_samples'] * f_impurity(left_values_class) -
                        right_node['weighted_n_node_samples'] * f_impurity(right_values_class)
                        )
                ###
                node_unvisited[leaf_c_i,current_i] = False
                current_i = parent_node_ind[current_i]
        
        importance_matrix.append(tree_importances/nodes[0]['weighted_n_node_samples'])

        _el = _time.time() - _pcfi_start
        print(f"[PCFI]   tree {_t_idx+1}/{_n_trees} done "
              f"({_el:.1f}s elapsed, {_el/(_t_idx+1):.2f}s/tree avg, "
              f"~{_el/(_t_idx+1)*(_n_trees-_t_idx-1):.1f}s left)", flush=True)

    importance_matrix = np.array(importance_matrix)
    importance_matrix = np.nan_to_num(importance_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    importance_matrix = np.mean(importance_matrix, axis = 0)

    row_sums = np.sum(importance_matrix, axis=1)    
    row_sums[row_sums == 0] = 1.0 
    
    importance_matrix = (importance_matrix.T / row_sums).T
    
    return(importance_matrix)

def get_feats_importance(rf_opt, classes, feats, test_label_names):
    '''
    Get the importance matrix of predictors for each class.
    Arguments:
        rf_opt - The random forest classifier to calculate the importance matrix for.
        classes - Classes classified with the model (can be a superset of trained classes)
        feats - Features selected for the model
    Returns:
        importance_matrix - The importance matrix with the importance of each predictor in predicting a class.
        A dataframe
    '''
    ### Matrix containing feature importances per feature-class pair
    print(f"[PCFI] get_feats_importance: {len(classes)} classes, {len(feats)} features", flush=True)
    imp_mat = calcImportanceMatrix(rf_opt)
    print("[PCFI] importance matrix ready; ordering by class...", flush=True)
    
    ### Get the actual classes the model was trained on
    model_classes = test_label_names
    
    ### GET ordered classes and GET their features and feature importances in that order
    imp_mat_ordered = []

    for cl in classes:
        if cl in model_classes:
            ind_of_class = np.where(test_label_names == cl)[0][0]
            imp_mat_ordered.append(imp_mat[ind_of_class])
        else:
            # If the model wasn't trained on this class, return zeros for its feature importance
            imp_mat_ordered.append(np.zeros(len(feats)))
        
    feats_importance_df = pd.DataFrame(imp_mat_ordered, columns=feats)
    feats_importance_df['classes'] = classes
    feats_importance_df = feats_importance_df.loc[:, ['classes'] + feats]
        
    return feats_importance_df