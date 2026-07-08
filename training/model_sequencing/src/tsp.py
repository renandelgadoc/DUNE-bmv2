import pyomo.environ as pyo
import numpy as np

import pyomo.environ as pyo
import numpy as np

def order_blocks_with_tsp(costMatrix_FP, cost_F1, print_gap=False):
    '''
    This function implements a Mixed Integer Linear Programming (MILP) model to solve a modified Traveling Salesman Problem (TSP)
    to find the optimal sequence of the blocks.
    '''
    n = costMatrix_FP.shape[0]

    # --- FIX 1: Sanitize the input matrix to replace NaNs with 0.0 ---
    # This prevents 'nan' from entering the LP equations
    costMatrix_FP_clean = np.nan_to_num(costMatrix_FP, nan=0.0)

    # --- FIX 2: Define a "Big M" penalty instead of np.inf ---
    # Solvers cannot parse literal infinity. 1,000,000 is large enough to be avoided.
    BIG_M_PENALTY = 1e6 

    cost_matrix = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                cost_matrix[i][j] = costMatrix_FP_clean[j][i] * (1 - cost_F1[i] / 100.0)
            else:
                cost_matrix[i][j] = BIG_M_PENALTY # Used instead of np.inf

    # 2 | initialize the model
    model = pyo.ConcreteModel()
    # Indexes for the cities
    model.M = pyo.RangeSet(n)
    model.N = pyo.RangeSet(n)
    # Index for the dummy variable u
    model.U = pyo.RangeSet(2, n)

    # 3 | initialize decision variables
    # Decision variables x_{i,j}
    model.x = pyo.Var(model.N, model.M, within=pyo.Binary)

    # Dummy variable u_i
    model.u = pyo.Var(model.N, within=pyo.NonNegativeIntegers, bounds=(0, n - 1))

    # --- FIX 3: Added 'within=pyo.Reals' to suppress the Pyomo domain warning ---
    # Cost Matrix cij
    model.c = pyo.Param(model.N, model.M, within=pyo.Reals, initialize=lambda model, i, j: cost_matrix[i - 1][j - 1])

    # 4 | define objective function
    def obj_func(model):
        return sum(model.x[i, j] * model.c[i, j] for i in model.N for j in model.M)

    model.objective = pyo.Objective(rule=obj_func, sense=pyo.minimize)

    # 5 | define constraints
    def rule_const1(model, i, j):
        '''
        this contraint ensures that only one edge is selected between two nodes
        '''
        if i != j:
            return model.x[i, j] + model.x[j, i] == 1
        else:
            return model.x[i, j] - model.x[i, j] == 0
    model.const1 = pyo.Constraint(model.N, model.M, rule=rule_const1)

    def rule_const2(model, i, j):
        '''
        this constraint eliminates the possibility of subtours
        '''
        if i != j:
            return model.u[i] - model.u[j] + model.x[i, j] * n <= n - 1
        else:
            return model.u[i] - model.u[i] == 0

    model.rest3 = pyo.Constraint(model.U, model.N, rule=rule_const2)

    # 6 | call the solver
    solver = pyo.SolverFactory('gurobi')
    completeResults = solver.solve(model, tee=True)

    # 7 | extract the results
    selected_edges = []
    for i in model.x:
        if model.x[i].value > 0.5: # Changed from > 0 to > 0.5 to safely handle floating point inaccuracies
            selected_edges.append((i, model.x[i].value))

    if print_gap:
        solution_gap = (completeResults.Problem._list[0]['Upper bound'] - completeResults.Problem._list[0]['Lower bound']) / completeResults.Problem._list[0]['Upper bound']
        print(solution_gap)


    def get_cluster_seq(selected_edges, n_of_clusters):
        '''
        The TSP problem returs a directed acyclic structure rather than a simple ordered tour.
        The directionality ensures that node x comes before node y in the sequence.
        For example, the solver provides, as a solution, the following:
                (2, 1)-->1.0
                (2, 3)-->1.0
                (2, 4)-->1.0
                (3, 1)-->1.0
                (4, 1)-->1.0
                (4, 3)-->1.0
        This indicates that the x_{2,1} variable is set to 1.0, which means that cluster 2 comes before cluster 1 in
        the sequence. Cluster 2 should also come before cluster 3 and 4,
        and cluster 4 should come before cluster 1 and 3.

        This function takes the selected edges and returns the sequence of clusters.
        '''
        store_cluster_occurrences = {cluster_idx: 0 for cluster_idx in range(1, n_of_clusters + 1)}
        for edge in selected_edges:
            node = edge[0][0]
            store_cluster_occurrences[node] += 1

        clusters_seq = sorted(store_cluster_occurrences, reverse=True, key=store_cluster_occurrences.get)

        return list(map(lambda x: x - 1, clusters_seq))

    clusters_seq = get_cluster_seq(selected_edges, n)

    return clusters_seq