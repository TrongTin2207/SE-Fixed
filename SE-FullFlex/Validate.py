import networkx as nx
import pulp as pl
import pickle

def check_solution(problem, slices, PHY):
    """
    Check the solution of the ILP problem to ensure it satisfies all constraints.
    
    Parameters:
    problem (pl.LpProblem): The solved ILP problem.
    slices (list[list[nx.DiGraph]]): List of slices with multiple configurations.
    PHY (nx.DiGraph): The physical network graph.
    
    Returns:
    bool: True if all constraints are satisfied, False otherwise.
    """
    # Extract the variables from the solved problem
    variables = {v.name: v.varValue for v in problem.variables()}
    
    # Extract node and edge attributes from the physical network
    aNode = nx.get_node_attributes(PHY, "a")
    aEdge = nx.get_edge_attributes(PHY, "a")
    
    def get_var(name):
        return variables.get(name, 0)
    
    # Pre-compute node and edge capacity usage
    node_capacity_used = {i: 0 for i in PHY.nodes}
    edge_capacity_used = {edge: 0 for edge in PHY.edges}

    for s, slice_config in enumerate(slices):
        for k, subgraph in enumerate(slice_config):
            for i in PHY.nodes:
                for v in subgraph.nodes:
                    node_capacity_used[i] += get_var(f'xNode_{s}_{k}_{i}_{v}') * subgraph.nodes[v]['r']
            for (i, j) in PHY.edges:
                for (v, w) in subgraph.edges:
                    edge_capacity_used[(i, j)] += get_var(f'xEdge_{s}_{k}_{i}_{j}_{v}_{w}') * subgraph.edges[v, w]['r']

    # Check node capacity constraints (C1)
    if not all(node_capacity_used[i] <= aNode[i] for i in PHY.nodes):
        print(f"Constraint 1 violated.")
        return False

    # Check edge capacity constraints (C2)
    if not all(edge_capacity_used[(i, j)] <= aEdge[(i, j)] for (i, j) in PHY.edges):
        print(f"Constraint 2 violated.")
        return False

    # Check one virtual node per physical node per slice constraint (C3)
    if not all(
        sum(get_var(f'xNode_{s}_{k}_{i}_{v}') for v in subgraph.nodes) <= get_var(f'z_{s}_{k}')
        for s, slice_config in enumerate(slices)
        for k, subgraph in enumerate(slice_config)
        for i in PHY.nodes
    ):
        print(f"Constraint 3 failed.")
        return False

    # Check each virtual node mapped to exactly one physical node if chosen (C4)
    if not all(
        sum(get_var(f'xNode_{s}_{k}_{i}_{v}') for i in PHY.nodes) == get_var(f'z_{s}_{k}')
        for s, slice_config in enumerate(slices)
        for k, subgraph in enumerate(slice_config)
        for v in subgraph.nodes
    ):
        print(f"Constraint 4 failed.")
        return False

    # Check flow conservation constraints (C5)
    M = 100  # Big-M value for relaxation
    if not all(
        -M * (1 - get_var(f'phi_{s}_{k}')) <= 
        get_var(f'xEdge_{s}_{k}_{i}_{j}_{v}_{w}') - 
        get_var(f'xEdge_{s}_{k}_{j}_{i}_{v}_{w}') - 
        (get_var(f'xNode_{s}_{k}_{i}_{v}') - get_var(f'xNode_{s}_{k}_{j}_{v}')) <= 
        M * (1 - get_var(f'phi_{s}_{k}'))
        for s, slice_config in enumerate(slices)
        for k, subgraph in enumerate(slice_config)
        for (v, w) in subgraph.edges
        for (i, j) in PHY.edges
    ):
        print(f"Constraint 5 violated.")
        return False

    # Check exactly one configuration chosen per slice (C6)
    if not all(
        sum(get_var(f'phi_{s}_{k}') for k in range(len(slice_config))) == get_var(f'pi_{s}_{k}')
        for s, slice_config in enumerate(slices)
    ):
        print(f"Constraint 6 violated.")
        return False


    # Check consistency between z, pi, and phi variables (C7)
    if not all(
        get_var(f'z_{s}_{k}') <= get_var(f'pi_{s}') and
        get_var(f'z_{s}_{k}') <= get_var(f'phi_{s}_{k}') and
        get_var(f'z_{s}_{k}') >= get_var(f'pi_{s}') + get_var(f'phi_{s}_{k}') - 1
        for s in range(len(slices))
        for k in range(len(slices[s]))
    ):
        print(f"Constraint 7 violated.")
        return False

    print("All constraints are satisfied.")
    return True

def save_solution(problem, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(problem, f)

def load_solution(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Example usage:
# save_solution(problem, 'problem.pkl')
# loaded_problem = load_solution('problem.pkl')
# result = check_solution(loaded_problem, slices, PHY)
