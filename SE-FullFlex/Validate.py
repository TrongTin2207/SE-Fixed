import networkx as nx 
import pulp as pl
from common import *

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
    
       # Check node capacity constraints
    for i in PHY.nodes:
        node_capacity_used = sum(
            get_var(f'xNode_{s}_{k}_{i}_{v}') * subgraph.nodes[v]['r']
            for s, slice_config in enumerate(slices)
            for k, subgraph in enumerate(slice_config)
            for v in subgraph.nodes
        )
        if node_capacity_used > aNode[i]:
            print(f"Constraint 1 failed")
            return False

    # Check edge capacity constraints
    for (i, j) in PHY.edges:
        edge_capacity_used = sum(
            get_var(f'xEdge_{s}_{k}_{i}_{j}_{v}_{w}') * subgraph.edges[v, w]['r']
            for s, slice_config in enumerate(slices)
            for k, subgraph in enumerate(slice_config)
            for (v, w) in subgraph.edges
        )
        if edge_capacity_used > aEdge[(i, j)]:
            print(f"Constraint 2 failed")
            return False

    # Check one virtual node per physical node per slice constraint
    for s, slice_config in enumerate(slices):
        for k, subgraph in enumerate(slice_config):
            for i in PHY.nodes:
                total_virtual_nodes = sum(get_var(f'xNode_{s}_{k}_{i}_{v}') for v in subgraph.nodes)
                if total_virtual_nodes > get_var(f'z_{s}_{k}'):
                    print(f"Constraint 3 failed")
                    return False

    # Check each virtual node mapped to exactly one physical node if chosen    
    if not all (pl.lpSum(get_var(f'xNode_{s}_{k}_{i}_{v}') for v in subgraph.nodes 
                                     for i in PHY.nodes)):
        print(f"Constraint 4 failed")
        return False

    # Check flow conservation constraints
    M = 100  # Big-M value for relaxation
    for (v, w) in subgraph.edges:
        for (i, j) in PHY.edges:
            lhs = get_var(f'xEdge_{s}_{k}_{i}_{j}_{v}_{w}') - get_var(f'xEdge_{s}_{k}_{j}_{i}_{v}_{w}') - (get_var(f'xNode_{s}_{k}_{i}_{v}') - get_var(f'xNode_{s}_{k}_{j}_{v}'))
            if not (-M * (1 - get_var(f'phi_{s}_{k}')) <= lhs <= M * (1 - get_var(f'phi_{s}_{k}'))):
                print(f"Constraint 5")
                return False

    # Check exactly one configuration chosen per slice
    for s in range(len(slices)):
        total_configurations = sum(get_var(f'phi_{s}_{k}') for k in range(len(slices[s])))
        if total_configurations != get_var(f'pi_{s}'):
            print(f"Constraint 6 failed")
            return False

    # Check consistency between z, pi, and phi variables
    for s in range(len(slices)):
        for k in range(len(slices[s])):
            z_var = get_var(f'z_{s}_{k}')
            pi_var = get_var(f'pi_{s}')
            phi_var = get_var(f'phi_{s}_{k}')
            if z_var > pi_var or z_var > phi_var or z_var < pi_var + phi_var - 1:
                print(f"Constraint 7 failed")
                return False

    print("All constraints are satisfied.")
    return True

# Check constraint 6 again
# If solution violates, return False and escape
# Changing variables naming 