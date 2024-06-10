import networkx as nx 
import pulp as pl

def create_physical_network(num_nodes: int, node_requirements: list[int], edge_requirements: list[int]) -> nx.DiGraph:
    """
    Create a physical network graph with a specified number of nodes and resource requirements for each node.
    
    Parameters:
    num_nodes (int): Number of nodes in the physical network.
    node_requirements (list[int]): List of resource requirements for each node.
    edge_requirements (list[int]): List of resource requirements for each edge.
    
    Returns:
    nx.DiGraph: The created physical network graph.
    """
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, a=node_requirements[i])
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, a=edge_requirements[i])
        G.add_edge(i + 1, i, a=edge_requirements[i])
    return G

def create_slice_configurations(num_config: int, sizes: list[int], node_requirements: list[list[int]], edge_requirements: list[list[int]]) -> list[nx.DiGraph]:
    """
    Create slice configurations with a specified number of nodes and resource requirements for each node.
    
    Parameters:
    num_config (int): Number of configurations for the slice.
    sizes (list[int]): List of sizes for each configuration.
    node_requirements (list[list[int]]): List of resource requirements for each node in each configuration.
    edge_requirements (list[list[int]]): List of resource requirements for each edge in each configuration.
    
    Returns:
    list[nx.DiGraph]: List of created configurations for the slice.
    """
    configurations = []
    for config_id in range(num_config):
        G = nx.DiGraph()
        for i in range(sizes[config_id]):
            G.add_node(i, r=node_requirements[config_id][i])
        for i in range(sizes[config_id] - 1):
            G.add_edge(i, i + 1, r=edge_requirements[config_id][i])
        configurations.append(G)
    return configurations

def build_ilp_problem(slices: list[list[nx.DiGraph]], PHY: nx.DiGraph) -> pl.LpProblem:
    """
    Build the ILP problem for mapping slices with multiple configurations onto a physical network.
    
    Parameters:
    slices (list[list[nx.DiGraph]]): List of slices with multiple configurations.
    PHY (nx.DiGraph): The physical network graph.
    
    Returns:
    pl.LpProblem: The built ILP problem.
    """
    problem = pl.LpProblem(name='Graph-Mapping', sense=pl.LpMaximize)

    # Define binary variables for node mapping
    xNode = pl.LpVariable.dicts(name = "xNode",
                          indices = ((s, k, i, v)
                          for s, slice_configs in enumerate(slices)
                          for k, subgraph in enumerate(slice_configs)
                          for v in subgraph.nodes
                          for i in PHY.nodes),
                          cat=pl.LpBinary
    )

    # Define binary variables for edge mapping
    xEdge = pl.LpVariable.dicts(name = "xEdge", 
                                indices = ((s, k, (i, j), (v, w)) 
                                 for s, slice_configs in enumerate(slices)
                                 for k, subgraph in enumerate(slice_configs)
                                 for v, w in subgraph.edges 
                                 for i, j in PHY.edges),
                                cat=pl.LpBinary)

    # Define binary variables to indicate whether a slice is mapped
    pi = pl.LpVariable.dicts(name = "pi", indices = (s for s in range(len(slices))), cat=pl.LpBinary)
    # Define binary variables to indicate whether a slice configuration is chosen
    phi = pl.LpVariable.dicts(name = "phi", 
                              indices = ((s, k) 
                               for s in range(len(slices)) 
                               for k in range(len(slices[s]))),
                              cat=pl.LpBinary)
    # Define auxiliary binary variables for constraint management
    z = pl.LpVariable.dicts(name = "z", 
                            indices = ((s, k)
                             for s in range(len(slices)) 
                             for k in range(len(slices[s]))),
                            cat=pl.LpBinary)

    # Attributes of the physical network
    aNode = nx.get_node_attributes(PHY, "a")
    aEdge = nx.get_edge_attributes(PHY, "a")

    # Constraints
    for s, slice_config in enumerate(slices):
        for k, subgraph in enumerate(slice_config):
            rNode = nx.get_node_attributes(subgraph, "r")
            rEdge = nx.get_edge_attributes(subgraph, "r")

            # C1: Ensure that the total resource demand of virtual nodes mapped to a physical node does not exceed its capacity
            for i in PHY.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, k, i, v)] * rNode[v]
                        for v in subgraph.nodes
                    ) <= aNode[i] * phi[(s, k)],
                    f'C1_{s}_{k}_{i}'
                )

            # C2: Ensure that the total resource demand of virtual edges mapped to a physical edge does not exceed its capacity
            for (i, j) in PHY.edges:
                problem += (
                    pl.lpSum(
                        xEdge[(s, k, (i, j), (v, w))] * rEdge[(v, w)]
                        for v, w in subgraph.edges
                    ) <= aEdge[(i, j)] * phi[(s, k)],
                    f'C2_{s}_{k}_{i}_{j}'
                )

            # C3: Ensure that each physical node hosts at most one virtual node from each slice configuration
            for i in PHY.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, k, i, v)]
                        for v in subgraph.nodes
                    ) <= z[(s, k)],
                    f'C3_{s}_{k}_{i}'
                )

            # C4: Ensure that each virtual node is mapped to exactly one physical node if the slice configuration is chosen
            for v in subgraph.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, k, i, v)]
                        for i in PHY.nodes
                    ) == phi[(s, k)],
                    f'C4_{s}_{k}_{v}'
                )

            # C5: Ensure the flow conservation for virtual edges using a big-M method for relaxation
            M = 100  # Define a sufficiently large value for M
            for (v, w) in subgraph.edges:
                for (i, j) in PHY.edges:
                    problem += (
                        xEdge[(s, k, (i, j), (v, w))] - xEdge[(s, k, (j, i), (v, w))] 
                        - (xNode[(s, k, i, v)] - xNode[(s, k, j, v)]) <= M * (1 - phi[(s, k)]),
                        f'C5_{s}_{k}_{v}_{w}_{i}_{j}_1'
                    )
                    problem += (
                        xEdge[(s, k, (i, j), (v, w))] - xEdge[(s, k, (j, i), (v, w))]
                        - (xNode[(s, k, i, v)] - xNode[(s, k, j, v)]) >= -M * (1 - phi[(s, k)]),
                        f'C5_{s}_{k}_{v}_{w}_{i}_{j}_2'
                    )

    # C6: Ensure that exactly one configuration is chosen for each slice
    for s in range(len(slices)):
        problem += (
            pl.lpSum(phi[(s, k)] for k in range(len(slices[s]))) == pi[s],
            f'C6_{s}'
        )

    # C7: Ensure consistency between z, pi, and phi variables
    for s in range(len(slices)):
        for k in range(len(slices[s])):
            problem += (
                z[(s, k)] <= pi[s],
                f'C7_{s}_{k}_1'
            )
            problem += (
                z[(s, k)] <= phi[(s, k)],
                f'C7_{s}_{k}_2'
            ) 
            problem += (
                z[(s, k)] >= pi[s] + phi[(s, k)] - 1,
                f'C7_{s}_{k}_3'
            )

    # Objective function: Maximize the number of mapped slices and minimize resource usage
    gamma = 0.99999
    problem += gamma * pl.lpSum(pi[s] for s in range(len(slices))) - (1 - gamma) * pl.lpSum(
    xEdge[(s, k, (i, j), (v, w))] * rEdge[(v, w)] if (v, w) in rEdge else 0
    for s, slice_configs in enumerate(slices)
    for k, subgraph in enumerate(slice_configs)
    for v, w in subgraph.edges
    for i, j in PHY.edges
    )
    
    return problem


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

    for s, slice_config in enumerate(slices):
        for k, subgraph in enumerate(slice_config):
            rNode = nx.get_node_attributes(subgraph, "r")
            rEdge = nx.get_edge_attributes(subgraph, "r")
    
    def get_var(name):
        return variables.get(name, 0)

    # Check node capacity constraints
    if not all(pl.lpSum(get_var(f'xNode_{s}_{k}_{i}_{v}') * rNode[v] for v in subgraph.nodes) <= aNode[i] * get_var(f'phi_{s}_{k}') for i in PHY.nodes):
        print(f"Constraint 1 failed")
        return False

    # Check edge capacity constraints

    if not all (pl.lpSum(get_var(f'xEdge_{s}_{k}_{i}_{j}_{v}_{w}') * rEdge.get((v, w), 0) for (v, w) in subgraph.edges) <= aEdge[(i, j)] * get_var(f'phi_{s}_{k}') 
                         for (i,j) in PHY.nodes):
        print(f"Constraint 2 failed)")
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