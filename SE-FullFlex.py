import networkx as nx
import pulp as pl

def create_physical_network(num_nodes: int, node_requirements: list[int]) -> nx.DiGraph:
    """
    Tạo đồ thị mạng vật lý với số lượng node và yêu cầu tài nguyên cho từng node.
    
    Parameters:
    num_nodes (int): Số lượng node trong mạng vật lý.
    node_requirements (list[int]): Danh sách yêu cầu tài nguyên của từng node.
    
    Returns:
    nx.DiGraph: Đồ thị mạng vật lý đã tạo.
    """
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, a=node_requirements[i])
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, a=10)
        G.add_edge(i + 1, i, a=10)
    return G

def create_slice_configurations(slice_id: int, num_config: int, sizes: list[int], node_requirements: list[list[int]]) -> list[nx.DiGraph]:
    """
    Tạo các cấu hình cho slice với số lượng node và yêu cầu tài nguyên cho từng node.
    
    Parameters:
    slice_id (int): ID của slice.
    num_config (int): Số lượng cấu hình cho slice.
    sizes (list[int]): Danh sách kích thước của từng cấu hình.
    node_requirements (list[list[int]]): Danh sách yêu cầu tài nguyên của từng node trong các cấu hình.
    
    Returns:
    list[nx.DiGraph]: Danh sách các cấu hình đã tạo cho slice.
    """
    configurations = []
    for config_id in range(num_config):
        G = nx.DiGraph()
        for i in range(sizes[config_id]):
            G.add_node(i, r=node_requirements[config_id][i])
        for i in range(sizes[config_id] - 1):
            G.add_edge(i, i + 1, r=5)
        configurations.append(G)
    return configurations

def build_ilp_problem(slices: list[list[nx.DiGraph]], N: nx.DiGraph) -> pl.LpProblem:
    """
    Xây dựng bài toán ILP cho việc ánh xạ các slice với nhiều cấu hình vào mạng vật lý.
    
    Parameters:
    slices (list[list[nx.DiGraph]]): Danh sách các slice với nhiều cấu hình.
    N (nx.DiGraph): Đồ thị mạng vật lý.
    
    Returns:
    pl.LpProblem: Bài toán ILP đã xây dựng.
    """
    problem = pl.LpProblem(name='Graph-Mapping', sense=pl.LpMaximize)

    xNode = pl.LpVariable.dicts("xNode",
                          ((s, k, n, i)
                          for s, slice_configs in enumerate(slices)
                          for k, subgraph in enumerate(slice_configs)
                          for n in subgraph.nodes
                          for i in N.nodes),
                          cat=pl.LpBinary
    )

    xEdge = pl.LpVariable.dicts("xEdge", 
                                ((s, k, w, v, i, j) 
                                 for s, slice_configs in enumerate(slices)
                                 for k, subgraph in enumerate(slice_configs)
                                 for w, v in subgraph.edges 
                                 for i, j in N.edges),
                                cat=pl.LpBinary)

    pi = pl.LpVariable.dicts("pi", (s for s in range(len(slices))), cat=pl.LpBinary)
    phi = pl.LpVariable.dicts("phi", 
                              ((s, k) 
                               for s in range(len(slices)) 
                               for k in range(len(slices[s]))),
                              cat=pl.LpBinary)
    z = pl.LpVariable.dicts("z", 
                            ((s, k)
                             for s in range(len(slices)) 
                             for k in range(len(slices[s]))),
                            cat=pl.LpBinary)

    # Attribute of the target graph
    aNode = nx.get_node_attributes(N, "a")
    aEdge = nx.get_edge_attributes(N, "a")

    # Constraints
    for s, slice_config in enumerate(slices):
        for k, subgraph in enumerate(slice_config):
            rNode = nx.get_node_attributes(subgraph, "r")
            rEdge = nx.get_edge_attributes(subgraph, "r")

            # C1 constraint
            for i in N.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, k, n, i)] * rNode[n]
                        for n in subgraph.nodes
                    ) <= aNode[i] * phi[(s, k)],
                    f'C1_{s}_{k}_{i}'
                )

            # C2 constraint
            for i, j in N.edges:
                problem += (
                    pl.lpSum(
                        xEdge[(s, k, w, v, i, j)] * rEdge[(w, v)]
                        for w, v in subgraph.edges
                    ) <= aEdge[(i, j)] * phi[(s, k)],
                    f'C2_{s}_{k}_{i}_{j}'
                )

            # C3 constraint
            for i in N.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, k, n, i)]
                        for n in subgraph.nodes
                    ) <= z[(s, k)],
                    f'C3_{s}_{k}_{i}'
                )

            # C4 constraint
            for n in subgraph.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, k, n, i)]
                        for i in N.nodes
                    ) == phi[(s, k)],
                    f'C4_{s}_{k}_{n}'
                )

            # C5 constraints
            big_M = 100  # Define a sufficiently large value for M
            for (w, v) in subgraph.edges:
                for i in N.nodes:
                    for j in N.nodes:
                        if i != j:
                            if (s, k, w, v, i, j) in xEdge and (s, k, w, v, j, i) in xEdge:
                                problem += (
                                    xEdge[(s, k, w, v, i, j)] - xEdge[(s, k, w, v, j, i)] 
                                    - (xNode[(s, k, v, i)] - xNode[(s, k, w, i)]) <= big_M * (1 - phi[(s, k)]),
                                    f'C5_{s}_{k}_{w}_{v}_{i}_{j}_1'
                                )
                                problem += (
                                    xEdge[(s, k, w, v, i, j)] - xEdge[(s, k, w, v, j, i)] 
                                    - (xNode[(s, k, v, i)] - xNode[(s, k, w, i)]) >= -big_M * (1 - phi[(s, k)]),
                                    f'C5_{s}_{k}_{w}_{v}_{i}_{j}_2'
                                )

    # C6 constraint
    for s in range(len(slices)):
        problem += (
            pl.lpSum(phi[(s, k)] for k in range(len(slices[s]))) == pi[s],
            f'C6_{s}'
        )

    # C7 constraint
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

    # Objective function
    gamma = 0.99999
    problem += gamma * pl.lpSum(pi[s] for s in range(len(slices))) - (1 - gamma) * pl.lpSum(xEdge[(s, k, w, v, i, j)] * rEdge[(w, v)]
    for s, slice_configs in enumerate(slices) 
    for k, subgraph in enumerate(slice_configs)
    for w, v in subgraph.edges
    for i, j in N.edges)

    return problem

def main():
    # Tạo đồ thị mạng vật lý với yêu cầu tài nguyên cho từng node
    N_requirements = [10, 10, 10, 10, 10]
    N = create_physical_network(5, N_requirements)
    
    # Tạo tập hợp các slice với nhiều cấu hình và yêu cầu tài nguyên cho từng node
    sizes = [
        [3, 2],
        [2, 4],
        [3, 3]
    ]
    requirements = [
        [[3, 3, 3], [2, 2]],
        [[4, 4], [2, 2, 2, 2]],
        [[1, 1, 1], [2, 2, 2]]
    ]
    
    slices = []
    for slice_id in range(3):
        slice_configs = create_slice_configurations(slice_id, len(sizes[slice_id]), sizes[slice_id], requirements[slice_id])
        slices.append(slice_configs)

    # Xây dựng bài toán ILP
    ilp_problem = build_ilp_problem(slices, N)

    # Giải bài toán ILP
    ilp_problem.solve()
    
    # In kết quả
    for var in ilp_problem.variables():
        print(f'{var.name}: {var.value()}')
    print(f'Tối ưu hóa giá trị: {pl.value(ilp_problem.objective)}')

if __name__ == '__main__':
    main()
