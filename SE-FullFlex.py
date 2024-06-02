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

def create_subgraph(num_nodes: int, node_requirements: list[int]) -> nx.DiGraph:
    """
    Tạo một đồ thị con với số lượng node và yêu cầu tài nguyên cho từng node.
    
    Parameters:
    num_nodes (int): Số lượng node trong đồ thị con.
    node_requirements (list[int]): Danh sách yêu cầu tài nguyên của từng node.
    
    Returns:
    nx.DiGraph: Đồ thị con đã tạo.
    """
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i, r=node_requirements[i])
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1, r=5)
    return G

def create_subgraph_set(K: int, sizes: list[int], requirements: list[list[int]]) -> list[nx.DiGraph]:
    """
    Tạo tập hợp K đồ thị con với các kích thước khác nhau và yêu cầu tài nguyên cho từng node.
    
    Parameters:
    K (int): Số lượng đồ thị con cần tạo.
    sizes (list[int]): Danh sách kích thước của từng đồ thị con.
    requirements (list[list[int]]): Danh sách yêu cầu tài nguyên của từng node trong các đồ thị con.
    
    Returns:
    list[nx.DiGraph]: Danh sách các đồ thị con đã tạo.
    """
    return [create_subgraph(size, req) for size, req in zip(sizes, requirements)]

def build_ilp_problem(GS: list[nx.DiGraph], N: nx.DiGraph) -> pl.LpProblem:
    """
    Xây dựng bài toán ILP cho việc ánh xạ các đồ thị con vào mạng vật lý.
    
    Parameters:
    GS (list[nx.DiGraph]): Danh sách các đồ thị con.
    N (nx.DiGraph): Đồ thị mạng vật lý.
    
    Returns:
    pl.LpProblem: Bài toán ILP đã xây dựng.
    """
    problem = pl.LpProblem(name='Graph-Mapping', sense=pl.LpMaximize)

    # Decision variables for nodes and edges
    xNode = pl.LpVariable.dicts("xNode",
                          ((s, n, i)
                          for s, subgraph in enumerate(GS)
                          for n in subgraph.nodes
                          for i in N.nodes),
                          cat=pl.LpBinary
    )

    xEdge = pl.LpVariable.dicts("xEdge", 
                                ((s, w, v, i, j) 
                                 for s, subgraph in enumerate(GS) 
                                 for w, v in subgraph.edges 
                                 for i, j in N.edges),
                                cat=pl.LpBinary)

    pi = pl.LpVariable.dicts("pi", (s for s in range(len(GS))), cat=pl.LpBinary)
    phi = pl.LpVariable.dicts("phi", 
                              ((s, k) 
                               for s in range(len(GS)) 
                               for k in N.nodes),
                              cat=pl.LpBinary)
    z = pl.LpVariable.dicts("z", 
                            ((s, k) 
                             for s in range(len(GS)) 
                             for k in N.nodes),
                            cat=pl.LpBinary)

    # Attribute of the target graph
    aNode = nx.get_node_attributes(N, "a")
    aEdge = nx.get_edge_attributes(N, "a")

    # Constraints
    for s, subgraph in enumerate(GS):
        rNode = nx.get_node_attributes(subgraph,"r")
        rEdge = nx.get_edge_attributes(subgraph,"r")

        # C1 constraint
        for i in N.nodes:
            problem += (
                pl.lpSum(
                    xNode[(s, n, i)] * rNode[n]
                    for n in subgraph.nodes
                ) <= aNode[i] * pl.lpSum(phi[(s, k)] for k in N.nodes),
                f'C1_{s}_{i}'
            )

        # C2 constraint
        for i, j in N.edges:
            problem += (
                pl.lpSum(
                    xEdge[(s, w, v, i, j)] * rEdge[(w, v)]
                    for w, v in subgraph.edges
                ) <= aEdge[(i, j)] * pl.lpSum(phi[(s, k)] for k in N.nodes),
                f'C2_{s}_{i}_{j}'
            )

        # C3 constraint
        for i in N.nodes:
            for k in N.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, n, i)]
                        for n in subgraph.nodes
                    ) <= z[(s, k)],
                    f'C3_{s}_{i}_{k}'
                )

        # C4 constraint
        for n in subgraph.nodes:
            for k in N.nodes:
                problem += (    
                    pl.lpSum(
                        xNode[(s, n, i)]
                        for i in N.nodes
                    ) == z[(s, k)],
                    f'C4_{s}_{n}_{k}'
                )

        # C5 constraints
        big_M = 100  # Define a sufficiently large value for M
        for (w, v) in subgraph.edges:
            for k in N.nodes:
                for i in N.nodes:
                    for j in N.nodes:
                        if i != j:
                            if (s, w, v, i, j) in xEdge and (s, w, v, j, i) in xEdge:
                                problem += (
                                    xEdge[(s, w, v, i, j)] - xEdge[(s, w, v, j, i)] 
                                    - (xNode[(s, v, i)] - xNode[(s, w, i)]) <= big_M * (1 - phi[(s, k)]),
                                    f'C5_{s}_{w}_{v}_{i}_{j}_{k}_1'
                                )
                                problem += (
                                    xEdge[(s, w, v, i, j)] - xEdge[(s, w, v, j, i)] 
                                    - (xNode[(s, v, i)] - xNode[(s, w, i)]) >= -big_M * (1 - phi[(s, k)]),
                                    f'C5_{s}_{w}_{v}_{i}_{j}_{k}_2'
                                )
    
    # C6 constraint
    for s in range(len(GS)):
        problem += (
            pl.lpSum(phi[(s, k)] for k in N.nodes) == pi[s],
            f'C6_{s}'
        )
    
    # C7 constraint
    for s in range(len(GS)):
        for k in N.nodes:
            problem += (
                z[(s, k)] <= pi[s],
                f'C7_{s}_{k}_1'
            )
            problem += (
                z[(s, k)] <= phi[(s, k)],
                f'C7_{s}_{k}_2'
            ) 
            problem += (
                z[(s, k)] <= pi[s] - phi[(s, k)] - 1,
                f'C7_{s}_{k}_3'
            )

    # Objective function
    problem += pl.lpSum(pi[s] for s in range(len(GS)))

    return problem

def main():
    # Tạo đồ thị mạng vật lý với yêu cầu tài nguyên cho từng node
    N_requirements = [100, 100, 100, 100, 100]
    N = create_physical_network(5, N_requirements)
    
    # Tạo tập hợp các đồ thị con với yêu cầu tài nguyên cho từng node
    sizes = [3, 3, 3]
    requirements = [
        [10, 20, 40],
        [10, 20, 30],
        [10, 10, 10]
    ]
    GS = create_subgraph_set(3, sizes, requirements)
    
    # Xây dựng bài toán ILP
    problem = build_ilp_problem(GS, N)
    
    print(problem)
    
    # Giải bài toán
    result = problem.solve()
    print(pl.LpStatus[result])
    
if __name__ == '__main__':
    main()