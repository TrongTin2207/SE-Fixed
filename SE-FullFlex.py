import networkx as nx 
import pulp as pl

def create_physical_network(num_nodes: int) -> nx.DiGraph:
    '''
    Create a physical network with large amount of nodes and links
    
    Parameters:
    num_nodes (int): Number of nodes in the physical network
    
    Returns:
    nx.DiGraph: The created physical network
    '''
    N = nx.DiGraph()
    for i in range (num_nodes):
        N.add_node(i, a=10)
    for i in range (num_nodes - 1):
        N.add_edge(i, i+1, a = 10)
        N.add_edge(i+1, i, a = 10)
    return N

def create_subgraph(n: int) -> nx.DiGraph:
    """
    Create a number of subgraph with provided amount
    
    Parameters:
    num_nodes (int): The number of nodes in subgraph.
    
    Returns:
    nx.DiGraph: The created subgraphs.
    """
    GS = nx.DiGraph()
    for i in range (n):
        GS.add_node(i, r=2)
    for i in range (n-1):
        GS.add_edge(i, i+1, r=5)
    return GS

def create_subgraph_set(K: int, sizes: list[int]) -> list[nx.DiGraph]:
    '''
    Create a set of K subgraphs with different sizes
    
    Parameters:
    k (int): Number of subgraphs needs to be created
    sizes (list[int]): List of different sizes for subgraphs
    '''
    return [create_subgraph(size) for size in sizes]

    

def build_ilp_problem(GS: list[nx.DiGraph], N: nx.DiGraph) -> pl.LpProblem:
    '''
    Create an ILP problem for mapping subgraphs to the physical network
    
    Parameters:
    GS (list[nx.DiGraph]): List of subgraphs
    N (nx.DiGraph): Physical network
    
    Returns:
    pl.LpProblem: ILP
    '''
#Create the ILP problem
    problem = pl.LpProblem(name='Graph-Mapping', sense = pl.LpMaximize)

#Decision variables for nodes and edges
    xNode = pl.LpVariable.dicts("xNode",
                      ((s, n, i)
                      for s, subgraph in enumerate(GS)
                      for n in subgraph.nodes
                      for i in N.nodes),
                      cat = pl.LpBinary
    )

    xEdge = pl.LpVariable.dicts("xEdge", 
                            ((s, w, v, i, j) 
                             for s, subgraph in enumerate(GS) 
                             for w, v in subgraph.edges 
                             for i, j in N.edges),
                            cat=pl.LpBinary)

    pi = pl.LpVariable.dicts("pi", (s for s in range(len(GS))), cat = pl.LpBinary)
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

    
    #C2 constraint
        for i, j in N.edges:
            problem += (
                pl.lpSum(
                    xEdge[(s, w, v, i, j)] * rEdge[(w, v)]
                    for w, v in subgraph.edges
                ) <= aEdge[(i, j)] * pl.lpSum(phi[(s, k)] for k in N.nodes),
                f'C2_{s}_{i}_{j}'
        )
        
    #C3 constraint
        for i in N.nodes:
            for k in N.nodes:
                problem += (
                    pl.lpSum(
                        xNode[(s, n, i)]
                        for n in subgraph.nodes
                    ) <= z[(s, k)],
                    f'C3_{s}_{i}_{k}'
        )
    
    #C4 constraint
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
    #C6 constraint
    for s in range(len(GS)):
        problem += (
            pl.lpSum(phi[(s, k)] for k in N.nodes) == pi[s],
            f'C6_{s}'
        )
    
    #C7 constraint
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
    #Create physical network
    N = create_physical_network(5)
    
    #Create a set of subgraphs
    size =[3, 4, 2]
    GS = create_subgraph_set(3, size)
    
    #Create an ILP problem
    problem = build_ilp_problem(GS, N)
    print(problem)
    
    #Solve the problem
    result = problem.solve()
    print(pl.LpStatus[result])
    
    for v in problem.variables():
        print(v.name, "=", v.varValue)
        
if __name__ == "__main__":
    main()
    
