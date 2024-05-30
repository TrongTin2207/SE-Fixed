
import networkx as nx
import pulp as pl

def createGraph(n):
    GS = nx.DiGraph()
    for i in range(n):
        GS.add_node(i, r=2)
    for i in range(n-1):
        GS.add_edge(i, i+1, r=5)
    return GS

# Create target graph N
N = nx.DiGraph()
for i in range(5):
    N.add_node(i, a=10)

for i in range(4):
    N.add_edge(i, i+1, a=10)
    N.add_edge(i+1, i, a=10)

# Create subgraph
GS_1 = createGraph(3)
GS_2 = createGraph(4)
GS_3 = createGraph(2)

GS = [GS_1, GS_2, GS_3]

# ILP Problem
problem = pl.LpProblem(name="graph-mapping", sense=pl.LpMaximize)

# Decision variables for nodes and edges
xNode = pl.LpVariable.dicts("xNode", 
                            ((s, n, i) 
                             for s, subgraph in enumerate(GS) 
                             for n in subgraph.nodes 
                             for i in N.nodes),
                            cat=pl.LpBinary)

xEdge = pl.LpVariable.dicts("xEdge", 
                            ((s, w, v, i, j) 
                             for s, subgraph in enumerate(GS) 
                             for w, v in subgraph.edges 
                             for i, j in N.edges),
                            cat=pl.LpBinary)

pi = pl.LpVariable.dicts("pi", (s for s in range(len(GS))), cat=pl.LpBinary)

# Attributes of the target graph
aNode = nx.get_node_attributes(N, "a")
aEdge = nx.get_edge_attributes(N, "a")

# Constraints
for s, subgraph in enumerate(GS):
    rNode = nx.get_node_attributes(subgraph, "r")
    rEdge = nx.get_edge_attributes(subgraph, "r")

    # C1 constraints
    for i in N.nodes:
        problem += (
            pl.lpSum(
                xNode[(s, n, i)] * rNode[n]
                for n in subgraph.nodes
            ) <= aNode[i],
            f'C1_{s}_{i}'
        )

    # C2 constraints
    for i, j in N.edges:
        problem += (
            pl.lpSum(
                xEdge[(s, w, v, i, j)] * rEdge[(w, v)]
                for (w, v) in subgraph.edges
            ) <= aEdge[(i, j)],
            f'C2_{s}_{i}_{j}'
        )

    # C3 constraints
    for i in N.nodes:
        problem += (
            pl.lpSum(
                xNode[(s, n, i)]
                for n in subgraph.nodes
            ) <= pi[s],
            f'C3_{s}_{i}'
        )

    # C4 constraints
    for n in subgraph.nodes:
        problem += (
            pl.lpSum(
                xNode[(s, n, i)]
                for i in N.nodes
            ) == pi[s],
            f'C4_{s}_{n}'
        )

    # C5 constraints
    for (w, v) in subgraph.edges:
        for i in N.nodes:
            for j in N.nodes:
                if i != j:
                    if (s, w, v, i, j) in xEdge and (s, w, v, j, i) in xEdge:
                        problem += (
                            pl.lpSum(
                                xEdge[(s, w, v, i, j)]
                            ) - pl.lpSum(
                                xEdge[(s, w, v, j, i)]
                            ) == xNode[(s, v, i)] - xNode[(s, w, i)],
                            f'C5_{s}_{w}_{v}_{i}_{j}'
                        )

# Objective function
problem += pl.lpSum(pi[s] for s in range(len(GS)))

print(problem)

# Solve the problem
result = problem.solve()
print(pl.LpStatus[result])

# Output
for v in problem.variables():
    print(v.name, "=", v.varValue)