import networkx as nx
import pulp as pl
from common import *
from Validate import *

def main():
    N_requirements = [10, 10, 10, 10, 10]
    E_requirements = [5, 5, 5, 5]
    PHY = create_physical_network(5, N_requirements, E_requirements)

    sizes = [
        [3, 2],
        [2, 4],
        [3, 3]
    ]
    node_requirements = [
        [[3, 3, 3], [2, 2]],
        [[4, 4], [2, 2, 2, 2]],
        [[1, 1, 1], [2, 2, 2]]
    ]
    edge_requirements = [
        [[2, 2], [1]],
        [[3], [1, 1, 1]],
        [[2, 2], [1, 1]]
    ]

    slices = []
    for slice_id in range(3):
        slice_configs = create_slice_configurations(len(sizes[slice_id]), sizes[slice_id], node_requirements[slice_id], edge_requirements[slice_id])
        slices.append(slice_configs)

    ilp_problem = build_ilp_problem(slices, PHY)

    ilp_problem.solve()

    ilp_problem.writeLP("FullFlexSolution.lp")

    save_solution(ilp_problem, 'problem.pkl')
    loaded_problem = load_solution('problem.pkl')
    result = check_solution(loaded_problem, slices, PHY)
    print(f'Constraint satisfied: {result}')
    
    if result == True:
        for var in ilp_problem.variables():
            print(f'{var.name}: {var.varValue}')
        print(f'Optimal value: {pl.value(ilp_problem.objective)}')
        print(pl.LpStatus[ilp_problem.status])
    
if __name__ == '__main__':
    main()