#!/usr/bin/env python3
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>

import exercise_2 as student

from collections import namedtuple


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


def make_pairwise(shape):
    c = {}
    for x_u in range(shape[0]):
        for x_v in range(shape[1]):
            c[x_u, x_v] = 1 if x_u == x_v else 0
    return c


def make_graph():
    nodes = [Node(costs=[0.5, 0.5]),
             Node(costs=[0.0, 0.0]),
             Node(costs=[0.2, 0.2])]

    edges = []
    for u, v in ((0, 1), (0, 2), (1, 2)):
        shape = tuple(len(nodes[x].costs) for x in (u, v))
        edges.append(Edge(left=u, right=v, costs=make_pairwise(shape)))

    return nodes, edges


def run_example():
    nodes, edges = make_graph()
    lp, x = student.convert_to_lp(nodes, edges)
    res = lp.solve()
    assert res

    # Print the LP solution
    print("LP Relaxation Results:")
    for i, node in enumerate(nodes):
        print(f"Node {i}: {[x[i, k].value() for k in range(len(node.costs))]}")

    # Set up and solve the ILP
    ilp = student.convert_to_ilp(nodes, edges)
    ilp.solve()

    # Print the ILP solution
    print("\nILP Solution Results:")
    ilp_labeling = student.ilp_to_labeling(nodes, edges, ilp)
    print(f"Node Labelings: {ilp_labeling}")

    for var in lp.variables():
        print('{} -> {}'.format(var.name, var.value()))


if __name__ == '__main__':
    run_example()

'''
Since the costs for each state of a node are identical, the LP relaxation has no preference
and assigns a fractional value of 0.5 to each state. As a result, all auxiliary variables y 
are set to fractional values as allowed by the Sherali-Adams constraints, reflecting the fractional
assignments of the nodes.

The naive rounding of the LP solution assigns all nodes to the first (0th) state without 
considering edge penalties, leading to a suboptimal integer solution.

In contrast, the ILP enforces binary assignments for the x variables and directly incorporates
edge penalties, leading to the optimal solution. This solution balances both node and edge costs
and in this case, assigns the states [1, 1, 0], accounting for the penalties effectively.
'''