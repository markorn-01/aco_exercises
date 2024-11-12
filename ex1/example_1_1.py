#!/usr/bin/env python3
#
# Author: Stefan Haller <stefan.haller@iwr.uni-heidelberg.de>
#
# You do not have to edit this example file (at least for Exercise 1.1). Put
# your implementation of the function `evaluate_energy` into a file called
# `exercise_1.py` in the same directory and simply run this script to get the
# results for the example graph provided on the exercise sheet.

import exercise_1 as student

from collections import namedtuple


Node = namedtuple('Node', 'costs')
Edge = namedtuple('Edge', 'left right costs')


def potts(x_u, x_v, l=1):
    return 0 if x_u == x_v else l


def make_potts(shape, l = 1):
    c = {}
    for x_u in range(shape[0]):
        for x_v in range(shape[1]):
            c[x_u, x_v] = potts(x_u, x_v, l)
    return c


def make_graph(l = 1):
    nodes = [Node(costs=[0.3, 0.9, 0.4]),
             Node(costs=[0.8, 0.1, 0.3]),
             Node(costs=[0.2, 0.5])]

    edges = []
    for u, v in ((0, 1), (0, 2), (1, 2)):
        shape = tuple(len(nodes[x].costs) for x in (u, v))
        edges.append(Edge(left=u, right=v, costs=make_potts(shape, l)))

    return nodes, edges


def run_example():
    print('# Exercise 1.1')
    for l in (0, 0.2, 0.5):
        nodes, edges = make_graph(l)
        for assignment in ((0, 0, 0), (0, 2, 1)):
            e = student.evaluate_energy(nodes, edges, assignment)
            print('x={} & lambda={:.1f}  =>  E={}'.format(assignment, l, e))
        print()

    # You can comment in the following code to use the same example graph for
    # Exercise 1.2.
    #
    print('# Exercise 1.2')
    for l in (0, 0.2, 0.5):
       nodes, edges = make_graph(l)
       print('lambda={:.1f}  =>  bruteforce returns {}'.format(
           l, student.bruteforce(nodes, edges)))
       
    
    # Exercise 1.3: Chain structure
    print('# Exercise 1.3')
    for l in (0, 0.2, 0.5):
        nodes, edges = make_graph(l)
        
        # Remove the edge (v1, v3) to create a chain structure (v1 - v2 - v3)
        edges = [edge for edge in edges if not (edge.left == 0 and edge.right == 2)]
        
        F, ptr = student.dynamic_programming(nodes, edges)
        optimal_assignment = student.backtrack(nodes, edges, F, ptr)
        optimal_energy = student.evaluate_energy(nodes, edges, optimal_assignment)
        
        print('lambda={:.1f}  =>  Optimal assignment: {} with energy: {}'.format(
            l, optimal_assignment, optimal_energy))

if __name__ == '__main__':
    run_example()
