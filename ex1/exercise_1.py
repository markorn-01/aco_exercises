#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>


# For exercise 1.1
def evaluate_energy(nodes, edges, assignment):
    unary_costs = 0.0
    pairwise_costs = 0.0
    for i, y in enumerate(assignment):
        unary_costs += nodes[i].costs[y]
    
    for edge in edges:
        u, v = edge.left, edge.right
        y_u, y_v = assignment[u], assignment[v]
        pairwise_costs += edge.costs[y_u, y_v]
        
    return unary_costs + pairwise_costs


# For exercise 1.2
# def bruteforce(nodes, edges):
#     assignment = [0] * len(nodes)
#     # TODO: implement brute-force algorithm here...
#     energy = evaluate_energy(nodes, edges, assignment)
#     return (assignment, energy)

def generate_assignments(nodes, current_assignment, index, all_assignments):
    if index == len(nodes):
        # If we've assigned labels to all nodes, save the assignment
        all_assignments.append(current_assignment[:])  # Copy the list to avoid mutation
        return
    
    # For each possible label of the current node, recurse
    for label in range(len(nodes[index].costs)):
        current_assignment[index] = label
        generate_assignments(nodes, current_assignment, index + 1, all_assignments)

# For exercise 1.2
def bruteforce(nodes, edges):
    # List to store all possible assignments
    all_assignments = []
    
    # Start recursive generation of assignments
    generate_assignments(nodes, [0] * len(nodes), 0, all_assignments)
    
    # Initialize minimum energy and best assignment
    energy = float('inf')
    assignment = None
    
    # Iterate over all possible assignments
    for guessed_assignment in all_assignments:
        # Calculate energy for the current assignment
        calculated_energy = evaluate_energy(nodes, edges, guessed_assignment)
        
        # Check if this is the minimum energy found so far
        if calculated_energy < energy:
            energy = calculated_energy
            assignment = guessed_assignment[:]
    
    # Return the best assignment and its corresponding energy
    return (assignment, energy)

# For exercise 1.3
# def dynamic_programming(nodes, edges):
#     F, ptr = None, None
#     return F, ptr

# def backtrack(nodes, edges, F, ptr):
#     assignment = [0] * len(nodes)
#     return assignment

# For exercise 1.3
import numpy as np

def dynamic_programming(nodes, edges, lambda_penalty):
    """
    Forward pass of the dynamic programming algorithm to compute the Bellman functions.
    
    Parameters:
        nodes (list of lists): Each sublist represents the node potential values for a node in the chain.
        edges (list of tuples): Each tuple represents an edge (u, v) where `u` and `v` are adjacent nodes in the chain.
        lambda_penalty (float): Penalty applied when adjacent nodes have different labels.
        
    Returns:
        F (list of dicts): Bellman functions where F[i][x] gives the minimum cost for node `i` with label `x`.
        ptr (list of dicts): Pointer table where ptr[i][x] points to the optimal label for node `i+1` given label `x` at node `i`.
    """
    num_nodes = len(nodes)
    F = [{} for _ in range(num_nodes)]
    ptr = [{} for _ in range(num_nodes - 1)]
    
    # Initialize the last node's Bellman function
    for x in range(len(nodes[-1].costs)):
        F[-1][x] = nodes[-1].costs[x]

    # Compute Bellman functions backwards from the second-last node
    for i in range(num_nodes - 2, -1, -1):
        for x in range(len(nodes[i].costs)):
            min_cost = float('inf')
            best_label = None
            for y in range(len(nodes[i + 1])):
                cost = nodes[i][x] + edges.costs[x, y] + F[i + 1][y]
                if cost < min_cost:
                    min_cost = cost
                    best_label = y
            F[i][x] = min_cost
            ptr[i][x] = best_label  # Store the best label for backtracking

    return F, ptr


def backtrack(nodes, edges, F, ptr):
    """
    Backtracking pass to recover the optimal label assignment from Bellman functions and pointers.
    
    Parameters:
        nodes (list of lists): Each sublist represents the node potential values for a node in the chain.
        edges (list of tuples): Each tuple represents an edge (u, v) where `u` and `v` are adjacent nodes in the chain.
        F (list of dicts): Bellman functions computed in the forward pass.
        ptr (list of dicts): Pointer table for tracking optimal labels.
        
    Returns:
        assignment (list): Optimal label assignment for each node.
    """
    num_nodes = len(nodes)
    assignment = [0] * num_nodes
    
    # Start from the optimal label at the first node
    assignment[0] = min(F[0], key=F[0].get)
    
    # Follow pointers to determine the rest of the labels
    for i in range(1, num_nodes):
        assignment[i] = ptr[i - 1][assignment[i - 1]]
    
    return assignment
