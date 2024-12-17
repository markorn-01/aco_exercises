#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Taha Erkoc <nx324@stud.uni-heidelberg.de>
# Author: Quang Minh Ngo <ve330@stud.uni-heidelberg.de>


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

def generate_assignments(nodes, current_assignment, index, all_assignments):
    if index == len(nodes):
        # If we've assigned labels to all nodes, save the assignment
        all_assignments.append(current_assignment[:])  # Copy the list to avoid mutation
        return
    
    # For each possible label of the current node, recurse
    for label in range(len(nodes[index].costs)):
        current_assignment[index] = label
        generate_assignments(nodes, current_assignment, index + 1, all_assignments)

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

def dynamic_programming(nodes, edges):
    # Initialize Bellman functions and pointers
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
            for y in range(len(nodes[i + 1].costs)):
                # Calculate cost for the edge and next node
                cost = nodes[i].costs[x] + edges[i].costs[x, y] + F[i + 1][y]
                if cost < min_cost:
                    min_cost = cost
                    best_label = y
            F[i][x] = min_cost
            ptr[i][x] = best_label  # Store the best label for backtracking

    return F, ptr

def backtrack(nodes, edges, F, ptr):
    # Initialize assignment list
    num_nodes = len(nodes)
    assignment = [0] * num_nodes
    
    # Start from the optimal label at the first node
    assignment[0] = min(F[0], key=F[0].get)
    
    # Follow pointers to determine the rest of the labels
    for i in range(1, num_nodes):
        assignment[i] = ptr[i - 1][assignment[i - 1]]
    
    return assignment

