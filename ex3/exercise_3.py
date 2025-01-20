#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Quang Minh Ngo <ve330@stud.uni-heidelberg.de>
# Author: Taha Erkoc <nx324@stud.uni-heidelberg.de>
#
# IMPORTANT: You should use the following function definitions without applying
# any changes to the interface, because the function will be used for automated
# checks. It is fine to declare additional helper functions.


#
# Iterated Conditional Modes (ICM)
#
import numpy as np
from grid import *

def icm_update_step(nodes, edges, grid, assignment, u):
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # `u` is the index of the current node for which an update should be performed.
    # Task: Update the assignemnt for node `u`.
    # Return: Nothing.
    num_labels = len(nodes[u].costs)  # Number of possible labels for node u
    costs = np.zeros(num_labels)  # Store costs for each label

    for label_u in range(num_labels):
        # Unary cost for the label
        unary_cost = nodes[u].costs[label_u]
        
        # Pairwise costs from connected edges
        pairwise_cost = 0
        for edge in edges:
            if edge.left == u:
                neighbor = edge.right
                neighbor_label = assignment[neighbor]
                pairwise_cost += edge.costs.get((label_u, neighbor_label), 0)
            elif edge.right == u:
                neighbor = edge.left
                neighbor_label = assignment[neighbor]
                pairwise_cost += edge.costs.get((neighbor_label, label_u), 0)

        # Total cost
        costs[label_u] = unary_cost + pairwise_cost

    # Assign the label with the minimum cost
    assignment[u] = np.argmin(costs)


def icm_single_iteration(nodes, edges, grid, assignment):
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # Task: Perform a full iteration of all ICM update steps.
    # Return: Nothing.
    for u in range(len(nodes)):  # Iterate over node indices
        icm_update_step(nodes, edges, grid, assignment, u)


def icm_method(nodes, edges, grid):
    # Task: Run ICM algorithm until convergence.
    # Return: Assignment after converging.
    # Initialize a random assignment
    assignment = [np.argmin(node.costs) for node in nodes]  # Initial assignment

    max_iterations = 100
    for iteration in range(max_iterations):
        old_assignment = assignment.copy()
        icm_single_iteration(nodes, edges, grid, assignment)
        
        # If the assignment does not change, we have converged
        if assignment == old_assignment:
            break

    return assignment
#
# Block ICM
#

def block_icm_update_step(nodes, edges, grid, assignment, subproblem):
    # `grid` is the helper structure for the grid graph representation
    # of `nodes` and `edges`. See grid example for details.
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # `subproblem` is the current chain-structured subproblem.
    # Task: Update the assignemnt for the current suproblem.
    # Return: Nothing.
    # Iterate through edges in the subproblem
    # Collect all unique nodes in the subproblem
    unique_nodes = {node for edge in subproblem for node in (edge.left, edge.right)}

    # Update nodes in a consistent order (e.g., sorted by node index)
    for u in sorted(unique_nodes):
        icm_update_step(nodes, edges, grid, assignment, u)


def block_icm_single_iteration(nodes, edges, grid, assignment):
    # Similar to ICM but you should iterate over all subproblems in the row
    # column decomposition.
    decomposition = row_column_decomposition(grid)

    for subproblem in decomposition:
        block_icm_update_step(nodes, edges, grid, assignment, subproblem)


def block_icm_method(nodes, edges, grid):
    # Task: Run the block ICM method until convergence.
    assignment = [np.argmin(node.costs) for node in nodes]  # Initial assignment

    max_iterations = 100
    for iteration in range(max_iterations):
        old_assignment = assignment.copy()
        block_icm_single_iteration(nodes, edges, grid, assignment)
        
        # If the assignment does not change, we have converged
        if assignment == old_assignment:
            break

    return assignment

#
# Subgradient
#

def subgradient_compute_single_subgradient(nodes, edges, grid, edge_idx):
    # Task: Compute the subgradient for given edge of index `edge_idx`.
    # Let n be the number of left labels and m be the number of right
    # labels, the subgradient has n + m elements.
    # Return: A tuple where the first term is a list of the n left
    # subgradient values and the second term is a list of the m right
    # subgradient values.
        # Get the edge and its endpoints
    edge = edges[edge_idx]
    u, v = edge.left, edge.right

    # Number of labels for nodes u and v
    num_labels_u = len(nodes[u].costs)
    num_labels_v = len(nodes[v].costs)

    # Initialize subgradient vectors for nodes u and v
    left_subgradient = [0] * num_labels_u
    right_subgradient = [0] * num_labels_v

    # Find minimizers for unary and pairwise terms
    min_label_u = min(range(num_labels_u), key=lambda s: nodes[u].costs[s])
    min_pairwise_label = min(
        ((s, t) for s in range(num_labels_u) for t in range(num_labels_v)),
        key=lambda pair: edge.costs.get(pair, 0)
    )

    s_u, s_pair_u = min_label_u, min_pairwise_label[0]

    # Compute subgradient for node u
    for s in range(num_labels_u):
        if s == s_u and s == s_pair_u:
            left_subgradient[s] = 0  # Gradient cancels
        elif s == s_u:
            left_subgradient[s] = -1
        elif s == s_pair_u:
            left_subgradient[s] = 1
        else:
            left_subgradient[s] = 0

    # Similarly, compute subgradient for node v
    s_v, s_pair_v = min_label_u, min_pairwise_label[1]
    for t in range(num_labels_v):
        if t == s_v and t == s_pair_v:
            right_subgradient[t] = 0
        elif t == s_v:
            right_subgradient[t] = -1
        elif t == s_pair_v:
            right_subgradient[t] = 1
        else:
            right_subgradient[t] = 0

    return left_subgradient, right_subgradient


def subgradient_compute_full_subgradient(nodes, edges, grid):
    # Task: Compute the full subgradient for the full problem.
    # Return dictionary with (u, v) => list of subgradient values.
    # Note: subgradient[u, v] is not identical to subgradient[v, u] (see book
    # where \phi_{u,v} is also not identical to \phi_{v,u}).
    subgradient = {}
    for edge_idx, edge in enumerate(edges):
        u, v = edge.left, edge.right
        left_grad, right_grad = subgradient_compute_single_subgradient(nodes, edges, grid, edge_idx)
        subgradient[(u, v)] = (left_grad, right_grad)
    return subgradient


def subgradient_apply_update(nodes, edges, grid, subgradient, stepsize):
    # Task: Reparametrize the model by modifying the costs (in direction of
    # `subgradient` multiplied by `stepsize`).
    for (u, v), (left_grad, right_grad) in subgradient.items():
        # Update the unary costs for node u
        for s in range(len(nodes[u].costs)):
            nodes[u].costs[s] -= stepsize * left_grad[s]

        # Update the unary costs for node v
        for t in range(len(nodes[v].costs)):
            nodes[v].costs[t] -= stepsize * right_grad[t]

def subgradient_update_step(nodes, edges, grid, stepsize):
    # Task: Compute subgradient for the problem and reparametrize the the
    # whole problem.
    # Compute the full subgradient
    subgradient = subgradient_compute_full_subgradient(nodes, edges, grid)

    # Apply the update
    subgradient_apply_update(nodes, edges, grid, subgradient, stepsize)


def subgradient_round_primal(nodes, edges, grid):
    # Task: Implement primal rounding as discussed in the lecture.
    # Return: Assignment (list that assigns each node one label).
    assignment = []
    for u in range(len(nodes)):
        # Assign the label with the minimum unary cost
        min_label = min(range(len(nodes[u].costs)), key=lambda s: nodes[u].costs[s])
        assignment.append(min_label)
    return assignment


def subgradient_method(nodes, edges, grid, iterations=100):
    # Task: Run subgradient method for given number of iterations.
    # Step size should be computed by yourself.
    # Return: Assignment.
    # Initialize step size
    f_star = 0  # Placeholder for optimal dual value (if known)
    gamma = 0.1  # Decay factor for step size

    for t in range(1, iterations + 1):
        # Compute the step size as described in the lecture
        stepsize = max(f_star, 1.0 / (1 + gamma * t))
        # Perform one update step
        subgradient_update_step(nodes, edges, grid, stepsize)

    # Round the primal solution
    return subgradient_round_primal(nodes, edges, grid)


#
# Min-Sum Diffusion
#

def min_sum_diffusion_accumulation(nodes, edges, grid, u):
    # Task: Implement the reparametrization for the accumulation phase of N:
    for neighbor in grid.neighbors(u):
        edge = grid.edge(u, neighbor) if u < neighbor else grid.edge(neighbor, u)
    for label in range(len(nodes[u].costs)):
        min_cost = min(edge.costs[(label, v_label)] for v_label in range(len(nodes[neighbor].costs)))
        nodes[u].costs[label] += min_cost
        for v_label in range(len(nodes[neighbor].costs)):
            edge.costs[(label, v_label)] -= min_cost


def min_sum_diffusion_distribution(nodes, edges, grid, u):
    # See accumulation, but for distribution.
    num_neighbors = len(grid.neighbors(u))
    for neighbor in grid.neighbors(u):
        edge = grid.edge(u, neighbor) if u < neighbor else grid.edge(neighbor, u)
        for label in range(len(nodes[u].costs)):
            redistribution = nodes[u].costs[label] / num_neighbors
            nodes[u].costs[label] -= redistribution
            for v_label in range(len(nodes[neighbor].costs)):
                edge.costs[(label, v_label)] += redistribution


def min_sum_diffusion_round_primal(nodes, edges, grid, u):
    # Implement the rounding technique as discussed in the lecture for the
    # given node `u`.
    assignment = []
    for u in range(len(nodes)):
        label = np.argmin(nodes[u].costs)  # Select label with minimal cost
        assignment.append(label)
    return assignment


def min_sum_diffusion_update_step(nodes, edges, grid, u):
    # Implement a single update step for the given node `u` (accumulation,
    # rounding, distribution).
    # Return the assignment/label for node `u`.
    min_sum_diffusion_accumulation(nodes, edges, grid, u)
    min_sum_diffusion_distribution(nodes, edges, grid, u)


def min_sum_diffusion_single_iteration(nodes, edges, grid):
    # Implement a single iteration for the Min-Sum diffusion method.
    # Iterate over all nodes and perform the update step on them.
    # Return the assignment/labeling for the full model.
    for u in range(len(nodes)):
        min_sum_diffusion_update_step(nodes, edges, grid, u)


def min_sum_diffusion_method(nodes, edges, grid):
    # Implement the Min-Sum diffusion method (run multiple iterations).
    # Return the assignment/labeling of the full model.
    max_iterations = 100
    for iteration in range(max_iterations):
        min_sum_diffusion_single_iteration(nodes, edges, grid)
    
    return min_sum_diffusion_round_primal(nodes, edges, grid)