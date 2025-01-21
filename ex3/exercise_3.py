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
    # # Iterate through edges in the subproblem
    # # Collect all unique nodes in the subproblem
    # unique_nodes = {node for edge in subproblem for node in (edge.left, edge.right)}

    # # Update nodes in a consistent order (e.g., sorted by node index)
    # for u in sorted(unique_nodes):
    #     icm_update_step(nodes, edges, grid, assignment, u)
    # sequence of nodes from the subproblem edges
    chain_nodes = []
    edge = subproblem[0]  
    chain_nodes.extend([edge.left, edge.right])
    
    # add remaining nodes from subsequent edges
    for edge in subproblem[1:]:
        chain_nodes.append(edge.right)
        
    n = len(chain_nodes)  
    L = len(nodes[chain_nodes[0]].costs)  
    
    dp = [[float('inf')] * L for _ in range(n)]
    backtrack = [[0] * L for _ in range(n)]
    
    # initialize first node
    u = chain_nodes[0]
    for s in range(L):
        dp[0][s] = nodes[u].costs[s]
        
        # costs from neighbors outside chain
        for v in grid.neighbors(u):
            if v not in chain_nodes:
                if v < u:
                    edge = grid.edge(v, u)
                    dp[0][s] += edge.costs[(assignment[v], s)]
                else:
                    edge = grid.edge(u, v) 
                    dp[0][s] += edge.costs[(s, assignment[v])]

    for i in range(1, n):
        u = chain_nodes[i]  
        v = chain_nodes[i-1]  
        
        # get connecting edge
        edge = grid.edge(v, u) if v < u else grid.edge(u, v)
        
        for s in range(L): 
            # initialize with unary cost
            base_cost = nodes[u].costs[s]
            
            # add costs from neighbors outside chain
            for w in grid.neighbors(u):
                if w not in chain_nodes:
                    if w < u:
                        e = grid.edge(w, u)
                        base_cost += e.costs[(assignment[w], s)]
                    else:
                        e = grid.edge(u, w)
                        base_cost += e.costs[(s, assignment[w])]
            
            # best previous label
            for t in range(L):  #
                # add pairwise cost based on edge direction
                pair_cost = edge.costs[(t, s)] if v < u else edge.costs[(s, t)]
                total_cost = dp[i-1][t] + base_cost + pair_cost
                
                if total_cost < dp[i][s]:
                    dp[i][s] = total_cost
                    backtrack[i][s] = t
    

    # optimal label for last node
    curr_label = min(range(L), key=lambda s: dp[n-1][s])
    assignment[chain_nodes[n-1]] = curr_label
    
    # trace back
    for i in range(n-2, -1, -1):
        curr_label = backtrack[i+1][curr_label]
        assignment[chain_nodes[i]] = curr_label

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
    edge = edges[edge_idx]
    u, v = edge.left, edge.right
    
    # find locally optimal labels
    # for node u
    min_u = float('inf')
    best_u = 0
    for s in range(len(nodes[u].costs)):
        cost = nodes[u].costs[s]
        if cost < min_u:
            min_u = cost
            best_u = s
            
    # for node v
    min_v = float('inf')
    best_v = 0
    for t in range(len(nodes[v].costs)):
        cost = nodes[v].costs[t]
        if cost < min_v:
            min_v = cost
            best_v = t
            
    # for edge uv
    min_uv = float('inf')
    best_s = best_t = 0
    for s in range(len(nodes[u].costs)):
        for t in range(len(nodes[v].costs)):
            cost = edge.costs[(s, t)]
            if cost < min_uv:
                min_uv = cost
                best_s = s
                best_t = t
                
    # compute subgradient according to equation (6.31)
    left_grad = [0] * len(nodes[u].costs)  
    right_grad = [0] * len(nodes[v].costs)
    
    # for each label of left node
    for s in range(len(nodes[u].costs)):
        if s == best_u and s != best_s:
            left_grad[s] = -1
        elif s != best_u and s == best_s:
            left_grad[s] = 1
            
    # for each label of right node  
    for t in range(len(nodes[v].costs)):
        if t == best_v and t != best_t:
            right_grad[t] = -1
        elif t != best_v and t == best_t:
            right_grad[t] = 1
            
    return (left_grad, right_grad)


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
    for edge in edges:
        u, v = edge.left, edge.right
        
        # get subgradients for both directions
        phi_uv = subgradient[(u,v)]  # φu,v
        phi_vu = subgradient[(v,u)]  # φv,u
        
        # update node costs
        for s in range(len(nodes[u].costs)):
            nodes[u].costs[s] -= stepsize * phi_uv[s]
            
        for t in range(len(nodes[v].costs)):
            nodes[v].costs[t] -= stepsize * phi_vu[t]
        
        # update edge costs
        for s in range(len(nodes[u].costs)):
            for t in range(len(nodes[v].costs)):
                edge.costs[(s,t)] += stepsize * (phi_uv[s] + phi_vu[t])

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
    
    label = min_sum_diffusion_round_primal(nodes, edges, grid, u)
    
    min_sum_diffusion_distribution(nodes, edges, grid, u)
    
    return label

def min_sum_diffusion_single_iteration(nodes, edges, grid):
    # Implement a single iteration for the Min-Sum diffusion method.
    # Iterate over all nodes and perform the update step on them.
    # Return the assignment/labeling for the full model.
    assignment = []
    for u in range(len(nodes)):
        label = min_sum_diffusion_update_step(nodes, edges, grid, u)
        assignment.append(label)
    return assignment

def min_sum_diffusion_method(nodes, edges, grid):
    # Implement the Min-Sum diffusion method (run multiple iterations).
    # Return the assignment/labeling of the full model.
    max_iterations = 100
    for iteration in range(max_iterations):
        min_sum_diffusion_single_iteration(nodes, edges, grid)
    
    return min_sum_diffusion_round_primal(nodes, edges, grid)