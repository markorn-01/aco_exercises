#!/usr/bin/env python3
# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>
#
# IMPORTANT: You should use the following function definitions without applying
# any changes to the interface, because the function will be used for automated
# checks. It is fine to declare additional helper functions.


#
# Iterated Conditional Modes (ICM)
#

def icm_update_step(nodes, edges, grid, assignment, u):
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # `u` is the index of the current node for which an update should be performed.
    # Task: Update the assignemnt for node `u`.
    # Return: Nothing.
    pass


def icm_single_iteration(nodes, edges, grid, assignment):
    # `assignment` is the current labeling (list that assigns a label to
    # each node).
    # Task: Perform a full iteration of all ICM update steps.
    # Return: Nothing.
    pass


def icm_method(nodes, edges, grid):
    # Task: Run ICM algorithm until convergence.
    # Return: Assignment after converging.
    pass


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
    pass


def block_icm_single_iteration(nodes, edges, grid, assignment):
    # Similar to ICM but you should iterate over all subproblems in the row
    # column decomposition.
    pass


def block_icm_method(nodes, edges, grid):
    # Task: Run the block ICM method until convergence.
    pass


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
    pass


def subgradient_compute_full_subgradient(nodes, edges, grid):
    # Task: Compute the full subgradient for the full problem.
    # Return dictionary with (u, v) => list of subgradient values.
    # Note: subgradient[u, v] is not identical to subgradient[v, u] (see book
    # where \phi_{u,v} is also not identical to \phi_{v,u}).
    pass


def subgradient_apply_update(nodes, edges, grid, subgradient, stepsize):
    # Task: Reparametrize the model by modifying the costs (in direction of
    # `subgradient` multiplied by `stepsize`).
    pass


def subgradient_update_step(nodes, edges, grid, stepsize):
    # Task: Compute subgradient for the problem and reparametrize the the
    # whole problem.
    pass


def subgradient_round_primal(nodes, edges, grid):
    # Task: Implement primal rounding as discussed in the lecture.
    # Return: Assignment (list that assigns each node one label).
    pass


def subgradient_method(nodes, edges, grid, iterations=100):
    # Task: Run subgradient method for given number of iterations.
    # Step size should be computed by yourself.
    # Return: Assignment.
    pass


#
# Min-Sum Diffusion
#

def min_sum_diffusion_accumulation(nodes, edges, grid, u):
    # Task: Implement the reparametrization for the accumulation phase of N:
    pass


def min_sum_diffusion_distribution(nodes, edges, grid, u):
    # See accumulation, but for distribution.
    pass


def min_sum_diffusion_round_primal(nodes, edges, grid, u):
    # Implement the rounding technique as discussed in the lecture for the
    # given node `u`.
    pass


def min_sum_diffusion_update_step(nodes, edges, grid, u):
    # Implement a single update step for the given node `u` (accumulation,
    # rounding, distribution).
    # Return the assignment/label for node `u`.
    pass


def min_sum_diffusion_single_iteration(nodes, edges, grid):
    # Implement a single iteration for the Min-Sum diffusion method.
    # Iterate over all nodes and perform the update step on them.
    # Return the assignment/labeling for the full model.
    pass


def min_sum_diffusion_method(nodes, edges, grid):
    # Implement the Min-Sum diffusion method (run multiple iterations).
    # Return the assignment/labeling of the full model.
    pass