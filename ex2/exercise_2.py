# This is the file where should insert your own code.
#
# Author: Your Name <your@email.com>

import pulp

##### Exercise 2.1 #####

# # Sherali-Adams linearization
# def convert_to_ilp(nodes, edges):
#     ilp = pulp.LpProblem('GM')
#     # populate ILP
#     return ilp


# # Fortet linearization
# def convert_to_ilp_fortet(nodes, edges):
#     ilp = pulp.LpProblem('GM')
#     # populate ILP
#     return ilp


# def ilp_to_labeling(nodes, edges, ilp):
#     labeling = []
#     # compute labeling
#     return labeling

def convert_to_ilp_fortet(nodes, edges):
    """
    Create an ILP using Fortet's linearization.
    Args:
        nodes (list): List of Node objects, each with a "costs" attribute (list of state costs).
        edges (list): List of Edge objects, each with "left", "right", and "costs" attributes (state cost dictionary).

    Returns:
        lp (pulp.LpProblem): Formulated ILP problem.
    """
    # Create the ILP problem
    ilp = pulp.LpProblem("GM", pulp.LpMinimize)

    # Define binary variables for each node's state
    x = {}  # x[i, k] is the binary variable for node i in state k
    for i, node in enumerate(nodes):
        for k in range(len(node.costs)):
            x[(i, k)] = pulp.LpVariable(f'x_{i}_{k}', cat="Binary")

    # Add objective for node costs
    for i, node in enumerate(nodes):
        for k, cost in enumerate(node.costs):
            ilp += cost * x[(i, k)]  # Add cost for each state

    # Add constraints: Each node must be assigned exactly one state
    for i, node in enumerate(nodes):
        ilp += pulp.lpSum(x[(i, k)] for k in range(len(node.costs))) == 1

    # Define auxiliary variables for edge state interactions
    aux = {}  # aux[(i, j, k, l)] for edge (i, j) with states (k, l)
    for edge in edges:
        i, j = edge.left, edge.right
        for (k, l), cost in edge.costs.items():
            aux[(i, j, k, l)] = pulp.LpVariable(f'aux_{i}_{j}_{k}_{l}', cat="Binary")
            # Add edge costs to the objective
            ilp += cost * aux[(i, j, k, l)]

            # Fortet's constraints for auxiliary variables
            ilp += aux[(i, j, k, l)] <= x[(i, k)]
            ilp += aux[(i, j, k, l)] <= x[(j, l)]
            ilp += aux[(i, j, k, l)] >= x[(i, k)] + x[(j, l)] - 1

    return ilp


def ilp_to_labeling(nodes, edges, ilp):
    """
    Extract the labeling from the solved ILP problem.
    Args:
        nodes (list): List of nodes.
        edges (list): List of edges.
        ilp (pulp.LpProblem): Solved ILP problem.

    Returns:
        labeling (list): Extracted labeling as the state assigned to each node.
    """
    # Extract the values of the node variables
    labeling = []
    for i, node in enumerate(nodes):
        for k in range(len(node.costs)):  # Iterate over all possible states
            var_name = f'x_{i}_{k}'
            variable = ilp.variablesDict()[var_name]  # Access the variable from the solved ILP
            if pulp.value(variable) == 1:  # Check if the variable is active (state selected)
                labeling.append(k)
                break  # Each node can only be assigned one state

    return labeling


##### Exercise 2.2 #####
# Relaxed Sherali-Adams linearization
def convert_to_lp(nodes, edges):
    lp = pulp.LpProblem('GM')
    # populate LP
    return lp


# Relaxed Fortet linearization
def convert_to_lp_fortet(nodes, edges):
    lp = pulp.LpProblem('GM')
    # populate LP
    return lp


def lp_to_labeling(nodes, edges, lp):
    labeling = []
    # compute labeling
    return labeling
