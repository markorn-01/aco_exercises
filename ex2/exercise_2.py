# This is the file where should insert your own code.
#
# Author: Quang Minh Ngo <ve330@stud.uni-heidelberg.de>
# Author: Taha Erkoc <nx324@stud.uni-heidelberg.de>

import pulp

##### Exercise 2.1 #####

# Sherali-Adams linearization
def convert_to_ilp(nodes, edges):
    """
    Create an ILP using Sherali-Adams linearization for node and edge costs.
    Args:
        nodes (list): List of Node objects, each with a "costs" attribute (list of state costs).
        edges (list): List of Edge objects, each with "left", "right", and "costs" attributes (state cost dictionary).

    Returns:
        lp (pulp.LpProblem): Formulated ILP problem.
    """
    # Create the ILP problem
    ilp = pulp.LpProblem("GM", pulp.LpMinimize)

    # Binary variables for each node's state
    x = {}  # x[i, k] is the binary variable for node i in state k
    for i, node in enumerate(nodes):
        for k in range(len(node.costs)):
            x[(i, k)] = pulp.LpVariable(f'x_{i}_{k}', cat="Binary")

    # Auxiliary variables for edge interactions
    y = {}

    # Objective function: Combine node costs and edge costs
    objective = []

    # Node costs
    for i, node in enumerate(nodes):
        for k, cost in enumerate(node.costs):
            objective.append(cost * x[(i, k)])

    # Edge costs and Sherali-Adams constraints
    for edge in edges:
        i, j = edge.left, edge.right
        for (k, l), cost in edge.costs.items():
            # Auxiliary variable for product of binary variables x[(i, k)] and x[(j, l)]
            y[(i, j, k, l)] = pulp.LpVariable(f'y_{i}_{j}_{k}_{l}', lowBound=0, upBound=1, cat="Continuous")
            objective.append(cost * y[(i, j, k, l)])

            # Sherali-Adams constraints:
            ilp += y[(i, j, k, l)] <= x[(i, k)]
            ilp += y[(i, j, k, l)] <= x[(j, l)]
            ilp += y[(i, j, k, l)] >= x[(i, k)] + x[(j, l)] - 1

    # Node constraints: Each node must be assigned exactly one state
    for i, node in enumerate(nodes):
        ilp += pulp.lpSum(x[(i, k)] for k in range(len(node.costs))) == 1

    # Set the objective function
    ilp += pulp.lpSum(objective)

    return ilp

# Fortet linearization
def convert_to_ilp_fortet(nodes, edges):
    """
    Create an ILP using Fortet's linearization for node and edge costs.
    Args:
        nodes (list): List of Node objects, each with a "costs" attribute (list of state costs).
        edges (list): List of Edge objects, each with "left", "right", and "costs" attributes (state cost dictionary).

    Returns:
        lp (pulp.LpProblem): Formulated ILP problem.
    """
    # Create the ILP problem
    ilp = pulp.LpProblem("GM", pulp.LpMinimize)

    # Binary variables for each node's state
    x = {}
    for i, node in enumerate(nodes):
        for k in range(len(node.costs)):
            x[(i, k)] = pulp.LpVariable(f'x_{i}_{k}', cat="Binary")

    # Auxiliary variables for edge interactions
    aux = {}

    # Objective function: Add node costs and edge costs
    objective = []

    # Node costs
    for i, node in enumerate(nodes):
        for k, cost in enumerate(node.costs):
            objective.append(cost * x[(i, k)])

    # Edge costs and Fortet's constraints
    for edge in edges:
        i, j = edge.left, edge.right
        for (k, l), cost in edge.costs.items():
            aux[(i, j, k, l)] = pulp.LpVariable(f'aux_{i}_{j}_{k}_{l}', cat="Binary")
            objective.append(cost * aux[(i, j, k, l)])

            # Fortet's linearization constraints
            ilp += aux[(i, j, k, l)] <= x[(i, k)]
            ilp += aux[(i, j, k, l)] <= x[(j, l)]
            ilp += aux[(i, j, k, l)] >= x[(i, k)] + x[(j, l)] - 1

    # Set the objective function
    ilp += pulp.lpSum(objective)

    # Node constraints: Each node must be assigned exactly one state
    for i, node in enumerate(nodes):
        ilp += pulp.lpSum(x[(i, k)] for k in range(len(node.costs))) == 1

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
    labeling = []
    for i, node in enumerate(nodes):
        assigned_state = None
        for k in range(len(node.costs)):
            var_name = f'x_{i}_{k}'
            variable = ilp.variablesDict()[var_name]
            if pulp.value(variable) == 1:
                assigned_state = k
                break
        labeling.append(assigned_state)
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
