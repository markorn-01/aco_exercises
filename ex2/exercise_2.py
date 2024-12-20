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
<<<<<<< HEAD

=======
>>>>>>> d627758991d8f41d54ded0b21340cc5992263d0f
    # marginalization constraints
    for e in edges:
        for s in range(len(nodes[e.left].costs)):
            ilp += pulp.lpSum(y[(e.left,e.right,s,t)] 
                            for t in range(len(nodes[e.right].costs))) == x[(e.left,s)]
        
        for t in range(len(nodes[e.right].costs)):
            ilp += pulp.lpSum(y[(e.left,e.right,s,t)] 
                            for s in range(len(nodes[e.left].costs))) == x[(e.right,t)]
<<<<<<< HEAD
            
=======
    
>>>>>>> d627758991d8f41d54ded0b21340cc5992263d0f
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
    y = {}

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
            y[(i, j, k, l)] = pulp.LpVariable(f'y_{i}_{j}_{k}_{l}', cat="Binary")
            objective.append(cost * y[(i, j, k, l)])

            # Fortet's linearization constraints
            ilp += y[(i, j, k, l)] <= x[(i, k)]
            ilp += y[(i, j, k, l)] <= x[(j, l)]
            ilp += y[(i, j, k, l)] >= x[(i, k)] + x[(j, l)] - 1

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
 
    lp = pulp.LpProblem("Sherali_Adams_Relaxation", pulp.LpMinimize)

    x = {}
    for i, node in enumerate(nodes):
        for k in range(len(node.costs)):
            x[(i, k)] = pulp.LpVariable(f"x_{i}_{k}", lowBound=0, upBound=1, cat="Continuous")

    y = {}

    objective = []
    for i, node in enumerate(nodes):
        for k, cost in enumerate(node.costs):
            objective.append(cost * x[(i, k)])

    for edge in edges:
        i, j = edge.left, edge.right
        for (k, l), cost in edge.costs.items():
            y[(i, j, k, l)] = pulp.LpVariable(f"y_{i}_{j}_{k}_{l}", lowBound=0, upBound=1, cat="Continuous")
            objective.append(cost * y[(i, j, k, l)])

            lp += y[(i, j, k, l)] <= x[(i, k)]
            lp += y[(i, j, k, l)] <= x[(j, l)]
            lp += y[(i, j, k, l)] >= x[(i, k)] + x[(j, l)] - 1

    for i, node in enumerate(nodes):
        lp += pulp.lpSum(x[(i, k)] for k in range(len(node.costs))) == 1

    lp += pulp.lpSum(objective)

<<<<<<< HEAD
=======
    # # Debug prints
    # print("Objective function:", lp.objective)
    # for constraint in lp.constraints.values():
    #     print("Constraint:", constraint)
>>>>>>> d627758991d8f41d54ded0b21340cc5992263d0f
    # marginalization constraints
    for e in edges:
        for s in range(len(nodes[e.left].costs)):
            lp += pulp.lpSum(y[(e.left,e.right,s,t)] 
                            for t in range(len(nodes[e.right].costs))) == x[(e.left,s)]
        
        for t in range(len(nodes[e.right].costs)):
            lp += pulp.lpSum(y[(e.left,e.right,s,t)] 
                            for s in range(len(nodes[e.left].costs))) == x[(e.right,t)]
<<<<<<< HEAD

=======
    
>>>>>>> d627758991d8f41d54ded0b21340cc5992263d0f
    return lp, x

# Relaxed Fortet linearization
def convert_to_lp_fortet(nodes, edges):
    """
    Create the LP relaxation using Fortet's linearization.
    Args:
        nodes (list): List of Node objects, each with a "costs" attribute.
        edges (list): List of Edge objects, each with "left", "right", and "costs".

    Returns:
        lp (pulp.LpProblem): LP relaxation problem.
        x (dict): Relaxed decision variables for node states.
    """
    # Create LP problem
    lp = pulp.LpProblem("Fortet_Relaxation", pulp.LpMinimize)

    # Continuous decision variables for node states
    x = {}
    for i, node in enumerate(nodes):
        for k in range(len(node.costs)):
            x[(i, k)] = pulp.LpVariable(f"x_{i}_{k}", lowBound=0, upBound=1, cat="Continuous")

    # Auxiliary variables for edge interactions
    y = {}

    # Objective function: Node costs
    objective = []
    for i, node in enumerate(nodes):
        for k, cost in enumerate(node.costs):
            objective.append(cost * x[(i, k)])

    # Edge costs and Fortet constraints
    for edge in edges:
        i, j = edge.left, edge.right
        for (k, l), cost in edge.costs.items():
            # Auxiliary variable for the product x[i, k] * x[j, l]
            y[(i, j, k, l)] = pulp.LpVariable(f"y_{i}_{j}_{k}_{l}", lowBound=0, upBound=1, cat="Continuous")
            objective.append(cost * y[(i, j, k, l)])

            # Fortet's linearization constrain
            lp += y[(i, j, k, l)] <= x[(i, k)]
            lp += y[(i, j, k, l)] <= x[(j, l)]
            lp += y[(i, j, k, l)] >= x[(i, k)] + x[(j, l)] - 1

    # Node constraints: Each node must sum to exactly 1
    for i, node in enumerate(nodes):
        lp += pulp.lpSum(x[(i, k)] for k in range(len(node.costs))) == 1

    # Set the objective function
    lp += pulp.lpSum(objective)

    return lp, x


# Extract labeling from LP solution
def lp_to_labeling(nodes, edges, lp, x):
    """
    Extract the labeling from the LP solution using improved rounding.
    Args:
        nodes (list): List of nodes.
        edges (list): List of edges.
        lp (pulp.LpProblem): Solved LP problem.
        x (dict): Relaxed decision variables.
        threshold (float): Minimum value to consider a state active.

    Returns:
        labeling (list): Rounded integer labeling.
    """
    labeling = []
    for i, node in enumerate(nodes):
        state_values = [pulp.value(x[(i, k)]) for k in range(len(node.costs))]

        # Rounding: choose state with highest value
        best_state = max(range(len(node.costs)), key=lambda k: state_values[k])
        labeling.append(best_state)
    return labeling
<<<<<<< HEAD


'''
The LP Optima is a relaxed solution. It is not guaranteed to be integer and provides a
lower bound for the ILP optimum in a minimazation problem with Sherali-Adams giving
tighter bounds than Fortet. The LP solution is much faster to solve than the ILP

The rounded solution uses an heuristic approach to convert the fractional solution to 
an integer solution. While it is a lot faster and simpler than solving the ILP, it is 
not neccessarily optimal and introduces a small error.

The ILP solution is the exact optimal integer solution that satisfies all constraints.
It guarantees the best possible solution for the objective function but is computationally
expensive to solve, especially for larger-scale problems.
'''
=======
>>>>>>> d627758991d8f41d54ded0b21340cc5992263d0f
