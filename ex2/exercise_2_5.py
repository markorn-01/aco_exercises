from model_2_5 import *  # Import all stereo models
import exercise_2 as student  # Use LP/ILP formulations from exercise_2
import time


def solve_and_analyze(nodes, edges, method_name, solver_type="ILP"):
    """
    Solve a given graph labeling problem using ILP/LP methods and measure performance.
    Args:
        nodes (list): List of nodes.
        edges (list): List of edges.
        method_name (str): Method to use ('Sherali-Adams' or 'Fortet').
        solver_type (str): Type of solver ('ILP' or 'LP').

    Returns:
        dict: A dictionary containing problem size, solution time, and objective value.
    """
    # Choose the appropriate solver method
    if method_name == "Sherali-Adams" and solver_type == "ILP":
        convert = student.convert_to_ilp
    elif method_name == "Fortet" and solver_type == "ILP":
        convert = student.convert_to_ilp_fortet
    elif method_name == "Fortet" and solver_type == "LP":
        convert = student.convert_to_lp_fortet
    else:
        raise ValueError("Invalid method_name or solver_type combination.")

    # Formulate the problem
    start_time = time.time()
    if solver_type == "LP":  # LP functions return (problem, x_vars)
        problem, x_vars = convert(nodes, edges)
    else:  # ILP functions return only the problem
        problem = convert(nodes, edges)
    setup_time = time.time() - start_time

    # Solve the problem
    solve_start = time.time()
    problem.solve()
    solve_time = time.time() - solve_start

    # Collect results
    results = {
        "variables": len(problem.variables()),
        "constraints": len(problem.constraints),
        "setup_time": setup_time,
        "solve_time": solve_time,
        "objective_value": problem.objective.value(),
    }

    return results



def analyze_all_models():
    """
    Analyze all stereo vision models for Sherali-Adams and Fortet linearizations using ILP and LP-Fortet.
    Returns:
        list of dict: A list of results for each method, downsampling, and solver type.
    """
    results = []

    for model, downsampling in zip(all_models(), reversed(ALL_MODEL_DOWNSAMPLINGS)):
        nodes, edges = model

        # Solve Sherali-Adams ILP
        result = solve_and_analyze(nodes, edges, "Sherali-Adams", solver_type="ILP")
        result["downsampling"] = downsampling
        result["method"] = "Sherali-Adams"
        result["solver"] = "ILP"
        results.append(result)

        # Solve Fortet ILP
        result = solve_and_analyze(nodes, edges, "Fortet", solver_type="ILP")
        result["downsampling"] = downsampling
        result["method"] = "Fortet"
        result["solver"] = "ILP"
        results.append(result)

        # Solve Fortet LP
        result = solve_and_analyze(nodes, edges, "Fortet", solver_type="LP")
        result["downsampling"] = downsampling
        result["method"] = "Fortet"
        result["solver"] = "LP"
        results.append(result)

    return results


if __name__ == "__main__":
    # Analyze all models and print the results
    analysis_results = analyze_all_models()

    print("Analysis of Stereo Vision Models:")
    print(f"{'Downsampling':<15}{'Method':<15}{'Solver':<5}{'#Vars':<10}{'#Constraints':<15}"
          f"{'Setup Time (s)':<15}{'Solve Time (s)':<15}{'Objective Value':<10}")
    for result in sorted(analysis_results, key=lambda x: (x['downsampling'], x['method'], x['solver'])):
        print(f"{result['downsampling']:<15}{result['method']:<15}{result['solver']:<5}"
              f"{result['variables']:<10}{result['constraints']:<15}"
              f"{result['setup_time']:<15.4f}{result['solve_time']:<15.4f}{result['objective_value']:<10.4f}")
