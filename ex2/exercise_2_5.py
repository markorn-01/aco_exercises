from model_2_5 import *  # Import all stereo models
import exercise_2 as student  # Use LP/ILP formulations from exercise_2
import time


def solve_and_analyze(nodes, edges, method_name, solver_type):
    """
    Solve a given graph labeling problem and measure performance.
    """
    # Map method names and solver types to functions
    conversion_methods = {
        ("Sherali-Adams", "ILP"): student.convert_to_ilp,
        ("Sherali-Adams", "LP"): student.convert_to_lp,
        ("Fortet", "ILP"): student.convert_to_ilp_fortet,
        ("Fortet", "LP"): student.convert_to_lp_fortet,
    }

    if (method_name, solver_type) not in conversion_methods:
        raise ValueError("Invalid method_name or solver_type combination.")

    # Get the conversion function
    convert = conversion_methods[(method_name, solver_type)]

    # Formulate the problem
    start_time = time.time()
    if solver_type == "LP":  # LP functions return (problem, x_vars)
        problem, _ = convert(nodes, edges)
    else:  # ILP functions return only the problem
        problem = convert(nodes, edges)
    setup_time = time.time() - start_time

    # Solve the problem
    solve_start = time.time()
    problem.solve()
    solve_time = time.time() - solve_start

    # Return results
    return {
        "variables": len(problem.variables()),
        "constraints": len(problem.constraints),
        "setup_time": setup_time,
        "solve_time": solve_time,
        "objective_value": problem.objective.value(),
    }


def analyze_all_models():
    """
    Analyze all stereo vision models using Sherali-Adams and Fortet methods for both ILP and LP.
    """
    methods = [("Sherali-Adams", "ILP"), ("Sherali-Adams", "LP"), ("Fortet", "ILP"), ("Fortet", "LP")]
    results = []

    for model, downsampling in zip(all_models(), reversed(ALL_MODEL_DOWNSAMPLINGS)):
        nodes, edges = model
        for method_name, solver_type in methods:
            result = solve_and_analyze(nodes, edges, method_name, solver_type)
            result.update({"downsampling": downsampling, "method": method_name, "solver": solver_type})
            results.append(result)

    return results


if __name__ == "__main__":
    # Analyze all models
    analysis_results = analyze_all_models()

    # Print header
    print("Analysis of Stereo Vision Models:")
    print(f"{'Downsampling':<15}{'Method':<15}{'Solver':<5}{'#Vars':<10}{'#Constraints':<15}"
          f"{'Setup Time (s)':<15}{'Solve Time (s)':<15}{'Objective Value':<10}")

    # Print results
    for result in sorted(analysis_results, key=lambda x: (x['downsampling'], x['method'], x['solver'])):
        print(f"{result['downsampling']:<15}{result['method']:<15}{result['solver']:<5}"
              f"{result['variables']:<10}{result['constraints']:<15}"
              f"{result['setup_time']:<15.4f}{result['solve_time']:<15.4f}{result['objective_value']:<10.4f}")
