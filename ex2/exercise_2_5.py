# Author: Quang Minh Ngo <ve330@stud.uni-heidelberg.de>
# Author: Taha Erkoc <nx324@stud.uni-heidelberg.de>

from model_2_5 import *
import exercise_2 as student
import time
import pulp


def solve_and_analyze(nodes, edges, method_name, solver_type, threshold_size):
    """
    Solve a given graph labeling problem and measure performance.
    Skips large ILP problems to avoid excessive runtimes.
    """
    if solver_type == "ILP" and len(nodes) * len(edges) > threshold_size:
        return None  # Skip large ILP instances

    # Map method names and solvers to conversion functions
    conversion_methods = {
        ("Sherali-Adams", "ILP"): student.convert_to_ilp,
        ("Sherali-Adams", "LP"): student.convert_to_lp,
        ("Fortet", "ILP"): student.convert_to_ilp_fortet,
        ("Fortet", "LP"): student.convert_to_lp_fortet,
    }

    if (method_name, solver_type) not in conversion_methods:
        raise ValueError("Invalid method_name or solver_type combination.")

    convert = conversion_methods[(method_name, solver_type)]
    
    try:
        start_time = time.time()
        if solver_type == "LP":
            problem, _ = convert(nodes, edges)
        else:
            problem = convert(nodes, edges)
        setup_time = time.time() - start_time

        # Solve the problem
        solve_start = time.time()
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        solve_time = time.time() - solve_start

        return {
            "variables": len(problem.variables()),
            "constraints": len(problem.constraints),
            "setup_time": setup_time,
            "solve_time": solve_time,
            "objective_value": problem.objective.value(),
        }
    except Exception as e:
        print(f"Error solving {method_name}-{solver_type}: {e}")
        return None


def analyze_all_models():
    """
    Analyze all stereo vision models using Sherali-Adams and Fortet methods for both LP and ILP.
    """
    methods = [("Sherali-Adams", "ILP"), ("Sherali-Adams", "LP"), ("Fortet", "ILP"), ("Fortet", "LP")]
    results = []
    threshold_size = 24 * 18  # Only solve small ILPs

    for model, downsampling in zip(all_models(), reversed(ALL_MODEL_DOWNSAMPLINGS)):
        nodes, edges = model
        for method_name, solver_type in methods:
            result = solve_and_analyze(nodes, edges, method_name, solver_type, threshold_size)
            if result:
                result.update({"downsampling": downsampling, "method": method_name, "solver": solver_type})
                results.append(result)

    return results


def print_summary(analysis_results):
    """
    Summarize and print results.
    """
    print("Analysis of Stereo Vision Models:")
    print(f"{'Downsampling':<15}{'Method':<15}{'Solver':<5}{'#Vars':<10}{'#Constraints':<15}"
          f"{'Setup Time (s)':<15}{'Solve Time (s)':<15}{'Objective Value':<10}")
    for result in sorted(analysis_results, key=lambda x: (x['downsampling'], x['method'], x['solver'])):
        print(f"{result['downsampling']:<15}{result['method']:<15}{result['solver']:<5}"
              f"{result['variables']:<10}{result['constraints']:<15}"
              f"{result['setup_time']:<15.4f}{result['solve_time']:<15.4f}{result['objective_value']:<10.4f}")


if __name__ == "__main__":
    results = analyze_all_models()
    print_summary(results)


'''
The resulting LP/ILP formulations are quite large, with the number of variables and constraints.
In fact, they are so large, that we run into a memory error, even with the
reduced downsampling and setting a timeout. So at least for our implementation,
we are not able to solve all instances in reasoaonble time neither with ILP nor LP
(including both Sherali-Adams as well as Fortet linearization).

Generally, we think that using general LP/ILP solvers for large-scale vision problems
can be challenging due to the size and computational demands of the problems.
For practical applications, specialized algorithms or approximate methods tailored to the problem
domain should be the way to tackle such problems.
'''