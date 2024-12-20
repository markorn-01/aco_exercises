# Author: Quang Minh Ngo <ve330@stud.uni-heidelberg.de>
# Author: Taha Erkoc <nx324@stud.uni-heidelberg.de>

from model_2_4 import *
import exercise_2 as student


def solve_models(models, use_ilp=False, method="fortet"):
    """
    Solve a set of models and compute the objective values for LP (local polytope), LP (Fortet), and ILP.

    Args:
        models: List of models (nodes, edges).
        use_ilp: Whether to use ILP (True) or LP (False).
        method: Which LP relaxation method to use ("fortet" or "sherali-adams").

    Returns:
        List of dictionaries containing the results for each model.
    """
    results = []
    for index, (nodes, edges) in enumerate(models):
        if use_ilp:
            # Solve using ILP
            problem = student.convert_to_ilp(nodes, edges)
        else:
            # Choose the method of linearization based on the 'method' parameter
            if method == "fortet":
                problem, _ = student.convert_to_lp_fortet(nodes, edges)
            elif method == "sherali-adams":
                problem, _ = student.convert_to_lp(nodes, edges)
            else:
                raise ValueError(f"Unknown method: {method}")

        # Solve the problem
        problem.solve()

        # Collect results
        results.append({
            "model_index": index,
            "objective_value": problem.objective.value()
        })

    return results


def main():
    """
    Compute and display objective values for LP (local polytope), LP (Fortet), and ILP
    for both acyclic and cyclic problems.
    """
    results = {
        "acyclic": {
            "LP(local)": solve_models(ACYCLIC_MODELS, use_ilp=False, method="sherali-adams"),
            "LP(Fortet)": solve_models(ACYCLIC_MODELS, use_ilp=False, method="fortet"),
            "ILP": solve_models(ACYCLIC_MODELS, use_ilp=True),
        },
        "cyclic": {
            "LP(local)": solve_models(CYCLIC_MODELS, use_ilp=False, method="sherali-adams"),
            "LP(Fortet)": solve_models(CYCLIC_MODELS, use_ilp=False, method="fortet"),
            "ILP": solve_models(CYCLIC_MODELS, use_ilp=True),
        },
    }

    # Display results
    print("\nResults for Acyclic Models:")
    print("-" * 70)
    print(f"{'Model':<8} {'LP(local)':<15} {'LP(Fortet)':<15} {'ILP':<15}")
    print("-" * 70)
    for i in range(len(ACYCLIC_MODELS)):
        lp_local = results["acyclic"]["LP(local)"][i]["objective_value"]
        lp_fortet = results["acyclic"]["LP(Fortet)"][i]["objective_value"]
        ilp = results["acyclic"]["ILP"][i]["objective_value"]
        print(f"{i+1:<8} {lp_local:<15.2f} {lp_fortet:<15.2f} {ilp:<15.2f}")

    print("\nResults for Cyclic Models:")
    print("-" * 70)
    print(f"{'Model':<8} {'LP(local)':<15} {'LP(Fortet)':<15} {'ILP':<15}")
    print("-" * 70)
    for i in range(len(CYCLIC_MODELS)):
        lp_local = results["cyclic"]["LP(local)"][i]["objective_value"]
        lp_fortet = results["cyclic"]["LP(Fortet)"][i]["objective_value"]
        ilp = results["cyclic"]["ILP"][i]["objective_value"]
        print(f"{i+1:<8} {lp_local:<15.2f} {lp_fortet:<15.2f} {ilp:<15.2f}")


if __name__ == "__main__":
    main()

'''
Acyclic Models:

The ILP and LP (Sherali-Adams) solutions are identical in all cases while the LP (Fortet) solution
consistently has a higher objective value (higher costs).
Acyclic graphs do not have circular dependencies, which simplifies the optimization problem.
LP relaxations are often tight in such cases, leading to optimal solutions without the need for ILP
enforcement.
The LP (Fortet) relaxation is more conservative and assigns higher costs to the edges, leading to
slightly higher objective values compared to the other methods.

Cyclic Models:

There are significant differences between the LP and ILP solutions for cyclic models
for both Sherali-Adams and Fortet linearizations while the latter deviates even more than the former.
Cylcic graphs have circular dependencies, which LP relaxations cannot capture accurately due to their
fractional nature. As a result, the LP solutions are less accurate and may oversimplify the problem leading
to the significant differences from the ILP.
In this case the Fortet relaxation's auxiliary variables further complicate the optimization process that
deviate even more from the ILP solution.

These results show that acyclic models should be preferred if possible, as they allow to recreate the
exact ILP solution using LP relaxations. Cyclic models, on the other hand, require more complex methods.
'''