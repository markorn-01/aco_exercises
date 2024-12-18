from model_2_4 import *
import exercise_2 as student

def solve_models(models, use_ilp=False, method="fortet"):
    results = []
    for index, (nodes, edges) in enumerate(models):
        if use_ilp:
            # Solve using ILP with the preferred linearization
            problem = student.convert_to_ilp(nodes, edges)
        else:
            # Choose the method of linearization based on the 'method' parameter
            if method == "fortet":
                problem, _ = student.convert_to_lp_fortet(nodes, edges)
            elif method == "sherali-adams":
                problem, _ = student.convert_to_lp(nodes, edges)
        
        problem.solve()

        # Collect results
        if use_ilp:
            labeling = student.ilp_to_labeling(nodes, edges, problem)
        else:
            labeling = student.lp_to_labeling(nodes, edges, problem, _)

        results.append(labeling)
    
    return results

def main():
    # Solve acyclic models
    print("Solving Acyclic Models")
    acyclic_lp_fortet_results = solve_models(ACYCLIC_MODELS, use_ilp=False, method="fortet")
    acyclic_lp_sherali_adams_results = solve_models(ACYCLIC_MODELS, use_ilp=False, method="sherali-adams")
    acyclic_ilp_results = solve_models(ACYCLIC_MODELS, use_ilp=True)

    # Solve cyclic models
    print("Solving Cyclic Models")
    cyclic_lp_fortet_results = solve_models(CYCLIC_MODELS, use_ilp=False, method="fortet")
    cyclic_lp_sherali_adams_results = solve_models(CYCLIC_MODELS, use_ilp=False, method="sherali-adams")
    cyclic_ilp_results = solve_models(CYCLIC_MODELS, use_ilp=True)

    # Output the results
    for i in range(len(ACYCLIC_MODELS)):
        print(f"Acyclic Model {i+1} - LP (Fortet): {acyclic_lp_fortet_results[i]}, LP (Sherali-Adams): {acyclic_lp_sherali_adams_results[i]}, ILP: {acyclic_ilp_results[i]}")
    
    for i in range(len(CYCLIC_MODELS)):
        print(f"Cyclic Model {i+1} - LP (Fortet): {cyclic_lp_fortet_results[i]}, LP (Sherali-Adams): {cyclic_lp_sherali_adams_results[i]}, ILP: {cyclic_ilp_results[i]}")

if __name__ == "__main__":
    main()
