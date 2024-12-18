from model_2_5 import *
import exercise_2 as student

def analyze_all_models():
    """
    Uses the all_models function to analyze the number of variables and constraints
    for all provided downsampled models.
    Returns:
        A list of dictionaries containing the analysis for each model.
    """
    results = []
    
    for model, downsampling in zip(all_models(), reversed(ALL_MODEL_DOWNSAMPLINGS)):
        nodes, edges = model
        num_vars = len(nodes)
        num_constraints = len(edges)
        results.append({
            "downsampling": downsampling,
            "variables": num_vars,
            "constraints": num_constraints,
        })
    return results

# Analyze all models
analysis_results = analyze_all_models()

# Print the results
for result in reversed(analysis_results):
    print(f"Downsampling={result['downsampling']}: Variables={result['variables']}, Constraints={result['constraints']}")
