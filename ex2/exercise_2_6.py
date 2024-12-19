# Author: Quang Minh Ngo <ve330@stud.uni-heidelberg.de>
# Author: Taha Erkoc <nx324@stud.uni-heidelberg.de>

import pulp
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def load_masks(directory='ex2/segments/'):
    """Load masks and convert them to binary arrays."""
    masks = {}
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            path = os.path.join(directory, filename)
            with Image.open(path) as img:
                masks[filename] = np.array(img).astype(bool)
    return masks

def create_ilp_model(masks):
    """Create an ILP model to select non-overlapping masks."""
    model = pulp.LpProblem("Maximize_Covered_Area", pulp.LpMaximize)
    x = {mask: pulp.LpVariable(f"x_{mask}", cat='Binary') for mask in masks}
    
    # Objective function to maximize the total area covered by selected masks
    model += pulp.lpSum(x[mask] * np.sum(masks[mask]) for mask in masks)

    # Constraint to ensure no two overlapping masks are selected
    keys = list(masks.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if np.any(np.logical_and(masks[keys[i]], masks[keys[j]])):
                model += x[keys[i]] + x[keys[j]] <= 1

    return model, x

def solve_ilp(model):
    """Solve the ILP model."""
    model.solve()
    return model

def visualize_results(masks, x):
    """Visualize the selected masks."""
    selected_masks = [mask for mask in x if pulp.value(x[mask]) == 1]
    combined_mask = np.zeros_like(next(iter(masks.values())), dtype=np.uint8)
    
    for mask in selected_masks:
        combined_mask = np.logical_or(combined_mask, masks[mask])
    
    plt.imshow(combined_mask, cmap='gray')
    plt.title('Selected Non-Overlapping Segmentations')
    plt.show()

def main():
    masks = load_masks()
    model, x = create_ilp_model(masks)
    solve_ilp(model)
    visualize_results(masks, x)

if __name__ == "__main__":
    main()