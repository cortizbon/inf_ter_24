
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

colors = ["#1A1F63", "#81D3CD"]  
cmap = ListedColormap(colors)


COLORES = {'verde':["#009966"],
               'ro_am_na':["#FFE9C5", "#F7B261","#D8841C", "#dd722a","#C24C31", "#BC3B26"],
               'az_verd': ["#CBECEF", "#81D3CD", "#0FB7B3", "#009999"],
               'ax_viol': ["#D9D9ED", "#2F399B", "#1A1F63", "#262947"],
               'ofiscal': ["#F9F9F9", "#2635bf"]}


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)  # Ensure non-negative
    mean = np.mean(array)
    if mean == 0:
        return 0
    array = np.sort(array)
    n = len(array)
    cumulative_diff = np.sum(np.abs(np.subtract.outer(array, array)))
    return cumulative_diff / (2 * n**2 * mean)

def lorenz_curve(array):
    """Calculate Lorenz curve values for a numpy array."""
    array = np.array(array, dtype=np.float64)
    array = np.sort(array)
    cum_values = np.cumsum(array) / np.sum(array)
    cum_values = np.insert(cum_values, 0, 0)  # Add 0 at start
    x = np.linspace(0, 1, len(cum_values))
    return x, cum_values

def gini_by_department(df, column='AsignacionEducacion'):
    """
    Calculate Gini coefficient for a column in a DataFrame (e.g., education assignation per department),
    and plot the Lorenz curve.
    """
    values = df[column].values
    gini_value = gini(values)
    
    # Lorenz curve
    x, y = lorenz_curve(values)
    plt.figure(figsize=(6,6))
    plt.plot(x, y, drawstyle='steps-post', label='Lorenz curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Line of equality')
    plt.fill_between(x, y, x, color="skyblue", alpha=0.4)
    plt.title(f'Lorenz Curve ({column})\nGini = {gini_value:.4f}')
    plt.xlabel('Cumulative share of departments')
    plt.ylabel('Cumulative share of education assignation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return gini_value

def path_to_labels_parents(df, path_cols, value_col, val_parent=""):
    """
    Transforms a DataFrame with a path hierarchy into labels and parents format for go.Treemap.

    Parameters:
        df (pd.DataFrame): Input data.
        path_cols (list): List of columns representing the hierarchy (from root to leaves).
        value_col (str): Column containing values for the treemap.

    Returns:
        pd.DataFrame with columns: 'id', 'label', 'parent', 'value'
    """
    from itertools import accumulate
    rows = []
    for _, row in df.iterrows():
        path_values = [str(row[col]) for col in path_cols]
        value = row[value_col]
        for i in range(len(path_values)):
            id_ = "/".join(path_values[:i+1])
            parent = "/".join(path_values[:i]) if i > 0 else val_parent
            label = path_values[i]
            rows.append({"id": id_, "label": label, "parent": parent, "value": value if i == len(path_values)-1 else None})

    treemap_df = pd.DataFrame(rows)
    treemap_df = treemap_df.groupby(["id", "label", "parent"], as_index=False).agg({"value": "sum"}).fillna(0)
    return treemap_df