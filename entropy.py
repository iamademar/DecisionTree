from collections import Counter
import math

def entropy(data, target):
    """
    Calculate the entropy of a dataset for a given target attribute.
    
    Parameters:
    - data: List of dictionaries representing the dataset
    - target: Name of the target variable (class label)
    
    Returns:
    - float: Entropy value
    """
    counts = Counter(row[target] for row in data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
