from entropy import entropy

def information_gain(data, feature, target):
    """
    Calculate the information gain of splitting the data by a given feature.
    
    Parameters:
    - data: List of dictionaries representing the dataset
    - feature: Name of the feature to split on
    - target: Name of the target variable (class label)
    
    Returns:
    - float: Information gain value
    """
    total_entropy = entropy(data, target)
    values = set(row[feature] for row in data)
    weighted_entropy = 0.0
    for value in values:
        subset = [row for row in data if row[feature] == value]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset, target)
    return total_entropy - weighted_entropy
