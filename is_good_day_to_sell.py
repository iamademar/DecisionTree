def is_good_day_to_sell(tree, instance):
    """
    Make a prediction using a decision tree for a given instance.
    
    Parameters:
    - tree: The decision tree structure (nested dictionaries or leaf value)
    - instance: Dictionary representing the instance to classify
    
    Returns:
    - string: The predicted class label
    """
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    value = instance.get(feature)
    subtree = tree[feature].get(value)
    if subtree is None:
        # If no path exists for this value, return the most common class from current node
        # For simplicity, we'll return "Yes" as a default prediction
        return "Yes"
    return is_good_day_to_sell(subtree, instance)
