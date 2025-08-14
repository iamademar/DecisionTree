from collections import Counter
from information_gain import information_gain

def build_tree(data, features, target):
    """
    Core recursive algorithm that constructs a decision tree using information gain.
    This implements the ID3 (Iterative Dichotomiser 3) algorithm.
    
    Parameters:
    - data: List of dictionaries representing the training dataset
    - features: List of feature names available for splitting
    - target: Name of the target variable (class label)
    
    Returns:
    - A tree structure (nested dictionaries) or a leaf value (string)
    """
    
    # Extract all target labels from the current data subset
    # Example: if target="Sell", labels = ["No", "Yes", "Yes", "Yes", "No"]
    labels = [row[target] for row in data]
    
    # BASE CASE 1: Pure Node (All labels are the same)
    # If all instances have the same target label, create a leaf node
    # Example: labels = ["Yes", "Yes", "Yes"] → return "Yes"
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    
    # BASE CASE 2: No Features Left (Feature exhaustion)
    # If we've used all features but still have mixed labels,
    # return the most common label (majority vote)
    # Example: labels = ["No", "Yes", "No"] → return "No" (majority)
    if not features:
        return Counter(labels).most_common(1)[0][0]

    # RECURSIVE CASE: Feature Selection and Splitting
    
    # Step 1: Calculate information gain for each remaining feature
    # Dictionary comprehension creates: {"Outlook": 0.25, "Temperature": 0.15}
    gains = {feature: information_gain(data, feature, target) for feature in features}
    
    # Step 2: Select the feature with the highest information gain
    # Uses max() with key=gains.get to find the key with maximum value
    best_feature = max(gains, key=gains.get)

    # Step 3: Create the tree node structure
    # Creates a dictionary where the key is the splitting feature
    # Example: {"Outlook": {}} - empty dict will hold the branches
    tree = {best_feature: {}}
    
    # Step 4: Get all unique values for the chosen feature
    # Example: if best_feature="Outlook", values = {"Sunny", "Rainy", "Overcast"}
    values = set(row[best_feature] for row in data)
    
    # Step 5: For each unique value, create a branch by recursive splitting
    for value in values:
        # Filter data to only include rows with this feature value
        # Example: subset for "Sunny" = all rows where Outlook="Sunny"
        subset = [row for row in data if row[best_feature] == value]
        
        # Recursively build a subtree for this subset using remaining features
        # Remove the current feature from available features to prevent reuse
        # Example: if we used "Outlook", remaining features = ["Temperature"]
        subtree = build_tree(subset, [f for f in features if f != best_feature], target)
        
        # Attach the returned subtree to the current tree node
        # Example: tree["Outlook"]["Sunny"] = {subtree for sunny conditions}
        tree[best_feature][value] = subtree

    # Return the completed tree structure
    # Final structure: {"Outlook": {"Sunny": {...}, "Rainy": {...}, "Overcast": "Yes"}}
    return tree
