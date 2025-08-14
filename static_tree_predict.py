def static_tree_predict(instance):
    """
    Make a prediction using a static, hard-coded decision tree.
    This represents a manually crafted tree for comparison purposes.
    
    Parameters:
    - instance: Dictionary representing the instance to classify
    
    Returns:
    - string: The predicted class label ("Yes" or "No")
    """
    outlook = instance["Outlook"]
    temperature = instance["Temperature"]
    if outlook == "Sunny":
        if temperature == "Hot":
            return "No"
        else:
            return "Yes"
    elif outlook == "Rainy":
        if temperature == "Cool":
            return "Yes"
        else:
            return "No"
    elif outlook == "Overcast":
        return "Yes"
