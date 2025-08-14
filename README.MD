## Decision Tree (ID3) – Interpretable ML in Python

This repository demonstrates a minimal, from-scratch implementation of a Decision Tree using the ID3 algorithm with entropy and information gain. It compares a hard-coded (static) decision rule tree with a learned tree on simple coffee franchise startup example to show where learning helps.

For the full walkthrough and motivation, see the blog post: [Is Today a Good Day to Sell Coffee?](https://blog.ademartutor.com/p/is-today-a-good-day-to-sell-coffee)

## What’s inside
- `entropy.py`: Calculates entropy for a dataset.
- `information_gain.py`: Computes information gain for splits.
- `build_tree.py`: Core ID3 training loop that builds a tree from data.
- `is_good_day_to_sell.py`: Uses a learned tree to predict.
- `static_tree_predict.py`: A simple, hard-coded baseline tree for comparison.
- `evaluation.py`: Datasets and utilities to compare static vs learned trees and print metrics.

## Quickstart
Requirements: Python 3.8+

```bash
git clone git@github.com:iamademar/DecisionTree.git
cd DecisionTree
python3 evaluation.py | cat
```

You’ll see:
- The learned tree structure
- Side-by-side predictions for several datasets
- Accuracy comparison and disagreement analysis between the static and learned trees

## Minimal usage example
```python
from build_tree import build_tree
from is_good_day_to_sell import is_good_day_to_sell

dataset = [
    {"Outlook": "Sunny", "Temperature": "Hot",  "Sell": "No"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "Yes"},
]

tree = build_tree(dataset, features=["Outlook", "Temperature"], target="Sell")
prediction = is_good_day_to_sell(tree, {"Outlook": "Sunny", "Temperature": "Mild"})
print(prediction)  # "Yes" or "No"
```

## Why build this project?
- Simple, readable code to illustrate entropy, information gain, and ID3.
- Highlights interpretability: the resulting tree is human-readable.
- Shows when a static rule may fail and how a learned model adapts.

