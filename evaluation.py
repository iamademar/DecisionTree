from build_tree import build_tree
from is_good_day_to_sell import is_good_day_to_sell
from static_tree_predict import static_tree_predict

# --- Dataset ---
dataset = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "No"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Sell": "No"}
]

features = ["Outlook", "Temperature"]
target = "Sell"

# Complex dataset
complex_dataset = [
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},     # contradiction to static
    {"Outlook": "Sunny", "Temperature": "Cool", "Sell": "No"},     # contradiction
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Hot", "Sell": "Yes"},     # unseen combo
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},     # contradiction
    {"Outlook": "Rainy", "Temperature": "Mild", "Sell": "Yes"},    # flipped
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Sell": "No"},  # exception
    {"Outlook": "Overcast", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},     # repeat for reinforcement
]

# Much more challenging dataset that exposes static tree limitations
challenging_dataset = [
    # Sunny days - Static tree says Hot=No, others=Yes
    # Let's make the pattern more complex: depends on both features together
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},      # Static: No, should be Yes
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},      # Reinforce pattern
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},      # Static gets this wrong
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "No"},      # Static: Yes, should be No
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "No"},      # Reinforce
    {"Outlook": "Sunny", "Temperature": "Cool", "Sell": "Yes"},     # Static: Yes, correct by luck
    {"Outlook": "Sunny", "Temperature": "Cool", "Sell": "Yes"},
    
    # Rainy days - Static tree says Cool=Yes, others=No  
    # Let's completely flip this pattern
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},      # Static: Yes, should be No
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},      # Reinforce
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},      # Static gets this wrong
    {"Outlook": "Rainy", "Temperature": "Mild", "Sell": "Yes"},     # Static: No, should be Yes
    {"Outlook": "Rainy", "Temperature": "Mild", "Sell": "Yes"},     # Reinforce
    {"Outlook": "Rainy", "Temperature": "Hot", "Sell": "Yes"},      # Static: No, should be Yes
    {"Outlook": "Rainy", "Temperature": "Hot", "Sell": "Yes"},      # Reinforce
    
    # Overcast days - Static tree always says Yes
    # Let's make it depend on temperature in a different way
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},    # Static: Yes, should be No
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},    # Reinforce
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},    # Static gets this wrong
    {"Outlook": "Overcast", "Temperature": "Mild", "Sell": "Yes"},  # Static: Yes, correct by luck
    {"Outlook": "Overcast", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Sell": "Yes"},  # Static: Yes, correct by luck
    {"Outlook": "Overcast", "Temperature": "Cool", "Sell": "Yes"},
    
    # Additional edge cases to really challenge the static tree
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},      # More evidence against static
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},      # More evidence against static
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},    # More evidence against static
]

# Extreme challenge dataset - completely opposite to static tree assumptions
extreme_dataset = [
    # Make the static tree fail catastrophically
    # Sunny + Hot should be Yes (static says No)
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Sunny", "Temperature": "Hot", "Sell": "Yes"},
    
    # Sunny + others should be No (static says Yes)
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "No"},
    {"Outlook": "Sunny", "Temperature": "Mild", "Sell": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Sell": "No"},
    {"Outlook": "Sunny", "Temperature": "Cool", "Sell": "No"},
    
    # Rainy + Cool should be No (static says Yes)
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},
    {"Outlook": "Rainy", "Temperature": "Cool", "Sell": "No"},
    
    # Rainy + others should be Yes (static says No)
    {"Outlook": "Rainy", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Hot", "Sell": "Yes"},
    {"Outlook": "Rainy", "Temperature": "Hot", "Sell": "Yes"},
    
    # Overcast should sometimes be No (static always says Yes)
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},
    {"Outlook": "Overcast", "Temperature": "Hot", "Sell": "No"},
    {"Outlook": "Overcast", "Temperature": "Mild", "Sell": "Yes"},
    {"Outlook": "Overcast", "Temperature": "Cool", "Sell": "Yes"},
]

def compare_predictions(tree, dataset, title=""):
    """Compare predictions between static tree and ML tree."""
    print(f"\n📊 {title}")
    print(f"{'Outlook':<10} {'Temp':<10} {'Actual':<7} {'Static':<8} {'ML Tree':<8} {'Match?'}")
    for row in dataset:
        actual = row["Sell"]
        static_pred = static_tree_predict(row)
        ml_pred = is_good_day_to_sell(tree, row)
        match = "✅" if static_pred == ml_pred else "❌"
        print(f"{row['Outlook']:<10} {row['Temperature']:<10} {actual:<7} {static_pred:<8} {ml_pred:<8} {match}")

def compute_accuracy(predict_fn, data, target):
    """Compute accuracy of a prediction function on a dataset."""
    correct = 0
    for row in data:
        if predict_fn(row) == row[target]:
            correct += 1
    return correct / len(data)

def detailed_comparison(tree, dataset, title=""):
    """Provide detailed analysis of where static vs ML tree differ"""
    print(f"\n🔍 DETAILED ANALYSIS: {title}")
    print("="*80)
    
    static_correct = 0
    ml_correct = 0
    both_correct = 0
    both_wrong = 0
    static_better = 0
    ml_better = 0
    
    disagreements = []
    
    for row in dataset:
        actual = row["Sell"]
        static_pred = static_tree_predict(row)
        ml_pred = is_good_day_to_sell(tree, row)
        
        static_right = static_pred == actual
        ml_right = ml_pred == actual
        
        if static_right:
            static_correct += 1
        if ml_right:
            ml_correct += 1
            
        if static_right and ml_right:
            both_correct += 1
        elif not static_right and not ml_right:
            both_wrong += 1
        elif static_right and not ml_right:
            static_better += 1
        elif not static_right and ml_right:
            ml_better += 1
            
        if static_pred != ml_pred:
            disagreements.append({
                'case': f"{row['Outlook']}-{row['Temperature']}",
                'actual': actual,
                'static': static_pred,
                'ml': ml_pred,
                'static_right': static_right,
                'ml_right': ml_right
            })
    
    total = len(dataset)
    print(f"📈 ACCURACY COMPARISON:")
    print(f"   Static Tree:  {static_correct:2d}/{total} = {static_correct/total:.1%}")
    print(f"   ML Tree:      {ml_correct:2d}/{total} = {ml_correct/total:.1%}")
    print(f"   Improvement:  {ml_correct-static_correct:+2d} predictions = {(ml_correct-static_correct)/total:+.1%}")
    
    print(f"\n🎯 AGREEMENT ANALYSIS:")
    print(f"   Both Correct: {both_correct:2d}/{total} = {both_correct/total:.1%}")
    print(f"   Both Wrong:   {both_wrong:2d}/{total} = {both_wrong/total:.1%}")
    print(f"   Static Better:{static_better:2d}/{total} = {static_better/total:.1%}")
    print(f"   ML Better:    {ml_better:2d}/{total} = {ml_better/total:.1%}")
    
    if disagreements:
        print(f"\n⚔️  DISAGREEMENTS ({len(disagreements)} cases):")
        print(f"{'Case':<15} {'Actual':<7} {'Static':<7} {'ML':<7} {'Winner'}")
        print("-" * 50)
        for d in disagreements:
            if d['static_right'] and not d['ml_right']:
                winner = "Static ✅"
            elif not d['static_right'] and d['ml_right']:
                winner = "ML ✅"
            elif not d['static_right'] and not d['ml_right']:
                winner = "Both ❌"
            else:
                winner = "Both ✅"
            print(f"{d['case']:<15} {d['actual']:<7} {d['static']:<7} {d['ml']:<7} {winner}")
    
    return {
        'static_accuracy': static_correct/total,
        'ml_accuracy': ml_correct/total,
        'improvement': (ml_correct-static_correct)/total,
        'disagreements': len(disagreements)
    }

if __name__ == "__main__":
    # Train the tree on original dataset
    tree = build_tree(dataset, features, target)
    print("🌳 Learned Tree Structure:")
    print(tree)

    # Simple demo
    print("\n" + "="*50)
    print("🚀 SIMPLE DEMO")
    print("="*50)
    test_instance = {"Outlook": "Sunny", "Temperature": "Mild"}
    print("Prediction for", test_instance, "->", is_good_day_to_sell(tree, test_instance))
    
    print("\n" + "="*100)
    print("🧪 TESTING DIFFERENT DATASETS TO EXPOSE STATIC TREE LIMITATIONS")
    print("="*100)

    # Test 1: Original simple data
    print("\n" + "🔹"*50)
    print("TEST 1: Original Training Data (should work well for both)")
    detailed_comparison(tree, dataset, "Original Training Data")

    # Test 2: Basic complex data  
    print("\n" + "🔹"*50)
    print("TEST 2: Basic Complex Data (some contradictions)")
    detailed_comparison(tree, complex_dataset, "Basic Complex Data")

    # Test 3: More challenging data
    print("\n" + "🔹"*50)
    print("TEST 3: Challenging Dataset (systematic patterns opposite to static tree)")
    challenging_tree = build_tree(challenging_dataset, features, target)
    print(f"🌳 New tree learned from challenging data: {challenging_tree}")
    detailed_comparison(challenging_tree, challenging_dataset, "Challenging Dataset")

    # Test 4: Extreme challenge
    print("\n" + "🔹"*50)
    print("TEST 4: Extreme Challenge (completely opposite patterns)")
    extreme_tree = build_tree(extreme_dataset, features, target)
    print(f"🌳 New tree learned from extreme data: {extreme_tree}")
    detailed_comparison(extreme_tree, extreme_dataset, "Extreme Challenge Dataset")


