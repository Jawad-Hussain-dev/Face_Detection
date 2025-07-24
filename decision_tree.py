from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def train_decision_tree(X, y, max_depth=10):
    """
    Train a Decision Tree classifier and print accuracy + rules.

    Args:
        X (list of np.ndarray): Feature vectors
        y (list of int): Labels
        max_depth (int): Tree depth (for readability)

    Returns:
        model (DecisionTreeClassifier): Trained model
    """
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Decision Tree Accuracy: {acc * 100:.2f}%")

    # Print rules
    rules = export_text(model, max_depth=2, feature_names=[f'f{i}' for i in range(X.shape[1])])
    print("\n Sample Tree Rules (Depth 2):\n")
    print(rules)

    return model
