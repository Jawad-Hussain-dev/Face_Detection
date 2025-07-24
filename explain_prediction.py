import numpy as np
from feature_extraction import extract_hog_features

def explain_prediction(model, image_path, label_map):
    """
    Predicts and explains the decision path for a single image.

    Args:
        model (DecisionTreeClassifier): Trained model
        image_path (str): Path to test image
        label_map (dict): Maps folder name to label
    """
    reverse_map = {v: k for k, v in label_map.items()}

    features, _ = extract_hog_features(image_path, visualize=True)
    features = np.array(features).reshape(1, -1)

    pred = model.predict(features)[0]
    print(f"\n Predicted label: {pred} (Person folder: {reverse_map[pred]})")

    node_indicator = model.decision_path(features)
    leaf_id = model.apply(features)

    print("\n Decision path:")
    for node_id in node_indicator.indices:
        if leaf_id[0] == node_id:
            print(f" Reached leaf node {node_id}")
        else:
            feature = model.tree_.feature[node_id]
            threshold = model.tree_.threshold[node_id]
            decision = "<=" if features[0, feature] <= threshold else ">"
            print(f" - Feature f{feature} ({decision} {threshold:.2f})")

    print("\n Top 5 most important features:")
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-5:][::-1]
    for idx in top_indices:
        print(f" - f{idx} (Importance: {importances[idx]:.4f})")
