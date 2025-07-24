import joblib

from dataset_builder import build_dataset
from decision_tree import train_decision_tree

dataset_path = r"C:\Users\jawad\Downloads\Processed_Dataset_2"
X, y, label_map = build_dataset(dataset_path)

model = train_decision_tree(X, y, max_depth=10)

joblib.dump((X, y, label_map), "hog_dataset.pkl")
print("Dataset saved to hog_dataset.pkl")

joblib.dump(model, "tree_model.pkl")
print("Model saved to tree_model.pkl")
