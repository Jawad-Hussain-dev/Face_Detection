import os
from feature_extraction import extract_hog_features

def build_dataset(root_dir):
    """
    Loops over all subfolders in root_dir and extracts HOG features for each image.
    
    Args:
        root_dir (str): Path to the root folder containing subfolders per person.

    Returns:
        X (list of np.ndarray): Feature vectors
        y (list of int): Corresponding class labels
        label_map (dict): Mapping from folder name (e.g. '01') to label (e.g. 0)
    """
    X = []
    y = []
    label_map = {}
    current_label = 0

    for person_folder in sorted(os.listdir(root_dir)):
        person_path = os.path.join(root_dir, person_folder)
        if not os.path.isdir(person_path):
            continue

        label_map[person_folder] = current_label
        print(f"Processing label {current_label} for folder: {person_folder}")

        for img_file in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_file)
            try:
                features = extract_hog_features(img_path)
                X.append(features)
                y.append(current_label)
            except Exception as e:
                print(f" Failed to process {img_path}: {e}")

        current_label += 1

    print(" Dataset built successfully.")
    return X, y, label_map
