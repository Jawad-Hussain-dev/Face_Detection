import cv2
import os

def preprocess_images(input_dir, output_dir, size=(256, 256)):
    for person in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person)
        out_path = os.path.join(output_dir, person)
        os.makedirs(out_path, exist_ok=True)

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Couldn't read image {img_path}")
                continue
            resized = cv2.resize(img, size)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(out_path, img_file), gray)

input_folder = r"C:\Users\jawad\Downloads\Dataset\Dataset"
output_folder = r"C:\Users\jawad\Downloads\Processed_Dataset_2"

preprocess_images(input_folder, output_folder)
