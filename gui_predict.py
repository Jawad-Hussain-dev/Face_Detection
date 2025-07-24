import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib
from explain_prediction import explain_prediction
from feature_extraction import extract_hog_features
import numpy as np

model = joblib.load("tree_model.pkl")
X, y, label_map = joblib.load("hog_dataset.pkl")
reverse_map = {v: k for k, v in label_map.items()}

root = tk.Tk()
root.title(" Person Identifier (HOG + Decision Tree)")
root.geometry("900x700")
root.configure(bg="#f0f4f7")

title = tk.Label(root, text="Person Identification System", font=("Helvetica", 20, "bold"), bg="#f0f4f7", fg="#2c3e50")
title.pack(pady=10)

image_frame = tk.Frame(root, bg="#f0f4f7")
image_frame.pack(pady=10)

original_label = tk.Label(image_frame, text="Original Image", font=("Helvetica", 12), bg="#f0f4f7")
original_label.grid(row=0, column=0)
image_label = tk.Label(image_frame, bd=2, relief="solid")
image_label.grid(row=1, column=0, padx=10)

hog_text_label = tk.Label(image_frame, text="HOG Visualization", font=("Helvetica", 12), bg="#f0f4f7")
hog_text_label.grid(row=0, column=1)
hog_label = tk.Label(image_frame, bd=2, relief="solid")
hog_label.grid(row=1, column=1, padx=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"), bg="#f0f4f7", fg="#27ae60")
result_label.pack(pady=10)

explain_frame = tk.Frame(root, bg="#f0f4f7")
explain_frame.pack(pady=10)
explain_title = tk.Label(explain_frame, text=" Decision Path Explanation", font=("Helvetica", 14), bg="#f0f4f7")
explain_title.pack()
explain_text = tk.Text(explain_frame, height=15, width=90, wrap="word", font=("Consolas", 10))
explain_text.pack()

def upload_and_predict():
    file_path = filedialog.askopenfilename(title="Choose an image")

    if file_path:
        img = Image.open(file_path).convert("L")
        img_resized = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img_resized)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        try:
            features, hog_img = extract_hog_features(file_path, visualize=True)
            features = np.array(features).reshape(1, -1)

            pred = model.predict(features)[0]
            result_label.config(text=f"Predicted: Person {reverse_map[pred]}")

            hog_img_uint8 = (hog_img * 255).astype(np.uint8)
            hog_pil = Image.fromarray(hog_img_uint8).resize((200, 200))
            hog_tk = ImageTk.PhotoImage(hog_pil)
            hog_label.config(image=hog_tk)
            hog_label.image = hog_tk

            node_indicator = model.decision_path(features)
            leaf_id = model.apply(features)
            explanation = f"Decision Path for Person {reverse_map[pred]}:\n\n"

            for node_id in node_indicator.indices:
                if leaf_id[0] == node_id:
                    explanation += f" Reached leaf node {node_id}\n"
                else:
                    feature = model.tree_.feature[node_id]
                    threshold = model.tree_.threshold[node_id]
                    decision = "<=" if features[0, feature] <= threshold else ">"
                    explanation += f" - f{feature} ({decision} {threshold:.2f})\n"

            explain_text.delete("1.0", tk.END)
            explain_text.insert(tk.END, explanation)

        except Exception as e:
            result_label.config(text=" Error processing image", fg="red")
            explain_text.delete("1.0", tk.END)
            explain_text.insert(tk.END, str(e))

tk.Button(root, text=" Upload Image & Predict", command=upload_and_predict,
          font=("Helvetica", 14), bg="#3498db", fg="white", padx=10, pady=5).pack(pady=15)

root.mainloop()
