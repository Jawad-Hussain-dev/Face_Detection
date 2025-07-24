# Person Identifier using HOG + Decision Tree

This project implements a modular and interpretable **Person Identification System** using **HOG (Histogram of Oriented Gradients)** for facial feature extraction and a **Decision Tree Classifier** for prediction. Built for the course *Introduction to Computational Thinking*, the project emphasizes **explainability**, **interpretable modeling**, and the application of **Computational Thinking principles**.

##  Project Structure

```
Person-Identifier-HOG/
│
├── feature_extraction.py     # Extract HOG features from images
├── dataset_builder.py        # Load and preprocess dataset
├── decision_tree.py          # Train + interpret Decision Tree
├── train2.py                 # Main training pipeline
├── test_predict.py           # Test/validate on single images
├── gui_predict.py            # GUI for live testing and image input
│               
├── README.md                 # Project documentation (this file)
```

---

## ⚙️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Jawad-Hussain-dev/Person-Identifier.git
cd Person-Identifier
```

### 2. Install Required Dependencies

Either using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python scikit-image scikit-learn matplotlib pillow joblib
```

### 3. Train the Model

This script extracts features and trains a Decision Tree classifier.

```bash
python train2.py
```

* Trains model on face dataset
* Saves the trained model to disk
* Reports accuracy and depth

### 4. Predict Using the GUI

Use the GUI to predict an identity from:

* An uploaded image

```bash
python gui_predict.py
```

Features:

* Upload and test a facial image
* Predict class using trained model
* Display original + HOG image side-by-side
* Visualize decision path and important features

---

## 🧠 CT Principles in Action

| Principle            | Implementation Example                          |
| -------------------- | ----------------------------------------------- |
| Decomposition        | Separate modules for extraction, training, GUI  |
| Pattern Recognition  | HOG captures facial gradient and orientation    |
| Abstraction          | Images encoded as compact feature vectors       |
| Algorithmic Thinking | Decision Tree applies rule-based classification |

---

## Team Members

* Jawad Hussain
* Abdullah Siraj Khan
* Abdul Hadi Javed

---

