from feature_extraction import extract_hog_features
import matplotlib.pyplot as plt

img_path = r"C:\Users\jawad\Downloads\Processed_Dataset_2\01\frame_00001.jpg"
features, hog_img = extract_hog_features(img_path, visualize=True)

print("HOG feature vector length:", len(features))

plt.imshow(hog_img, cmap='gray')
plt.title("HOG Visualization")
plt.axis('off')
plt.show()
