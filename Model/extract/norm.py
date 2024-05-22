import numpy as np

image_feature_vectors = np.load('image_feat.npy')
text_feature_vectors = np.load('text_feat-v1.npy')


# Step 2: Find the minimum and maximum values of the entire array
image_min_val = image_feature_vectors.min()
image_max_val = image_feature_vectors.max()

text_min_val = text_feature_vectors.min()
text_max_val = text_feature_vectors.max()

# Step 3: Normalize the feature vectors using min-max scaling
normalized_image_feature_vectors = -1 + ((image_feature_vectors - image_min_val) / (image_max_val - image_min_val)) * 2
normalized_text_feature_vectors = -1 + ((text_feature_vectors - text_min_val) / (text_max_val - text_min_val)) * 2

# Optional: Save the normalized feature vectors to a new .npy file
np.save('image_feat-v2.npy', normalized_image_feature_vectors)
np.save('text_feat-v2.npy', text_feature_vectors)


print("Normalization complete. The feature vectors are now in the range (-1, 1).")
