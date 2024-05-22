import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50

product_data = pd.read_csv('data_product.csv')

model = resnet50(pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_features = []
for index, row in product_data.iterrows():
    image_path = os.path.join('image', row['image_path_jpg'])
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        image = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            feature = model(image)
        image_features.append(feature.squeeze().numpy())
        print(f"Process image for product: {row['product_id']} success.")
    else:
        print(f"Image not found for product ID: {row['product_id']}")

image_features = np.array(image_features)

print(image_features)
print(image_features.size)
rows, cols = image_features.shape
print(rows)
print(cols)

for row in image_features:
    for element in row:
        print(element, end = ' ')
    break

#np.save('image_feat.npy', image_features)