import torch
import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
df = pd.read_csv('data_product.csv')
features_list = []

for sentence in tqdm(df['processed_description'], desc="Processing descriptions"):
    # Tokenize the sentence and truncate if necessary
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=256)
    
    # Convert to tensor and add batch dimension
    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        outputs = phobert(input_ids)
        features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        features_list.append(features)

features_array = np.array(features_list)

np.save('text_feat-v1.npy', features_array)

print(f"Features saved to 'text_feat-v1.npy'")


