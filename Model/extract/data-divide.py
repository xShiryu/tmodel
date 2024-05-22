import pandas as pd
import numpy as np
import torch
from datetime import datetime

df = pd.read_csv('data_reviews_purchase.csv')

df = df.sort_values(by=['user_id'])

# Count occurrences of each user_id
user_counts = df['user_id'].value_counts()

# Filter user_ids that appeared 3 times or more
filtered_user_ids = user_counts[user_counts >= 2].index.tolist()

# Filter the DataFrame based on the selected user_ids
temp_df = df[df['user_id'].isin(filtered_user_ids)]
extracted_df = temp_df[['user_id', 'product_id', 'rating', 'cmt_date']]
extracted_df.rename(columns={'cmt_date': 'timestamp'}, inplace=True)
extracted_df.rename(columns={'user_id': 'userID'}, inplace=True)
extracted_df.rename(columns={'product_id': 'itemID'}, inplace=True)
extracted_df['timestamp'] = extracted_df['timestamp'].apply(lambda x: int(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()))
extracted_df['userID'], _ = pd.factorize(extracted_df['userID'])

x_labels = []
count = extracted_df['userID'].value_counts()

count_array = [count.get(user_id, 0) for user_id in range(count.index.min(), count.index.max() + 1)]

for i in count_array:
    if i < 3:
        for j in range(i):
            x_labels.append(0)
    else:
        for j in range(i-2):
            x_labels.append(0)
        x_labels.append(1)
        x_labels.append(2)

extracted_df['x_label'] = x_labels



df_sorted = extracted_df.sort_values(by=['userID', 'timestamp'])
index = df_sorted.groupby(['userID', 'itemID'])['timestamp'].idxmax()
df_no_dup = df_sorted.loc[index, ['userID', 'itemID', 'timestamp']].sort_values(by=['userID', 'timestamp'], ascending=False)

list_user = df_no_dup['userID'].value_counts().index[:3952]
df_purchase = df_no_dup[df_no_dup['userID'].isin(list_user)]

df_purchase['x_label'] = np.arange(len(df_purchase)) % 3
val_indices = sorted(df_purchase[df_purchase['x_label'] == 1].index.tolist())
test_indices = sorted(df_purchase[df_purchase['x_label'] == 2].index.tolist())


extracted_df['x_label'] = 0
extracted_df.loc[val_indices, 'x_label'] = 1
extracted_df.loc[test_indices, 'x_label'] = 2

print(len(val_indices))
print(len(test_indices))
extracted_df.to_csv('skincare-indexed-final.inter', index=False, sep='\t')

df_purchase.to_csv('check.inter', index=False, sep='\t')
