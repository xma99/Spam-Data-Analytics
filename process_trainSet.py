import pandas as pd
import re

# Read date from test set
train_data = pd.read_csv('set1.csv', names=['combined'], header=None)

# Since this is an unlabeled train_data, you need to label each column of the train_data first
def labels(row):
    parts = row.split('\t', 1)
    return pd.Series([parts[0], parts[1]])
train_data[['type', 'text']] = train_data['combined'].apply(labels)

# Convert all letters in the text to lowercase
# Clear the text of special characters and keep the letters and numbers
def train_normalize(text):
    text = text.lower()
    text = re.sub('[^a-z0-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

# Input to the test_set.csv and will represent ham and spam with 1 and 0 respectively
train_data['new_text'] = train_data['text'].apply(train_normalize)
train_data['type'] = train_data['type'].map({'ham': 1, 'spam': 0})
train_data = train_data.drop(['combined', 'text'], axis=1) # Without unprocessed train_data
train_data.to_csv('train_set.csv', index=False)
