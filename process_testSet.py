import pandas as pd
import re

# Read date from test set
test_data = pd.read_csv('set2.csv')

# Convert all letters in the text to lowercase
# Clear the text of special characters and keep the letters and numbers
def test_normalize(text):
    text = text.lower()
    text = re.sub('[^a-z0-9]+', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()

# Input to the test_set.csv and will represent ham and spam with 1 and 0 respectively
test_data['new_text'] = test_data['text'].apply(test_normalize)
test_data['type'] = test_data['type'].map({'ham': 1, 'spam': 0})
test_data = test_data.drop('text', axis=1) # Without unprocessed data
test_data.to_csv('test_set.csv', index=False, header=True)