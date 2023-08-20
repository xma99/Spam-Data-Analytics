from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# From train model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('ada_boost.pkl', 'rb') as f:
    ada_boost = pickle.load(f)

test_data = pd.read_csv('test_set.csv')
test_data.dropna(subset=['new_text'], inplace=True)
X_test = vectorizer.transform(test_data['new_text'])
y_test = test_data['type']
y_test_pred = ada_boost.predict(X_test)
# Count totalmistakes
correct = (y_test == y_test_pred).sum()
incorrect = (y_test != y_test_pred).sum()
# check acc sore
print("Accuracy rate:")
print(accuracy_score(y_test, y_test_pred))
# Each mistakes
count_mistakes = test_data[y_test != y_test_pred].copy()
count_mistakes['predicted_type'] = y_test_pred[y_test != y_test_pred]
ham_mistakes = count_mistakes[count_mistakes['type'] == 1].shape[0]
spam_mistakes = count_mistakes[count_mistakes['type'] == 0].shape[0]

with open('countMistakes.txt', 'w', encoding='utf-8') as f:
    for index, row in count_mistakes.iterrows():
        f.write(f"{index}, True: {row['type']}, Predicted: {row['predicted_type']}, Text: {row['new_text']}\n")

    f.write(f"\nNumber of Classification Ham count_mistakes: {ham_mistakes}\n")
    f.write(f"Number of Classification Spam count_mistakes: {spam_mistakes}\n")
