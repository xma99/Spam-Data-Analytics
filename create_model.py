import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle

# Read data
process_data = pd.read_csv('train_set.csv')
process_data.dropna(subset=['new_text'], inplace=True)
# From Bag of words
v = CountVectorizer()
# Read lines
X = v.fit_transform(process_data['new_text'])
# Read labels
y = process_data['type']
# Fitting the data to a logit function
x_num, val_x, y_num, val_y = train_test_split(X, y, test_size=0.2, random_state=40) # # Divide the training set
logistics = LogisticRegression(solver='liblinear', random_state=40)
# Solve the problem of too little spam data
ada = AdaBoostClassifier(base_estimator=logistics, n_estimators=100, random_state=40)

# Print output
ada.fit(x_num, y_num)
y_pred = ada.predict(val_x)
print(classification_report(val_y, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(val_y, y_pred))
print("Accuracy rate:")
print(accuracy_score(val_y, y_pred))

with open('v.pkl', 'wb') as f:
    pickle.dump(v, f)
with open('ada.pkl', 'wb') as f:
    pickle.dump(ada, f)
