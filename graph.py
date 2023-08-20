import pandas as pd
import matplotlib.pyplot as plt

# Statistical data size of the training set
data = pd.read_csv('train_set.csv')
# Count the number of spam and ham data
data_count = data['type'].value_counts()

# Title
print("Data Comparison:")
print(data_count)

# Draw a pie chart to record the number of labels each
sizes = [data_count[1], data_count[0]]
labels = ['ham', 'spam']
colors = ['#ffa500', '#4169e1']
plt.pie(data_count, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Spam and Ham Sample Size')
# data.shape  # View data set size
# data.describe()  # Dataset Overview
plt.show()
