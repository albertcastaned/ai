import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("email_dataset.csv")
data.head()

data.describe()