from os import getcwd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def avg_char_capital_uninterrupted(content):
    lengths_uninterrupted = []
    temp_count = 0
    for char in content:
        if char.isalpha() and char.isupper():
            temp_count = temp_count + 1
        else:
            lengths_uninterrupted.append(temp_count)
            temp_count = 0
    print(lengths_uninterrupted)
    return sum(lengths_uninterrupted) / len(lengths_uninterrupted)    

print(avg_char_capital_uninterrupted('Hola como ESTAS JAJAJ$ estoy viendo algo RARO'))

data = pd.read_csv(f"{getcwd()}/spambase.csv")

x = data.iloc[: ,:-1]
y = data.iloc[:, -1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5)

tree_clf = DecisionTreeClassifier()
tree_clf.fit(x_train,y_train)


y_pred = tree_clf.predict(x_test)

print("Accuracy score: " + str(accuracy_score(y_pred, y_test)))

