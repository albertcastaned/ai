import matplotlib.pyplot as plt
from os import getcwd
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class SpamClassifier(object):
    def __init__(self):
        self.data = pd.read_csv(f"{getcwd()}/spambase.csv")

        x = self.data.iloc[: ,:-1]
        y = self.data.iloc[:, -1:]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/5)

        self.tree_clf = DecisionTreeClassifier(criterion="entropy")
        self.tree_clf.fit(x_train,y_train)


        y_pred = self.tree_clf.predict(x_test)

        print("Model accuracy score: " + str(accuracy_score(y_pred, y_test)))

    def query(self, query):
        df = pd.DataFrame(query, columns=list(self.data.columns[:-1]))
        prediction = self.tree_clf.predict(df)
        if prediction[0] == 0:
            print("Email classified as not spam")
        else:
            print("Email classified as spam")