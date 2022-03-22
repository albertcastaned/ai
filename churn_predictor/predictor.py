import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression as ScikitLR
from logistic_regression import LogisticRegression
from sklearn import tree

class ChurnPredictor:
    
    def __init__(self) -> None:
        loaded_dataset = self._load_dataset_()
        self.dataset = self._pre_process_dataset_(loaded_dataset)
        self.logistic_model = LogisticRegression(epochs=8000)
        self.scikit_logistic_regression_model = ScikitLR(max_iter=8000)
        self.scikit_decision_tree = tree.DecisionTreeClassifier()

    def _pre_process_dataset_(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # One Hot Encoding Time for categorical features
        DUMMY_DICT = [
            {
                "column": dataset.MultipleLines,
                "prefix": "MultipleLines"
            },
            {
                "column": dataset.InternetService,
                "prefix": "InternetService",
            },
            {
                "column": dataset.PaymentMethod,
                "prefix": "PaymentMethod",
            },
            {
                "column": dataset.OnlineSecurity,
                "prefix": "OnlineSecurity",
            },
            {
                "column": dataset.OnlineBackup,
                "prefix": "OnlineBackup",
            },
            {
                "column": dataset.Contract,
                "prefix": "Contract",
            },
            {
                "column": dataset.StreamingMovies,
                "prefix": "StreamingMovies",
            },
            {
                "column": dataset.StreamingTV,
                "prefix": "StreamingTV",
            },
            {
                "column": dataset.TechSupport,
                "prefix": "TechSupport",
            },
            {

                "column": dataset.DeviceProtection,
                "prefix": "DeviceProtection",
            },
        ]
        
        BINARY_COLUMNS = [
            "Partner",
            "Dependents",
            "PhoneService",
            "Churn"
        ]
        DROP_COLUMNS = [
            "gender",
            "customerID",
            "PaperlessBilling",
        ]

        for dummy_element in DUMMY_DICT:
            dummy = pd.get_dummies(dummy_element["column"], prefix=dummy_element["prefix"])
            dataset = dataset.drop(columns=dummy_element["prefix"])
            dataset = dataset.join(dummy)

        # Transform "Yes/No" features to binary
        for col in BINARY_COLUMNS:
            dataset[col] = dataset[col].map({'Yes' : 1, 'No' : 0})

        # Drop irrelevant columns
        for col in DROP_COLUMNS:
            dataset = dataset.drop(columns=col)

        # Clean empty total charges column
        # Making asumption that total charge = tenure * monthly charge

        dataset["TotalCharges"] = dataset["TotalCharges"].replace(
            " ", None
        )
        dataset["TotalCharges"] = dataset["TotalCharges"].fillna(
            dataset.MonthlyCharges * dataset.tenure
        )

        dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"])

        return dataset

    def _load_dataset_(self) -> pd.DataFrame:
        return pd.read_csv("churn_dataset.csv")

    def accuracy(self, y_real, y_prediction):
        accuracy = np.sum(y_real == y_prediction) / len(y_real)
        return accuracy
    
