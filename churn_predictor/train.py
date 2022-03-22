import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from predictor import ChurnPredictor

if __name__ == "__main__":
    predictor = ChurnPredictor()
    dataset = predictor.dataset

    logistic_model = predictor.logistic_model
    
    x = dataset.loc[:, dataset.columns != 'Churn'].values
    y = dataset['Churn']

    # Split test and train data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2,
    )
    scaler = preprocessing.StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Save scaler as pickle
    file = open("scaler.pkl", 'wb')
    pickle.dump(scaler, file, 2)
    file.close()

    logistic_model.fit(x_train, y_train)
    predictions = logistic_model.predict(x_test)

    print(f"Logistic Regression Hand Implementation Accuracy: {predictor.accuracy(y_test, predictions) * 100} %")

    scikit_logistic_regression_model = predictor.scikit_logistic_regression_model.fit(x_train, y_train)
    scikit_predictions = scikit_logistic_regression_model.predict(x_test)

    print(f"SciKit Logistic Regression Implementation Accuracy: {predictor.accuracy(y_test, scikit_predictions) * 100} %")
    
    scikit_decision_tree_model = predictor.scikit_decision_tree.fit(x_train, y_train)
    scikit_dt_predictions = scikit_decision_tree_model.predict(x_test)

    print(f"SciKit Decision Tree Implementation Accuracy: {predictor.accuracy(y_test, scikit_dt_predictions) * 100} %")
