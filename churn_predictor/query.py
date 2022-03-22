import pickle

import numpy as np
import pandas as pd
from predictor import ChurnPredictor
from sklearn import preprocessing
from collections.abc import Iterable

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def build_vector(X: int, Y: int):
    vector = []
    for i in range(1, Y + 1):
        if i == (int)(X):
            vector.append(1)
            continue
        vector.append(0)
    return vector

QUESTIONS = [
    {
        "question": "Are you a senior citizen? (0 for no, 1 for yes)",
        "on_answer": lambda x: x,
    },
    {
        "question": "Do you live with a partner? (0 for no, 1 for yes)",
        "on_answer": lambda x: x,
    },
    {
        "question": "Do you have dependents? (0 for no, 1 for yes)",
        "on_answer": lambda x: x,
    },
    {
        "question": "How many months have you been with the service? (integer)",
        "on_answer": lambda x: x,
    },
    {
        "question": "Do you have phone service? (0 for no, 1 for yes)",
        "on_answer": lambda x: x,
    },
    {
        "question": "How many much do you pay monthly? (float)",
        "on_answer": lambda x: x,
    },
    {
        "question": "How many much have you payed in total? (float)",
        "on_answer": lambda x: x,
    },
    {
        "question": "Do you have multiple lines? (1 - No, 2 - No phone service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "What kind of internet service do you have? (1 - DSL, 2 - Fiber optic, 3 - No service)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "What kind of payment method do you use? (1 - Bank Transfer, 2 - Credit Card, 3 - Electronic Check, 4 - Mailed Check)",
        "on_answer": lambda x: build_vector(x, 4),
    },
    {
        "question": "Do you have online security? (1 - No, 2 - No internet service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "Do you have online backup? (1 - No, 2 - No internet service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "What kind of contract do you have? (1 - Month-to-month, 2 - One year, 3 - Two year)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "Do you have streaming movies service? (1 - No, 2 - No internet service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "Do you have streaming tv service? (1 - No, 2 - No internet service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "Do you have tech support service? (1 - No, 2 - No internet service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
    {
        "question": "Do you have device protection service? (1 - No, 2 - No internet service, 3 - Yes)",
        "on_answer": lambda x: build_vector(x, 3),
    },
]

if __name__ == "__main__":
    predictor = ChurnPredictor()
    scratch_logistic_model = predictor.logistic_model
    input_data = []

    with open("pickles/scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    if scratch_logistic_model or not scaler:
        i = 1
        for question in QUESTIONS:
            answer = input(str(i) + ". " + question["question"] + "\n")
            input_data.append(question["on_answer"](answer))
            i = i + 1
        input_data = pd.DataFrame(list(flatten(np.array(input_data, dtype=object)))).T

        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col])

        prediction = scratch_logistic_model.predict([scaler.transform(input_data)])[0]

        print(f"Prediction: {'Churn' if prediction else 'Not churn'}")
    else:
        raise Exception("Linear regression model not trained or invalid. Run the train script to fit it.")
