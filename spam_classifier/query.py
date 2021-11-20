import pickle

from train import NaiveBayes
from enum import Enum
from sklearn.tree import DecisionTreeClassifier

from utils import clean_email, transform_emails_to_set

USE_FRAMEWORKS = True

class QueryOption(Enum):
    CLI = 1
    FILE = 2
    QUIT = 3

if __name__ == '__main__':
    with open("scratch_bayes.pkl", "rb") as file:
        scratch_bayes: NaiveBayes = pickle.load(file)
    
    if not scratch_bayes:
        raise Exception("Scratch naive bayes model not exported. Please run train.py first")

    with open("decision_tree.pkl", "rb") as file:
        decision_tree: DecisionTreeClassifier = pickle.load(file)
    
    if not decision_tree:
        raise Exception("Decision tree model not exported. Please run train.py first")

    with open("vocabulary.pkl", "rb") as file:
        vocabulary: list = pickle.load(file)
    
    if not vocabulary:
        raise Exception("Vocabulary not exported. Please run train.py first")

    print("Models loaded succesfully")
    while True:
        try:
            option_select = int(input("Please select one option:\n1) Query from command line input\n2) Query from text file\n3) Quit\n\n"))
            option = QueryOption(option_select)
        except ValueError:
            print("\nInvalid option.\n")
            continue
        if option == QueryOption.CLI:
            query = input("Please type the email content: ")
            clean_query = clean_email(query)
            scratch_bayes_prediction = scratch_bayes.classify(clean_query)
            if USE_FRAMEWORKS:
                transformed_query = transform_emails_to_set([clean_query.split()], vocabulary)
                decision_tree_prediction = decision_tree.predict(transformed_query)[0]

        elif option == QueryOption.FILE:
            file_dir = input("Please type the file location: ")
            with open(file_dir, 'r') as file:
                query = file.read()
                print(f"File content: {query}\n")
            if query:
                clean_query = clean_email(query)
                scratch_bayes_prediction = scratch_bayes.classify(clean_query)
                if USE_FRAMEWORKS:
                    transformed_query = transform_emails_to_set([clean_query.split()], vocabulary)
                    decision_tree_prediction = decision_tree.predict(transformed_query)[0]
            else:
                print("Error reading file")
        elif option == QueryOption.QUIT:
            break
        print(f"\nScratch naive bayes prediction: {scratch_bayes_prediction}\n")
        if USE_FRAMEWORKS:
            print(f"Decision tree prediction: {decision_tree_prediction}\n")
    print("Exiting program...")