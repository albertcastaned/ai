import pandas as pd
import math
import time
import pickle

from pandas.core.frame import DataFrame
from utils import SKIP_WORDS, clean_data, transform_set, transform_emails_to_set
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class NaiveBayes(object):
    def classify(self, message: str) -> str:
        message = message.split()

        p_spam_given_message = math.log(self.percentage_spam)
        p_ham_given_message = math.log(self.percentage_ham)

        for word in message:
            if word in self.parameters_spam:
                p_spam_given_message += math.log(self.parameters_spam[word])

            if word in self.parameters_ham:
                p_ham_given_message += math.log(self.parameters_ham[word])

        if p_ham_given_message > p_spam_given_message:
            return 'ham'
        elif p_ham_given_message < p_spam_given_message:
            return 'spam'
        else:
            return 'equal'

    def test_accuracy(self, test_set: DataFrame) -> tuple[str, DataFrame]:
        print("Testing naive bayes accuracy...")
        test_set['PREDICTED'] = test_set['TEXT'].apply(self.classify)
        correct = 0
        real_spam = 0
        real_ham = 0
        spam = 0
        ham = 0
        equal = 0
        total = test_set.shape[0]
        for row in test_set.iterrows():
            row = row[1]
            if row['LABEL'] == 'spam':
                real_spam += 1
            if row['LABEL'] == 'ham':
                real_ham += 1
            if row['PREDICTED'] == 'spam':
                spam += 1
            if row['PREDICTED'] == 'ham':
                ham += 1
            if row['PREDICTED'] == 'equal':
                equal += 1
            if row['LABEL'] == row['PREDICTED']:
                correct += 1
        return correct/total, test_set['PREDICTED']

    def train(self, feature_set, vocabulary) -> None:
        spam_emails = feature_set[feature_set['LABEL'] == 'spam']
        ham_emails = feature_set[feature_set['LABEL'] == 'ham']

        self.percentage_spam = len(spam_emails) / len(feature_set)
        self.percentage_ham = len(ham_emails) / len(feature_set)

        n_spam = spam_emails['WORD_COUNT'].sum()
        n_ham = ham_emails['WORD_COUNT'].sum()
        n_vocabulary = len(vocabulary)
        alpha = 1
        self.parameters_spam = {unique_word:0 for unique_word in vocabulary}
        self.parameters_ham = {unique_word:0 for unique_word in vocabulary}

        dividend_spam = (n_spam + alpha*n_vocabulary)
        dividend_ham = (n_ham + alpha*n_vocabulary) 
        for word in vocabulary:
            n_word_given_spam = spam_emails[word].sum()
            p_word_given_spam = (n_word_given_spam + alpha) / dividend_spam
            self.parameters_spam[word] = p_word_given_spam

            n_word_given_ham = ham_emails[word].sum()
            p_word_given_ham = (n_word_given_ham + alpha) / dividend_ham
            self.parameters_ham[word] = p_word_given_ham

if __name__ == '__main__':
    # Read dataset
    data = pd.read_csv(r"spam_ham_dataset.csv")
    # Clean dataset preprocessing
    data = clean_data(data)

    # Create training set and test set from random sample
    # Train 80%, Test 20% of random sample
    data_randomized = data.sample(frac=1)

    print("Using " + str(len(data_randomized)) + " samples from dataset")
    training_test_index = round(len(data_randomized) * 0.8)

    training_set = data_randomized[:training_test_index].reset_index(drop=True)
    test_set = data_randomized[training_test_index:].reset_index(drop=True)
    
    print("Training set " + str(len(training_set)/len(data_randomized) * 100.00) + "% of dataset")
    print("Test set " + str(len(test_set)/len(data_randomized) * 100.00) + "% of dataset")

    print("Transforming data set...")
    feature_set, vocabulary = transform_set(training_set)
    print("Saving vocacbulary to pickle...")
    with open("vocabulary.pkl", "wb") as file:
        pickle.dump(vocabulary, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Vocabulary saved to vocabulary.pkl\n")

    print("--- Training Naive Bayes model ---")
    naiveBayes = NaiveBayes()
    start_time = time.time()
    print("Starting training...")
    naiveBayes.train(feature_set, vocabulary)
    naive_train_time = str(time.time() - start_time)
    print("Finished training - " + naive_train_time + " seconds")
    accuracy = naiveBayes.test_accuracy(test_set)
    print('Naive Bayes accuracy: ', accuracy[0])
    print('Saving Naive bayes model to pickle...')
    
    with open("scratch_bayes.pkl", "wb") as file:
        pickle.dump(naiveBayes, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Scratch naive bayes model exported!")
    # Compare with other models
    print("---- Framework models comparison ---")
    print("-- Decision tree --")
    x_train = feature_set.iloc[:, 2:]
    y_train = feature_set.iloc[:, 0]

    start_time = time.time()
    print("Starting training...")
    tree_clf = DecisionTreeClassifier(
        criterion="entropy"
    )
    tree_clf.fit(x_train, y_train)
    print("Finished training - " + str(time.time() - start_time) + " seconds")

    word_counts_per_email = {unique_word: [0] * 1 for unique_word in vocabulary}

    test_set['TEXT'] = test_set['TEXT'].str.split()
    start_time = time.time()
    print("Transforming test set...")
    x_test = transform_emails_to_set(test_set['TEXT'], vocabulary)
    y_test = test_set.iloc[:, 0]
    print("Transformed test set - " + str(time.time() - start_time) + " seconds")

    y_pred = tree_clf.predict(x_test)

    print("Decision tree accuracy: " + str(accuracy_score(y_pred, y_test)))

    print("Saving decision tree model to pickle...")
    with open("decision_tree.pkl", "wb") as file:
        pickle.dump(tree_clf, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Decision tree model exported!")

    print("-------------")
    print("-- Scikit Naive Bayes --")
    start_time = time.time()
    print("Starting training...")
    scikit_naive_bayes = MultinomialNB()
    scikit_naive_bayes.fit(x_train, y_train)
    print("Finished training - " + str(time.time() - start_time) + " seconds")
    y_pred = scikit_naive_bayes.predict(x_test)
    print("Naive bayes scikit accuracy: " + str(accuracy_score(y_pred, y_test)))

    print("-------------")
    print("\nFinished training models\n")