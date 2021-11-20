import pickle
import pandas as pd
import re
import sys
import os, psutil
from pandas.core.frame import DataFrame
from sklearn.feature_extraction.text import CountVectorizer

SKIP_WORDS = ['subject', 'enron', 'etc', 'ect', 'hou', 'nbsp', 'cc', 'will']

def clean_data(emails) -> DataFrame:
    skip_words_regex = r'\b({})\b'.format('|'.join(SKIP_WORDS))
    emails['TEXT'] = emails['TEXT'].str.replace(r'\W+', ' ', regex=True).str.replace(r'\d+', ' ', regex=True).str.lower().str.replace(r'\b\w\b', ' ', regex=True).str.replace(skip_words_regex, ' ', regex=True)
    return emails

def transform_set(training_set):
        print("making feature set")
        cv = CountVectorizer()
        cv.fit(training_set['TEXT'])
        results = cv.transform(training_set['TEXT'])
        features = cv.get_feature_names_out()
        feature_set = pd.DataFrame(results.toarray(), columns=features)
        n_words = training_set['TEXT'].apply(len)
        vocabulary = list(feature_set.columns)
        feature_set.insert(0, 'LABEL', training_set['LABEL'], True)
        feature_set.insert(1, 'WORD_COUNT', n_words, True)
        print("finished feature set.")
        return feature_set, vocabulary

def transform_emails_to_set(emails, vocabulary):
    word_counts_per_email = {unique_word: [0] * len(emails) for unique_word in vocabulary}
    for index, email in enumerate(emails):
        for word in email:
            if word in word_counts_per_email:
                word_counts_per_email[word][index] += 1

    word_counts = DataFrame(word_counts_per_email)

    return word_counts

def clean_email(email: str) -> str:
    skip_words_regex = r'\b({})\b'.format('|'.join(SKIP_WORDS))
    # Lowercase string
    result = email.lower()
    # Remove skip words
    result = re.sub(skip_words_regex, ' ', result)
    result = re.sub(r'\W+', ' ', result)
    # Remove numbers
    result = re.sub(r'\d+', ' ', result)
    # Remove one letter words
    result = re.sub(r'\b\w\b', ' ', result)
    return result