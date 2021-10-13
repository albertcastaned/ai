import sys
from os import getcwd
import pandas as pd


def avg_char_capital_uninterrupted(content):
    lengths_uninterrupted = []
    temp_count = 0
    for char in content:
        if char.isalpha() and char.isupper():
            temp_count = temp_count + 1
        else:
            if temp_count != 0:
                lengths_uninterrupted.append(temp_count)
            temp_count = 0

    if temp_count > 0:
        lengths_uninterrupted.append(temp_count)
    return sum(lengths_uninterrupted) / len(lengths_uninterrupted)    


if len(sys.argv) != 2:
    print("usage: query.py <email_text_file>")
else:
    with open(sys.argv[1], 'r') as file:
        content = file.read()
        content_split = content.split()
        dataset = pd.read_csv(f"{getcwd()}/spambase.csv")
        words_count_columns = list(dataset.columns)[:-10]

        word_count_data = [
            content_split.count(word_count) for word_count in words_count_columns
        ]

        char_freq_columns = list(dataset.columns)[48:54]
        content_length = len(content)

        char_count_data = [
            100 * content.count(char) / content_length
            for char in char_freq_columns
        ]
        print(char_count_data)