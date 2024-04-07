import os
import wget
import zipfile

import pandas as pd
from absl import logging

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv("/Users/shuaikeliu/Downloads/smsspamcollection/SMSSpamCollection", delimiter="\t", encoding="latin-1", header=None)

print(dataset.head())