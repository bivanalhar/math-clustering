"""
This program is intended to classify all the mathematical problems
by using the supervised learning method. The labelling consists of 3 labels:
Calculus, Linear Algebra and Probability & Statistics
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import csv

"""
Model to be implemented : Long Short-Term Memory (LSTM)
which is believed to be able to depict the continuation of the problem's story
"""

df = pd.read_csv("./eng_linalg.csv", delimiter = ",", encoding = "latin-1")
df.head()