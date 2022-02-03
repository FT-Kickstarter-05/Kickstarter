import pandas as pd
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from sys import prefix

data = pd.read_csv('https://raw.githubusercontent.com/FT-Kickstarter-05/Kickstarter/main/2018_ks_data.csv')

"""# Cleaning and preprocessing """

unneeded_columns = ['ID', 'name']
data = data.drop(unneeded_columns, axis=1)

data['usd pledged'] = data['usd pledged'].fillna(data['usd pledged'].mean())

data = data.drop(data.query("state != 'failed' and state != 'successful'").index, axis=0).reset_index(drop=True)


"""# Feature Engineering and Encoding
### Change Date to month and year and delete date
### For deadline and launched dates
"""

data['deadline'].apply(lambda x: x[0:4])

data['deadline'].apply(lambda x: x[5:7])

data['deadline_year'] = data['deadline'].apply(lambda x: np.float(x[0:4]))
data['deadline_month'] = data['deadline'].apply(lambda x: np.float(x[5:7]))

data['launched_year'] = data['launched'].apply(lambda x: np.float(x[0:4]))
data['launched_month'] = data['launched'].apply(lambda x: np.float(x[5:7]))

data = data.drop(['deadline', 'launched'], axis=1)

"""### Binary encode 'state' column"""

data['state'].apply(lambda x: 1 if x == 'successful' else 0)

data['state'] = data['state'].apply(lambda x: 1 if x == 'successful' else 0)


"""### OneHotEncode 'category','main_category'
### 'currency'	, 'country'
"""


def onehot_encode(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

data = onehot_encode(
    data,
    ['category', 'main_category', 'currency', 'country'],
    ['cat', 'main_cat', 'curr', 'country']
)
"""# Our Data is fully numeric
### We are ready to split and scale the data
"""
print(list(data.columns))