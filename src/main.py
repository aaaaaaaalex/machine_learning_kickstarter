import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, model_selection

import random
from datetime import datetime

NUM_TRAINING_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
DATA_PATH = "../assets/kickstarter-projects/ks-projects-201801.csv"


# reads data from a file, specify how many random rows to read with num_samples
def read_in_data(filename, total_rows, num_samples):
    # get a list of non-duplicating, random numbers that represent what rows to skip in the csv
    skip_rows = sorted( random.sample( range(total_rows), total_rows-num_samples ))

    data_file = open(filename)
    data = pd.read_csv(data_file,
            skiprows=skip_rows,
            usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12],
            names=[
                "id",
                "name",
                "category",
                "main_category",
                "currency",
                "deadline",
                "goal",
                "launched",
                "pledged",
                "state",
                "backers",
                "country",
                "usd_pledged"
                ])
    return data


# the 'launched' column's format is YYYY-MM-DD HH:MM:SS
# for simplicity, we will slice off the hours, minutes, and seconds
def convert_date_to_int (row):
    row = row.split(" ")[0]
    timestamp = datetime.strptime(row , "%Y-%m-%d").timestamp()
    return timestamp


def __main__():
    data = read_in_data('./assets/ks-projects-201801.csv', 379000, 7000)

    # after this line, we are left with columns: deadline, launched, goal, and state
    filtered = data.drop(['id', 'name', 'category', 'main_category', 'currency', 'pledged', 'backers', 'country', 'usd_pledged' ], axis=1)

    # convert string-type dates to numbers
    filtered['launched'] = filtered['launched'].apply(
        lambda row:
            convert_date_to_int(row)
    )
    filtered['deadline'] = filtered['deadline'].apply(
        lambda row:
            convert_date_to_int(row)
    )

    y = np.array( filtered['state'] )
    X = np.array( filtered.drop(['state'], axis=1) )

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=3)
    classifier = neighbors.KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    print("accuracy", accuracy)
    return

__main__()

