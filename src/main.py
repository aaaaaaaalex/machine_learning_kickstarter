import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, model_selection, naive_bayes
from sklearn.feature_extraction import text as sktext

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
            #skiprows=skip_rows,
            skiprows=total_rows-num_samples,
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


# normalize numerical values in a np-array
def normalize_array(ndarr):
    ndarr /= np.max(np.abs(ndarr), axis=0)
    return ndarr



def preproccess_knn_data(df):
    # after this line, we are left with columns: deadline, launched, goal, and state
    filtered = df.drop(['id', 'name', 'category', 'main_category', 'currency', 'pledged', 'backers', 'country', 'usd_pledged' ], axis=1)

    # convert string-type dates to numbers and normalize them
    filtered['launched'] = filtered['launched'].apply( convert_date_to_int )
    filtered['launched'] = normalize_array( filtered['launched'].values )

    filtered['deadline'] = filtered['deadline'].apply( convert_date_to_int )
    filtered['deadline'] = normalize_array( filtered['deadline'].values )

    filtered['goal'] = normalize_array( filtered['goal'].values )

    # reduce the columns 'deadline' and 'launched' to a single column 'alotted_time', which disregards
    #     time of year, and represents the time-delta for-which the project is allowed to be active
    filtered['alotted_time'] = filtered['deadline'] - filtered['launched']
    filtered = filtered.drop(['deadline', 'launched'], axis=1)

    filtered['state'] = (filtered['state'] == "successful" )

    num_pos = len( filtered[ (filtered['state'] == True)] )
    num_entries = len(filtered['state'])
    print("percentage successful projects..." , num_pos/num_entries)

    return filtered

def preproccess_nb_data(df):
    # drop unusable columns, leaving us with columns: category, main_category, name, currency, country
    filtered = df.drop(['id', 'pledged', 'backers', 'usd_pledged', 'deadline', 'launched', 'goal', 'category', 'main_category', 'currency', 'country'], axis=1)

    filtered['state'] = (filtered['state'] == "successful")
    return filtered



# takes a dataframe of data relevant to KNN classification and returns a trained classifier
def train_classifier(df, classifier_type):
    # split data into attributes and labels, train and test sets
    y = np.array( df['state'] )
    X = np.array( df.drop(['state'], axis=1) )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=3)
    
    # train
    if classifier_type is 'KNN':
        classifier = neighbors.KNeighborsClassifier(n_neighbors=5)
    elif classifier_type is "NB":
        classifier = naive_bayes.BernoulliNB()

        cvctr = sktext.CountVectorizer()
        tfid = sktext.TfidfTransformer()

        temp = cvctr.fit_transform( np.reshape(X_train, -1) )
        X_train = tfid.fit_transform(temp)

        X_test = cvctr.transform( np.reshape(X_test, -1) )


    classifier.fit(X_train, y_train)

    # test
    accuracy = classifier.score( X_test, y_test )

    return classifier, accuracy




def __main__():
    # read data and preprocess it
    data = read_in_data('./assets/ks-projects-201801.csv', 379000, 7000)
    knn_filtered = preproccess_knn_data(data)
    nb_filtered = preproccess_nb_data(data)
    
    # train / test
    knn_classifier, knn_accuracy = train_classifier(knn_filtered, "KNN")
    bayes_classifier, bayes_accuracy = train_classifier(nb_filtered, "NB")




    print("KNN accuracy", knn_accuracy)
    print("Bayes accuracy", bayes_accuracy)
    return

__main__()

