import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, model_selection, naive_bayes
from sklearn.feature_extraction import text as sktext

import random
from datetime import datetime

NUM_TRAINING_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
DATA_PATH = "../assets/kickstarter-projects/ks-projects-201801.csv"
RANDOM_SEED = 42


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



def preproccess_data(df):
    # remove columns that are not relevant to the use-case
    # the remaining columns are: name, category, main_category, currency, deadline, goal, launched, state, country
    filtered = df.drop(['id', 'pledged', 'backers', 'usd_pledged'], axis=1)

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


def split_dataframe(df, rand_seed=RANDOM_SEED):
    # split data into attributes and labels, train and test sets
    y = np.array( df['state'] )
    X = np.array( df.drop(['state'], axis=1) )
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=rand_seed)

    return X_train, X_test, y_train, y_test



# takes a dataframe of data relevant to KNN classification and returns a trained classifier
def train_classifier(df, classifier_type):
    X_train, X_test, y_train, y_test = split_dataframe(df)


    # train
    if classifier_type is 'KNN':
        print(X_train)
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



def ensemble_classify(knn_classifier, nb_classifier, knn_df, nb_df):
    knn_X_train, knn_X_test, knn_y_train, knn_y_test = split_dataframe(knn_df)
    nb_X_train,  nb_X_test,  nb_y_train,  nb_y_test  = split_dataframe(nb_df)

    knn_results = knn_classifier.predict(knn_X_test)

    # prepare naive bayes input data
    cvctr = sktext.CountVectorizer()
    temp = cvctr.fit_transform( np.reshape(nb_X_train, -1) ) # this step must be done to give CountVectorizer the same vocabulary as in previous tests
    nb_X_test = cvctr.transform( np.reshape(nb_X_test, -1) )

    nb_results = nb_classifier.predict(nb_X_test)

    print("knn_results", knn_results)
    print("nb_results", nb_results)

    return 1



def __main__():
    # read data and preprocess it
    data = read_in_data('./assets/ks-projects-201801.csv', 379000, 7000)
    data_filtered = preproccess_data(data)    

    # after this line, we are left with columns: deadline, launched, goal, and state
    knn_filtered = data_filtered.copy(deep=True)
    knn_filtered = knn_filtered.drop(['name', 'category', 'main_category', 'currency','country'], axis=1)

    #nb_filtered has columns: category, main_category, name, currency, country, state
    nb_filtered = data_filtered.copy(deep=True)
    nb_filtered = nb_filtered.drop(['alotted_time', 'goal', 'category', 'main_category', 'currency', 'country' ], axis=1)
    
    # train / test
    knn_classifier, knn_accuracy = train_classifier(knn_filtered, "KNN")
    nb_classifier, bayes_accuracy = train_classifier(nb_filtered, "NB")

    # test ensemble classification
    ensemble_accuracy = ensemble_classify(knn_classifier, nb_classifier, knn_filtered, nb_filtered)

    print("KNN accuracy", knn_accuracy)
    print("Bayes accuracy", bayes_accuracy)
    print("Ensemble Accuracy", ensemble_accuracy)
    return

__main__()

