import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors, model_selection, naive_bayes
from sklearn.feature_extraction import text as sktext

import random
from datetime import datetime
import tkinter
import matplotlib.pyplot as plt

NUM_TRAINING_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_NEIGHBOURS = 5
DATA_PATH = "../assets/kickstarter-projects/ks-projects-201801.csv"
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = .5

# save fitted-vectorizer
COUNT_VECTOR = {}



# reads data from a file, specify how many random rows to read with num_samples
def read_in_data(filename, total_rows, num_samples):
    # get a list of non-duplicating, random numbers that represent what rows to skip in the csv
    skip_rows = sorted( random.sample( range(total_rows), total_rows-num_samples ))

    data_file = open(filename)
    data = pd.read_csv(data_file,
            skiprows=skip_rows,
            #skiprows=total_rows-num_samples,
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
    filtered = filtered.dropna()

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
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=rand_seed)

    return X_train, X_test, y_train, y_test



# takes a dataframe of data relevant to KNN classification and returns a trained classifier
def train_classifier(df, classifier_type):
    global COUNT_VECTOR
    global KNN_ACCURACY
    global NB_ACCURACY
    X_train, X_test, y_train, y_test = split_dataframe(df)


    # train
    if classifier_type is 'KNN':
        classifier = neighbors.KNeighborsClassifier(n_neighbors=NUM_NEIGHBOURS)

    elif classifier_type is "NB":
        classifier = naive_bayes.BernoulliNB()

        cvctr = sktext.CountVectorizer()
        tfid = sktext.TfidfTransformer()

        temp = cvctr.fit_transform( np.reshape(X_train, -1) )
        COUNT_VECTOR = cvctr #save the fitted vectorizer

        X_train = tfid.fit_transform(temp)
        X_test = cvctr.transform( np.reshape(X_test, -1) )

    classifier.fit(X_train, y_train)

    # test
    accuracy = classifier.score( X_test, y_test )

    if classifier_type is "NB":
        NB_ACCURACY = accuracy
    elif classifier_type is "KNN":
        KNN_ACCURACY = accuracy

    return classifier, accuracy


def get_num_true_positives(cls_predictions, cls_labels):
    # filter answers that were correctly True
    temp_answers_correct = (cls_predictions == cls_labels)
    temp_correct_answers = cls_predictions[ temp_answers_correct ]
    true_pos = len( temp_correct_answers[ temp_correct_answers ])

    return true_pos

def get_num_false_positives(cls_predictions, cls_labels):
    # filter answers that were incorrectly True
    temp_answers_incorrect = (cls_predictions != cls_labels)
    temp_incorrect_answers = cls_predictions[ temp_answers_incorrect ]
    false_pos = len(temp_incorrect_answers[ temp_incorrect_answers ])

    return false_pos


# evaluate and compare both KNN and NB algorithms
def evaluate_algorithms(knn_classifier, nb_classifier, knn_df, nb_df):
    global COUNT_VECTOR

    knn_X_train, knn_X_test, knn_y_train, knn_y_test = split_dataframe(knn_df)
    nb_X_train,  nb_X_test,  nb_y_train,  nb_y_test  = split_dataframe(nb_df)

    # prepare naive bayes input data
    cvctr = COUNT_VECTOR
    nb_X_test = cvctr.transform( np.reshape(nb_X_test, -1) )

    #---------------------------------------------------------average the probabilities of both classifiers
    nb_answers = nb_classifier.predict(nb_X_test)
    knn_answers = knn_classifier.predict(knn_X_test)

    # get precision
    nb_precision  = np.mean( nb_answers == nb_y_test )
    knn_precision = np.mean(knn_answers == knn_y_test)

    #get True Positives and Recall values
    nb_true_pos  = get_num_true_positives ( nb_answers, nb_y_test)
    nb_false_pos = get_num_false_positives( nb_answers, nb_y_test)
    nb_recall = nb_true_pos / ( nb_true_pos + nb_false_pos )    

    knn_true_pos  = get_num_true_positives ( knn_answers, knn_y_test)
    knn_false_pos = get_num_false_positives( knn_answers, knn_y_test)
    knn_recall = knn_true_pos / ( knn_true_pos + knn_false_pos )

    # determine F-score
    knn_f = 2 * ( (knn_recall * knn_precision) / (knn_recall + knn_precision) )
    nb_f  = 2 * ( (nb_recall * nb_precision) / (nb_recall + nb_precision) )

    return knn_recall, nb_recall, knn_f, nb_f


def plot_barchart(knn_scores, nb_scores):
    x =  [ u'KNN Precision', u'KNN Recall Value', u'KNN F-Score', u'']
    y =  [ knn_scores['accuracy'], knn_scores['recall'], knn_scores['fscore'], 0]

    x1 = [ u'Naive Bayes Precision', u'Naive Bayes Recall Value', u'Naive Bayes F-Score']
    y1 = [ nb_scores['accuracy'],  nb_scores['recall'],  nb_scores['fscore']]

    plt.bar(x, y, label="KNN precision, recall, and f-score")
    plt.bar(x1, y1, label="Naive Bayes precision, recall and f-score")

    plt.title("Comparison of Accuracy, Recall Value, and F-Score of K-NN and Naive Bayes Algorithms")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

    return

# return number of words in sentence
def wordlen(sentence):
    return len(sentence.split(' '))

def __main__():
    # read data and preprocess it
    data = read_in_data('./assets/ks-projects-201801.csv', 379000, 7000)
    data_filtered = preproccess_data(data)    

    # after this line, we are left with columns: alotted_time, goal, and state
    knn_filtered = data_filtered.copy(deep=True)
    knn_filtered = knn_filtered.drop(['name', 'category', 'main_category', 'currency','country'], axis=1)

    #nb_filtered has columns: name and state
    nb_filtered = data_filtered.copy(deep=True)
    nb_filtered = nb_filtered.drop(['alotted_time', 'goal', 'category', 'main_category', 'currency', 'country' ], axis=1)
    
    arrlen = np.vectorize(wordlen)
    print("\nMean Wordcount in Project Name: {}\n".format(np.mean(arrlen( data['name'] ))))

    # train / test
    knn_classifier, knn_accuracy = train_classifier(knn_filtered, "KNN")
    nb_classifier, bayes_accuracy = train_classifier(nb_filtered, "NB")

    # test ensemble classification
    knn_recall, nb_recall, knn_f, nb_f = evaluate_algorithms(knn_classifier, nb_classifier, knn_filtered, nb_filtered)

    print("KNN accuracy", knn_accuracy)
    print("KNN F-Score", knn_f)
    print("KNN Recall Value", knn_recall, end="\n\n")
    print("Bayes accuracy", bayes_accuracy)
    print("Bayes F-Score", nb_f)
    print("Bayes Recall Value", nb_recall, end="\n\n")

    plot_barchart({
            'accuracy': knn_accuracy,
            'recall': knn_recall,
            'fscore': knn_f
        },
        {
            'accuracy': bayes_accuracy,
            'recall': nb_recall,
            'fscore': nb_f
        })

    return

__main__()

