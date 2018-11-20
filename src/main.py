import pandas as pd
import random


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
                "ID",
                "Name",
                "Category",
                "Main_Category",
                "Currency",
                "Deadline",
                "Goal",
                "Launched",
                "Pledged",
                "State",
                "Backers",
                "Country",
                "USD_Pledged"
                ])
    return data


def __main__():
    data = read_in_data('./assets/ks-projects-201801.csv', 379000, 7000)
    print(data)
    return

__main__()

